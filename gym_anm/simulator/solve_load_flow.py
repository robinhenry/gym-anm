import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, hstack as shstack, vstack as svstack


def solve_pfe_newton_raphson(simulator, xtol=1e-5):
    """
    Solve load flow using the Newton-Raphson iterative algorithm.

    This function directly updates all the bus and branch variables in the
    `simulator` object.

    Parameters
    ----------
    simulator : simulator.Simulator
        The electricity network simulator.
    xtol : float, optional
        The desired tolerance in the solution.

    Returns
    -------
    V : numpy.ndarray
        The (N,) complex bus voltage vector.
    stable : bool
        True if the algorithm has converged for the desired tolerance; False
        otherwise.
    """

    # Collect P and Q nodal power injections (except slack bus).
    p, q = [], []
    for i in range(simulator.N_bus):
        if i in simulator.buses.keys():
            if not simulator.buses[i].is_slack:
                p.append(simulator.buses[i].p)
                q.append(simulator.buses[i].q)
        else:
            p.append(0)
            q.append(0)

    # Construct initial guess for nodal V.
    v_guess = [0.] * (simulator.N_bus - 1) + [1.] * (simulator.N_bus - 1)
    v_guess = np.array(v_guess)

    # Solve power flow equations using Netwon-Raphson method.
    v, n_iter, diff, converged = \
        _newton_raphson_sparse(v_guess, np.array(p), np.array(q),
                               simulator.Y_bus, x_tol=xtol)

    # Check if a stable solution has been reached.
    stable = True if converged and diff <= xtol else False

    # Reconstruct complex V nodal vector.
    V = _construct_v_from_guess(v)

    # Compute nodal current injections as I = YV.
    I = np.dot(simulator.Y_bus.toarray(), V)

    # Update simulator.
    for i, bus in enumerate(simulator.buses.values()):
        bus.v = V[i]
        bus.i = I[i]

        # Update slack bus power injection (set to np.inf if nan).
        if bus.is_slack:
            s = V[0] * np.conj(I[0])
            bus.p = s.real if not np.isnan(s.real) else np.inf
            bus.q = s.imag if not np.isnan(s.imag) else np.inf

    # Update slack device injections.
    for dev in simulator.devices.values():
        if dev.is_slack:
            dev.p = simulator.buses[dev.bus_id].p
            dev.q = simulator.buses[dev.bus_id].q

    # Compute branch I_{ij}, P_{ij}, and Q_{ij} flows.
    for branch in simulator.branches.values():
        v_f = simulator.buses[branch.f_bus].v
        v_t = simulator.buses[branch.t_bus].v
        branch.compute_currents(v_f, v_t)
        branch.compute_power_flows(v_f, v_t)

    return V, stable


def _f(guess, p, q, Y):
    """
    The function f(x) = 0 to be solved.

    Parameters
    ----------
    guess : numpy.ndarray
        The current v_guess for the roots of f(x), of size 2*(N-1), where elements
        [0,...,N-2] are the nodal voltage angles \theta_i and [N-1,...,2(N-1)]
        the nodal voltage magnitudes |V_i|. The slack bus variables are
        excluded.
    p : numpy.ndarray
        The vector of nodal active power injections of size (N-1), excluding the
        slack bus.
    q : numpy.ndarray
        The vector of nodal reactive power injections of size (N-1), excluding
        the slack bus.
    Y : scipy.sparse.csc_matrix
        The nodal admittance matrix of shape (N, N) as a sparse matrix.

    Returns
    -------
    F : numpy.ndarray
        The value f(x), where F[0,...N-2] is the real part of f(x) and
        F[N-1,...,2(N-1)] the imaginary part of f(x).
    """

    # Construct nodal voltage vector V, setting V_slack = 1+0j.
    v = _construct_v_from_guess(guess)

    # Compute the difference between V (YV)^* and the real power injections S.
    s = p + 1j * q
    mismatch = (v * np.conj(Y * v))[1:] - s

    F = np.concatenate((mismatch.real, mismatch.imag))

    return F


def _dfdx(guess, Y):
    """
    Compute the Jacobian matrix.

    Parameters
    ----------
    guess : numpy.ndarray
        The current v_guess for the roots of f(x), of size 2*(N-1), where elements
        [0,...,N-2] are the nodal voltage angles \theta_i and [N-1,...,2(N-1)]
        the nodal voltage magnitudes |V_i|. The slack bus variables are
        excluded.
    Y : scipy.sparse.csc_matrix
        The nodal admittance matrix of shape (N, N) as a sparse matrix.

    Returns
    -------
    J : scipy.sparse.csr_matrix
        The Jacobian matrix as a sparse matrix.
    """

    # Construct nodal voltage vector V, setting V_slack = 1+0j.
    v = _construct_v_from_guess(guess)

    index = np.array(range(len(v)))

    # Construct sparse diagonal matrices.
    v_diag = csr_matrix((v, (index, index)))
    v_norm_diag = csr_matrix((v / abs(v), (index, index)))
    i_diag = csr_matrix((Y * v, (index, index)))

    # Construct the Jacobian matrix.
    dS_dVa = 1j * v_diag * np.conj(i_diag - Y * v_diag)  # dS / d \theta
    dS_dVm = v_norm_diag * np.conj(i_diag) + v_diag * np.conj(Y * v_norm_diag)  # dS / d |V|

    J00 = dS_dVa[1:, 1:].real
    J01 = dS_dVm[1:, 1:].real
    J10 = dS_dVa[1:, 1:].imag
    J11 = dS_dVm[1:, 1:].imag

    J = svstack([shstack([J00, J01]), shstack([J10, J11])], format='csr')

    return J


def _construct_v_from_guess(guess):
    assert len(guess) % 2 == 0
    n = int(len(guess) / 2)
    v_nonslack = guess[n:] * np.exp(1j * guess[:n])
    v = np.concatenate((np.array([1+0j]), v_nonslack))

    return v


def _newton_raphson_sparse(v_guess, p, q, Y, x_tol=1e-10, lim_iter=100):
    """
    Solve f(x) = 0 with initial v_guess for x and dfdx(x).

    dfdx(x) should return a sparse Jacobian. Terminate if error on norm of
    f(x) is < x_tol or there were more than lim_iter iterations.

    Parameters
    ----------
    v_guess : numpy.ndarray
        The current v_guess for the roots of f(x), of size 2*(N-1), where elements
        [0,...,N-2] are the nodal voltage angles \theta_i and [N-1,...,2(N-1)]
        the nodal voltage magnitudes |V_i|. The slack bus variables are
        excluded.
    p : numpy.ndarray
        The vector of nodal active power injections of size (N-1), excluding the
        slack bus.
    q : numpy.ndarray
        The vector of nodal reactive power injections of size (N-1), excluding
        the slack bus.
    Y : scipy.sparse.csc_matrix
        The nodal admittance matrix of shape (N, N) as a sparse matrix.
    x_tol : float
        The tolerance to achieve in the solution.
    lim_iter : int
        The maximum number of iterations.

    Returns:
    v_guess : numpy.ndarray
        The final x to which the algorithm converged.
    n_iter : int
        The number of iterations.
    diff : float
        The final error on the norm of f(x).
    converged : bool
        True if the algorithm converged; False otherwise.
    """

    n_iter = 0
    F = _f(v_guess, p, q, Y)
    diff = norm(F, np.Inf)

    while diff > x_tol and n_iter < lim_iter:
        n_iter += 1
        v_guess = v_guess - spsolve(_dfdx(v_guess, Y), F)
        F = _f(v_guess, p, q, Y)
        diff = norm(F, np.Inf)

    converged = False if np.isnan(diff) else True

    return v_guess, n_iter, diff, converged
