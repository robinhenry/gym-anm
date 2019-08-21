import os
from gym_smartgrid import RENDERING_FOLDER, WEB_FILES


def sample_action(np_random, action_space):
    """
    Sample a random action from the action space.

    Parameters
    ----------
    np_random : numpy.random.RandomState
        The random seed to use. This should be the one used by the environment.
    action_space : gym.spaces.Tuple
        The action space of the environment.

    Returns
    -------
    gym.spaces.Tuple
        An action randomly selected from the action space.
    """

    actions = []
    for space in action_space:
        a = np_random.uniform(space.low, space.high, space.shape)
        actions.append(a)

    return tuple(actions)


def write_html(svg_data):
    """
    Update the index.html file used for rendering the environment state.

    Parameters
    ----------
    svg_data : dict of {str : str}
        The paths to the SVG data needed for the visualization, with keys
        {'labels', 'network'}.
    """

    s = """<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="css/styles.css">
    <script src="js/init.js"></script>
    <script src="js/devices.js"></script>
    <script src="js/graph.js"></script>
    <script src="{0}"></script>
    <script src="js/calendar.js"></script>
    <title>SmartGrid-gym</title>
</head>

<body onload="init();">

    <header></header>

    <object id="svg-network" data="{1}"
            type="image/svg+xml" class="network">
    </object>

    <aside>
        <div class="calendar">
            <time datetime="2014-09-20" class="icon">
                <strong id="month"></strong>
                <span id="day"></span>
            </time>
        </div>
        
        <div class="clock">
            <canvas id="clock" width="100" height="100"></canvas>
        </div>
    </aside>
    
    <div class="legend">
        <p> hello </p>
    </div>
    
    <script src="js/clock.js"></script>

</body>
</html>

    """.format(svg_data['labels'], svg_data['network'])

    html_file = os.path.join(RENDERING_FOLDER, WEB_FILES['index'])

    with open(html_file, 'w') as f:
        f.write(s)
