"use strict";

/*
 * This class implements an electric network.
 */
class Graph {
  /**
   * @param {Array.<Object.<string, string>>} svgDevices The SVG tags of the
   * elements used for visualizing electric devices.
   * @param {Array.<Object.<string, string>>} svgLines The SVG tags of the
   * elements used for visualizing transmission lines.
   * @param {Array.<Object.<string, string>>} svgStorage The SVG tags of the
   * elements used for visualizing storage units.
   * @param {Array.<Object.<string, string>>} svgBuses The SVG tags of the
   * elements used for visualizing buses.
   * @param {Array.<Object>} svgParams General parameters for the visualization.
   * @param {Array.<number>} devType The device type of each device (-1 = load,
   * 0 = slack, 1 = power plant, 2 = wind, 3 = solar, 4 = storage).
   * @param {Array.<number>} pMax The maximum absolute real power injection of
   * each device.
   * @param {Array.<number>} qMax The maximum absolute reactive power injection
   * of each device.
   * @param {Array.<number>} sRate The maximum current magnitude in each
   * transmission line.
   * @param {Array.<number>} vMagnMin The minimum voltage magnitude allowed at
   * each bus.
   * @param {Array.<number>} vMagnMax The maximum voltage magnitude allowed at
   * each bus.
   * @param {Array.<number>} socMax The maximum state of charge of each storage
   * unit
   */
  constructor(svgDevices, svgLines,
              svgStorage, svgBuses, svgParams,
              devType, pMax, qMax, sRate,
              vMagnMin, vMagnMax, socMax) {

    /** The number of devices in the network. */
    this.nodesN = svgDevices.length;

    /** The number of buses in the network. */
    this.busesN = svgBuses.length;

    /** The number of transmission lines in the network. */
    this.linesN = svgLines.length;

    /** The number of storage units in the network. */
    this.storageN = svgStorage.length;

    /** The devices connected to the grid. */
    this.devices = new Array(this.nodesN);

    /** The buses. */
    this.buses = new Array(this.busesN);

    /** The transmission lines. */
    this.lines = new Array(this.linesN);

    /** The storage units. */
    this.storage = new Array(this.storageN);

    // Instantiate all devices.
    for (let i = 0; i < this.nodesN; i++) {
      this.devices[i] = new PowerInjection(i, svgDevices[i], devType[i], 0,
          pMax[i], 0, qMax[i], svgParams);
    }

    // Instantiate all transmission lines.
    for (let i = 0; i < this.linesN; i++) {
      this.lines[i] = new TransmissionLine(svgLines[i], sRate[i], svgParams);
    }

    // Instantiate all storage units.
    for (let i = 0; i < this.storageN; i++) {
      this.storage[i] = new StorageUnit(svgStorage[i], 0, socMax[i],
          svgParams);
    }

    // Instantiate all buses.
    for (let i = 0; i < this.busesN; i++) {
      this.buses[i] = new Bus(svgBuses[i], vMagnMin[i], vMagnMax[i], svgParams);
    }
  }

  /**
   * Update the state of the network.
   * @param {Array.<number>} pInjections The new real power injection at each
   * device.
   * @param {Array.<number>} qInjections The new reactive power injection at
   * each device.
   * @param {Array.<number>} sFlows The new branch apparent flow in
   * each transmission line.
   * @param {Array.<number>} socStorage The new state of charge of each storage
   * unit.
   * @param {Array.<number>} pPotential The potential real power injection from
   * each VRE generator before curtailment.
   * @param {Array.<number>} vMagn The new voltage magnitude at each bus.
   */
  update(pInjections, qInjections, sFlows,
         socStorage, pPotential, vMagn) {

    // Update all devices.
    let pPotIdx = 0;
    for (let i = 0; i < this.nodesN; i++) {
      if (this.devices[i].isNonSlackGen) {
        this.devices[i].update(pInjections[i], qInjections[i], pPotential[pPotIdx]);
        pPotIdx += 1;
      } else {
        this.devices[i].update(pInjections[i], qInjections[i], null);
      }
    }

    // Update all transmission lines.
    for (let i = 0; i < this.linesN; i++) {
      this.lines[i].update(sFlows[i]);
    }

    // Update all storage units.
    for (let i = 0; i < this.storageN; i++) {
      this.storage[i].update(socStorage[i]);
    }

    // Update all buses.
    for (let i = 0; i < this.busesN; i++) {
      this.buses[i].update(vMagn[i]);
    }
  }
}