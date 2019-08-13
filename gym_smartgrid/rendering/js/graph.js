"use strict";

/*
 * This class implements an electric network.
 */
class Graph {
  /**
   * @param {Array.<Object.<string, string>>} svgDevices The SVG tags of the
   * elements used for visualizing electric devices.
   * @param {Array.<Object.<string, string>>} svgLines The SVg tags of the
   * elements used for visualizing transmission lines.
   * @param {Array.<Object.<string, string>>} svgStorage The SVg tags of the
   * elements used for visualizing storage units.
   * @param {Array.<Object>} svgParams General parameters for the visualization.
   * @param {Array.<number>} devType The device type of each device (-1 = load,
   * 0 = slack, 1 = power plant, 2 = wind, 3 = solar, 4 = storage).
   * @param {Array.<number>} pMin The minimum real power injection of each
   * device.
   * It should be non-positive for all loads and zero for devices that inject
   * both positive and negative amount of real power (e.g. slack or storage).
   * @param {Array.<number>} pMax The maximum real power injection of each
   * device.
   * @param {Array.<number>} iMax The maximum current magnitude in each
   * transmission line.
   * @param {Array.<number>} socMin The minimum state of charge of each storage
   * unit.
   * @param {Array.<number>} socMax The maximum state of charge of each storage
   * unit
   */
  constructor(svgDevices, svgLines, svgStorage, svgParams, devType, pMin, pMax,
              iMax, socMin, socMax) {

    /** The number of nodes in the network. */
    this.nodesN = svgDevices.length;

    /** The number of transmission lines in the network. */
    this.linesN = svgLines.length;

    /** The number of storage units in the network. */
    this.storageN = svgStorage.length;

    /** The devices connected to the grid. */
    this.devices = new Array(this.nodesN);

    /** The transmission lines. */
    this.lines = new Array(this.linesN);

    /** The storage units. */
    this.storage = new Array(this.storageN);

    // Instantiate all devices.
    for (let i = 0; i < this.nodesN; i++) {
      this.devices[i] = new PowerInjection(i, svgDevices[i], devType[i], pMin[i],
          pMax[i], svgParams);
    }

    // Instantiate all transmission lines.
    for (let i = 0; i < this.linesN; i++) {
      this.lines[i] = new TransmissionLine(svgLines[i], iMax[i], svgParams);
    }

    // Instantiate all storage units.
    for (let i = 0; i < this.storageN; i++) {
      this.storage[i] = new StorageUnit(svgStorage[i], socMin[i], socMax[i],
          svgParams);
    }
  }

  /**
   * Update the state of the network.
   * @param {Array.<number>} pInjections The new real power injection at each
   * device.
   * @param {Array.<number>} iCurrents The new branch current magnitude in
   * each transmission line.
   * @param {Array.<number>} socStorage The new state of charge of each storage
   * unit.
   * @param {Array.<number>} pBranchFlows The new real power flow in each
   * transmission line.
   * @param {Array.<number>} pPotential The potential real power injection from
   * each VRE generator before curtailment.
   */
  update(pInjections, iCurrents, socStorage, pBranchFlows, pPotential) {

    // Update all devices.
    let pPotIdx = 0;
    for (let i = 0; i < this.nodesN; i++) {
      if (this.devices[i].isVRE) {
        this.devices[i].update(pInjections[i], pPotential[pPotIdx]);
        pPotIdx += 1;
      } else {
        this.devices[i].update(pInjections[i], null);
      }
    }

    // Update all transmission lines.
    for (let i = 0; i < this.linesN; i++) {
      this.lines[i].update(iCurrents[i], pBranchFlows[i]);
    }

    // Update all storage units.
    for (let i = 0; i < this.storageN; i++) {
      this.storage[i].update(socStorage[i]);
    }
  }
}