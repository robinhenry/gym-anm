"use strict";

let graph;

/*
 * Once the SVG file representing the network is loaded, store the content of
 * its document object in variable svgNetwork.
 */
let svgNetwork;
window.addEventListener("load", function() {
  svgNetwork = document.getElementById('svg-network').contentDocument;
});

/*
 * This class implements an electric network.
 */
class Graph {
  constructor(svgDevices, svgLines, svgStorage, svgParams, pMin, pMax, iMax,
              socMin, socMax) {

    this.nodesN = svgDevices.length;
    this.linesN = svgLines.length;
    this.storageN = svgStorage.length;

    this.devices = new Array(this.nodesN);
    this.lines = {};
    this.storage = new Array(this.storageN);

    for (let i = 0; i < this.nodesN; i++) {
      this.devices[i] = new PowerInjection(svgDevices[i], pMin[i], pMax[i],
                                           svgParams.pInjHeight);
    }
    for (let i = 0; i < this.linesN; i++) {
      let key = '(' + String(svgLines[i].nodePair) + ')';
      this.lines[String(key)] =
          new TransmissionLine(svgLines[i], iMax[String(key)], svgParams.iCurHeight);
    }

    for (let i = 0; i < this.storageN; i++) {
      this.storage[i] = new StorageUnit(svgStorage[i], socMin[i], socMax[i]);
    }
  }

  update(pInjections, iCurrents, socStorage) {
    for (let i = 0; i < this.nodesN; i++) {
      this.devices[i].update(pInjections[i]);
    }
    for (const [key, value] of Object.entries(iCurrents)) {
      debugger;
      this.lines[key].update(iCurrents[key]);
    }
    for (let i = 0; i < this.storageN; i++) {
      this.storage[i].update(socStorage[i]);
    }
  }
}

class StorageUnit {
  constructor(svgStorage, socMin, socMax) {
    this.socMinTick = svgStorage.socMinTick;
    this.socMaxTick = svgStorage.socMaxTick;
    this.socText = svgStorage.socText;
    this.batteryIcon = svgStorage.batteryIcon;
    this.socMin = socMin;
    this.socMax = socMax;

    svgNetwork.getElementById(this.socMinTick).innerHTML = this.socMin;
    svgNetwork.getElementById(this.socMaxTick).innerHTML = this.socMax;
    svgNetwork.getElementById(this.socText).innerHTML = '0 kWh';
  }

  update(curValue) {
    svgNetwork.getElementById(this.socText).innerHTML = curValue + ' kWh';
  }
}

class TransmissionLine {
  constructor(svgLine, iMax, rectMaxHeight)  {
    this.nodePair = svgLine.nodePair;
    this.iCurrentRect = svgLine.iCurrentRect;
    this.iCurrentText = svgLine.iCurrentText;
    this.iMax = iMax;
    this.rectMaxHeight = rectMaxHeight;
    svgNetwork.getElementById(this.iCurrentRect).setAttribute('height', 0);
    svgNetwork.getElementById(this.iCurrentText).innerHTML = '0 %';
  }

  update(curValue) {
    let barHeight = this.rectMaxHeight * (curValue - this.pMin)
        / (this.pMax - this.pMin);
    barHeight = Math.min(barHeight, this.rectMaxHeight);
    let percentage = 100 * (curValue - this.pMin) / (this.pMax - this.pMin);

    svgNetwork.getElementById(this.iCurrentRect).setAttribute('height', barHeight);
    svgNetwork.getElementById(this.iCurrentText).innerHTML = percentage + ' %';
  }
}

/*
 * This class implements a single electric device (node) of the network.
 */
class PowerInjection {
  constructor(svgDevice, pMin, pMax, rectMaxHeight) {
    this.pMinTick = svgDevice.pMinTick;
    this.pMaxTick = svgDevice.pMaxTick;
    this.pInjRect = svgDevice.pInjRect;
    this.pInjText = svgDevice.pInjText;
    this.pMin = pMin;
    this.pMax = pMax;
    this.rectMaxHeight = rectMaxHeight;

    svgNetwork.getElementById(this.pMinTick).innerHTML = this.pMin;
    svgNetwork.getElementById(this.pMaxTick).innerHTML = this.pMax;
    svgNetwork.getElementById(this.pInjRect).setAttribute('height', 0);
    svgNetwork.getElementById(this.pInjText).innerHTML = '0 kW';
  }

  update(curValue) {
    let barHeight = this.rectMaxHeight * (curValue - this.pMin)
        / (this.pMax - this.pMin);

    svgNetwork.getElementById(this.pInjRect).setAttribute('height', barHeight);
    svgNetwork.getElementById(this.pInjText).innerHTML = curValue + ' kW';
  }
}


/*
 * This function initializes the WebSocket connection to the server. More
 *  specifically, it defines functions to handle WebSocket events.
 */
function init() {

  // Connect to Web Socket server.
  let ws = new WebSocket("ws://localhost:9001/");

  // When connection with the WebSocket server is established.
  ws.onopen = function() {};

  // When a message from the WebSocket server is received.
  ws.onmessage = function(e) {
    let messageJson = JSON.parse(e.data);

    switch (messageJson.messageLabel) {
      case 'init':
        graph = new Graph(svgDevices, svgLines, svgStorageUnits, svgParams,
                    messageJson.pMin, messageJson.pMax, messageJson.iMax,
                    messageJson.socMin, messageJson.socMax);
        break;
      case 'update':
        graph.update(messageJson.pInjections, messageJson.iCurrents,
                     messageJson.socStorage);
        break;
    }
  };

  // When the connection with the WebSocket server is closed.
  ws.onclose = function() {};

  // When an error message is received from the WebSocket server.
  ws.onerror = function(e) {
    console.log(e);
  };
}
