"use strict";

let graph;
let ws;

let months = ['January', 'February', 'March', 'April', 'Mai', 'June', 'July',
              'August', 'September', 'October', 'November', 'December'];

/*
 * Once the SVG file representing the network is loaded, store the content of
 * its document object in variable svgNetwork.
 */
let svgNetwork;
let svgRoot;
window.addEventListener("load", function() {
  svgNetwork = document.getElementById('svg-network').contentDocument;
  svgRoot = svgNetwork.getElementById(svgParams.svgRootId);
});

/*
 * This class implements an electric network.
 */
class Graph {
  constructor(svgDevices, svgLines, svgStorage, svgParams, devType, pMin, pMax,
              iMax, socMin, socMax) {

    this.nodesN = svgDevices.length;
    this.linesN = svgLines.length;
    this.storageN = svgStorage.length;

    this.devices = new Array(this.nodesN);
    this.lines = new Array(this.linesN);
    this.storage = new Array(this.storageN);

    for (let i = 0; i < this.nodesN; i++) {
      this.devices[i] = new PowerInjection(i, svgDevices[i], devType[i], pMin[i],
          pMax[i], svgParams);
    }
    for (let i = 0; i < this.linesN; i++) {
      this.lines[i] = new TransmissionLine(svgLines[i], iMax[i], svgParams);
    }

    for (let i = 0; i < this.storageN; i++) {
      this.storage[i] = new StorageUnit(svgStorage[i], socMin[i], socMax[i],
          svgParams);
    }
  }

  update(pInjections, iCurrents, socStorage, pBranchFlows, pPotential) {
    let pPotIdx = 0;
    for (let i = 0; i < this.nodesN; i++) {
      if (this.devices[i].devType > 0) {
        this.devices[i].update(pInjections[i], pPotential[pPotIdx]);
        pPotIdx += 1;
      } else {
        this.devices[i].update(pInjections[i], null);
      }
    }
    for (let i = 0; i < this.linesN; i++) {
      this.lines[i].update(iCurrents[i], pBranchFlows[i]);
    }
    for (let i = 0; i < this.storageN; i++) {
      this.storage[i].update(socStorage[i]);
    }
  }
}

class StorageUnit {
  constructor(svgStorage, socMin, socMax, svgParams) {
    this.socMinTick = svgStorage.socMinTick;
    this.socMaxTick = svgStorage.socMaxTick;
    this.socText = svgStorage.socText;
    this.batteryRect = svgStorage.batteryRect;
    this.arrow = svgStorage.arrow;
    this.socMin = socMin;
    this.socMax = socMax;

    this.rectMaxHeight = svgParams.batteryHeight;
    this.socTextDecN = svgParams.socTextDecN;
    this.socUnits = svgParams.socUnits;
    this.socRectColor = svgParams.colors.socColor;

    svgNetwork.getElementById(this.socMinTick).innerHTML = this.socMin;
    svgNetwork.getElementById(this.socMaxTick).innerHTML = this.socMax;
  }

  update(curValue) {
    let barHeight = this.rectMaxHeight * (curValue - this.socMin)
        / (this.socMax - this.socMin);

    svgNetwork.getElementById(this.batteryRect).setAttribute('height', barHeight);
    svgNetwork.getElementById(this.batteryRect).style.fill = this.socRectColor;
    svgNetwork.getElementById(this.socText).innerHTML = curValue.toFixed(this.socTextDecN) + ' '
        + this.socUnits;
  }
}

class TransmissionLine {
  constructor(svgLine, iMax, svgParams)  {
    this.iCurrentRect = svgLine.iCurrentRect;
    this.iCurrentText = svgLine.iCurrentText;
    this.brokenCross = svgLine.brokenCross;
    this.arrows = svgLine.arrows;

    this.iMax = iMax;
    this.iMin = 0;

    this.rectMaxHeight = svgParams.iCurHeight;
    this.iTextDecN = svgParams.iTextDecN;
    this.colorGradient = svgParams.colors.colorGradient;
    this.brokenLineColor = svgParams.colors.brokenLine;

    for (const id of this.brokenCross) {
      svgNetwork.getElementById(id).style.stroke = 'none';
    }

    let rotationOutput = this.initArrowRotation();
    this.rotations = rotationOutput[0];
    this.arrowCenters = rotationOutput[1];
  }

  initArrowRotation() {
    let rotations = Array(this.arrows.length);
    let arrowCenters = Array(this.arrows.length);
    for (let i = 0; i < this.arrows.length; i++) {
      let arrow = this.arrows[i];
      let box = svgNetwork.getElementById(arrow).getBBox();
      let arrowCenter = [box.x + (box.width / 2), box.y + (box.height / 2)];
      let rotation = svgRoot.createSVGTransform();
      rotation.setRotate(0, arrowCenter[0], arrowCenter[1]);
      rotation = svgNetwork.getElementById(arrow).transform.baseVal.appendItem(rotation);

      rotations[i] = rotation;
      arrowCenters[i] = arrowCenter;
    }
    return [rotations, arrowCenters];
  }

  update(iBranch, pBranchFlow) {
    if (iBranch < 0) {
      console.log(iBranch);
    }

    let current_magn = Math.abs(iBranch);
    let barHeight = this.rectMaxHeight * (current_magn - this.iMin)
        / (this.iMax - this.iMin);
    barHeight = Math.min(barHeight, this.rectMaxHeight);
    let percentage = 100 * (current_magn - this.iMin) / (this.iMax - this.iMin);
    let barColor = this.colorGradient[Math.round(Math.min(100, percentage)) - 1];

    svgNetwork.getElementById(this.iCurrentRect).setAttribute('height', barHeight);
    svgNetwork.getElementById(this.iCurrentRect).style.fill = barColor;
    svgNetwork.getElementById(this.iCurrentText).innerHTML = percentage.toFixed(this.iTextDecN) + ' %';

    let crossColor;
    if (percentage > 100) {
      crossColor = this.brokenLineColor;
    } else {
      crossColor = 'none';
    }
    for (const id of this.brokenCross) {
      svgNetwork.getElementById(id).style.stroke = crossColor;
    }

    this.rotateCurrentArrows(pBranchFlow);
  }

  rotateCurrentArrows(pBranchFlow) {
    let rotationAngle = (pBranchFlow < 0) ? 180 : 0;
    for (let i = 0; i < this.arrows.length; i++) {
      this.rotations[i].setRotate(rotationAngle, this.arrowCenters[i][0], this.arrowCenters[i][1]);
    }
  }
}

/*
 * This class implements a single electric device (node) of the network.
 */
class PowerInjection {
  constructor(id, svgDevice, devType, pMin, pMax, svgParams) {
    this.devId = id;
    this.devType = devType;
    this.isVRE = devType > 0;
    this.pMinTick = svgDevice.pMinTick;
    this.pMaxTick = svgDevice.pMaxTick;
    this.pInjRect = svgDevice.pInjRect;
    this.pInjText = svgDevice.pInjText;
    this.arrow = svgDevice.arrow;
    this.pPotentialRect = svgDevice.pPotential;

    this.pMin = 0;
    this.pMax = Math.max(Math.abs(pMin), Math.abs(pMax));

    this.rectMaxHeight = svgParams.pInjHeight;
    this.genRectColor = svgParams.colors.genRectColor;
    this.loadRectColor = svgParams.colors.loadRectColor;
    this.pTextDecN = svgParams.pTextDecN;
    this.pUnits = svgParams.pUnits;
    this.potentialColor = svgParams.potentialColor;

    svgNetwork.getElementById(this.pMinTick).innerHTML = this.pMin;
    svgNetwork.getElementById(this.pMaxTick).innerHTML = this.pMax;

    if (this.isVRE) {
      svgNetwork.getElementById(this.pPotentialRect).style.fill = this.potentialColor;
    }

    let rotationOutput = this.initArrowRotation();
    this.arrowRotation = rotationOutput[0];
    this.arrowCenter = rotationOutput[1];
  }

  update(curValue, potential) {
    let barHeight = this.rectMaxHeight * (Math.abs(curValue) - this.pMin)
        / (this.pMax - this.pMin);
    let barColor;
    if (curValue >= 0) {
      barColor = this.genRectColor;
    } else {
      barColor = this.loadRectColor;
    }
    svgNetwork.getElementById(this.pInjRect).setAttribute('height', barHeight);
    svgNetwork.getElementById(this.pInjRect).style.fill = barColor;
    svgNetwork.getElementById(this.pInjText).innerHTML = curValue.toFixed(this.pTextDecN) + ' ' + this.pUnits;

    if (this.isVRE) {
      let potBarHeight = this.rectMaxHeight * (Math.abs(potential) - this.pMin)
        / (this.pMax - this.pMin);
      svgNetwork.getElementById(this.pPotentialRect).setAttribute('height', potBarHeight);
    }

    this.rotateCurrentArrow(curValue);
  }

  initArrowRotation() {
    let box = svgNetwork.getElementById(this.arrow).getBBox();
    let arrowCenter = [box.x + (box.width / 2), box.y + (box.height / 2)];
    let rotation = svgRoot.createSVGTransform();
    rotation.setRotate(0, arrowCenter[0], arrowCenter[1]);
    rotation = svgNetwork.getElementById(this.arrow).transform.baseVal.appendItem(rotation);
    return [rotation, arrowCenter];
  }

  rotateCurrentArrow(curValue) {
    let rotationAngle;
    if (this.devType < 0) {
      rotationAngle = (curValue > 0) ? 180 : 0;
    } else {
      rotationAngle = (curValue < 0) ? 180 : 0;
    }
    this.arrowRotation.setRotate(rotationAngle, this.arrowCenter[0], this.arrowCenter[1]);
  }
}


/*
 * This function initializes the WebSocket connection to the server. More
 *  specifically, it defines functions to handle WebSocket events.
 */
function init() {
  // Connect to Web Socket server.
  ws = new WebSocket("ws://127.0.0.1:9001/");

  // When connection with the WebSocket server is established.
  ws.onopen = function() {};

  // When a message from the WebSocket server is received.
  ws.onmessage = function(e) {
    let messageJson = JSON.parse(e.data);
    switch (messageJson.messageLabel) {
      case 'init':
        graph = new Graph(svgDevices, svgLines, svgStorageUnits, svgParams,
                    messageJson.deviceType, messageJson.pMin, messageJson.pMax,
                    messageJson.iMax, messageJson.socMin, messageJson.socMax);
        break;
      case 'update':
        graph.update(messageJson.pInjections, messageJson.iCurrents,
                     messageJson.socStorage, messageJson.pBranchFlows,
                     messageJson.pPotential);
        update_calendar(messageJson.time[0], messageJson.time[1]);
        update_clock(messageJson.time[2], messageJson.time[3]);
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

function update_calendar(month, day) {
  document.getElementById('month').innerHTML = months[month - 1];
  document.getElementById('day').innerHTML = day;
}

