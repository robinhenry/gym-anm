"use strict";

/**
 * A single electric device (node) of the network.
 */
class PowerInjection {
  /**
   * @param {number} id The unique ID of the device.
   * @param {Object.<string, string>} svgDevice A dictionary containing SVG tags
   * related to this device.
   * @param {number} devType The device type (-1 = load, 0 = slack, 1 = power
   * plant, 2 = wind, 3 = solar, 4 = storage).
   * @param {number} pMin The minimum real power injection.
   * @param {number} pMax The maximum real power injection.
   * @param {Array.<Object>} svgParams General parameters for the visualization.
   */
  constructor(id, svgDevice, devType, pMin, pMax, svgParams) {
    this.devId = id;
    this.devType = devType;
    this.isVRE = (devType === 2) || (devType === 3);
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
    this.potentialColor = svgParams.colors.potGenRect;

    // Change the minimum and maximum power injection ticks.
    svgNetwork.getElementById(this.pMinTick).innerHTML = this.pMin;
    svgNetwork.getElementById(this.pMaxTick).innerHTML = this.pMax;

    // If the device is a curtailed VRE generator.
    if (this.isVRE) {
      // Set height of potential P injection bar to 0.
      svgNetwork.getElementById(this.pPotentialRect).setAttribute('height', 0);
      // Change the color of the potential P injection.
      svgNetwork.getElementById(this.pPotentialRect).style.fill = this.potentialColor;
      // Get the original y-position of the potential P bar.
      this.pPotentialY = parseFloat(svgNetwork.getElementById(this.pPotentialRect).getAttribute("y"));
    }

    // Initialize the arrow indicating the direction of power flow.
    let rotationOutput = initArrowDirection(this.arrow);
    this.arrowRotation = rotationOutput[0];
    this.arrowCenter = rotationOutput[1];
  }

  /**
   * Update the displayed information of the device.
   * @param {number} curValue The new real power injection.
   * @param {number} potential The potential real power injection before
   * injection, if the device is a VRE generator.
   */
  update(curValue, potential) {

    // Compute the height of the color bar representing power injection.
    let barHeight = this.rectMaxHeight * (Math.abs(curValue) - this.pMin)
        / (this.pMax - this.pMin);

    // Choose the color of the injection bar based on the sign of the injection.
    let barColor;
    if (curValue >= 0) {
      barColor = this.genRectColor;
    } else {
      barColor = this.loadRectColor;
    }

    // Update the height and the color of the injection bar.
    svgNetwork.getElementById(this.pInjRect).setAttribute('height', barHeight);
    svgNetwork.getElementById(this.pInjRect).style.fill = barColor;

    // Update the text below the injection bar.
    svgNetwork.getElementById(this.pInjText).innerHTML =
        curValue.toFixed(this.pTextDecN) + ' ' + this.pUnits;

    // Update the height, position and color of the potential generation bar.
    if (this.isVRE) {
      let potBarHeight;
      let deltaY;
      if (potential > curValue) {
        potBarHeight = this.rectMaxHeight * (Math.abs(potential) - this.pMin)
            / (this.pMax - this.pMin) - barHeight;
        deltaY = barHeight;
      } else {
        potBarHeight = 0;
        deltaY = 0;
      }
      svgNetwork.getElementById(this.pPotentialRect).setAttribute('height', potBarHeight);
      svgNetwork.getElementById(this.pPotentialRect).setAttribute('y', deltaY + this.pPotentialY);
    }

    // Update the direction of the power flow arrow.
    this.rotateCurrentArrow(curValue);
  }

  /**
   * Update the direction of the power flow arrow linked to the device.
   * @param {number} curValue The new real power injection.
   */
  rotateCurrentArrow(curValue) {
    let rotationAngle;

    // Choose the angle of rotation, based on the sign of the power injection.
    if (this.devType < 0) {
      rotationAngle = (curValue > 0) ? 180 : 0;
    } else {
      rotationAngle = (curValue < 0) ? 180 : 0;
    }

    // Update the arrow direction by rotating it by 0 or 180 degrees.
    this.arrowRotation.setRotate(rotationAngle, this.arrowCenter[0], this.arrowCenter[1]);
  }
}

/**
 * A transmission line of an electrical network.
 */
class TransmissionLine {
  /**
   * @param {Object.<string, string>} svgLine A dictionary containing SVG tags
   * related to this transmission line.
   * @param {number} iMax The current rating of the transmission line.
   * @param {Array.<Object>} svgParams General parameters for the visualization.
   */
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

    // Remove the crosses indicating line overheating.
    for (const id of this.brokenCross) {
      svgNetwork.getElementById(id).style.stroke = 'none';
    }

    // Initialize the arrow indicating the direction of power flow.
    let rotationOutput = this.initArrowRotation();
    this.rotations = rotationOutput[0];
    this.arrowCenters = rotationOutput[1];
  }

  /**
   * Initialize the rotation angles of all arrows on the transmission line.
   * @returns {[any[], any[]]}
   */
  initArrowRotation() {

    let rotations = Array(this.arrows.length);
    let arrowCenters = Array(this.arrows.length);

    // Iterate over all arrows on the transmission line.
    for (let i = 0; i < this.arrows.length; i++) {
      let out = initArrowDirection(this.arrows[i]);
      rotations[i] = out[0];
      arrowCenters[i] = out[1];
    }

    return [rotations, arrowCenters];
  }

  /**
   * Update the information displayed on the transmission line.
   * @param sFlow
   */
  update(sFlow) {

    // Compute the new height of the power flow bar.
    let current_magn = Math.abs(sFlow);
    let barHeight = this.rectMaxHeight * (current_magn - this.iMin)
        / (this.iMax - this.iMin);
    barHeight = Math.min(barHeight, this.rectMaxHeight);

    // Compute the % to be displayed next to the bar.
    let percentage = 100 * (current_magn - this.iMin) / (this.iMax - this.iMin);
    let barColor = this.colorGradient[Math.round(Math.min(100, percentage)) - 1];

    // Update the height and color of the bar.
    svgNetwork.getElementById(this.iCurrentRect).setAttribute('height', barHeight);
    svgNetwork.getElementById(this.iCurrentRect).style.fill = barColor;

    // Update the text next to the bar.
    svgNetwork.getElementById(this.iCurrentText).innerHTML = percentage.toFixed(this.iTextDecN) + ' %';

    // Get the color to fill the cross over the line.
    let crossColor;
    if (percentage > 100) {
      // If the current flow is over the line rate.
      crossColor = this.brokenLineColor;
    } else {
      // If the current flow is below the line rate.
      crossColor = 'none';
    }

    // Update the stroke of each bar of the line cross.
    for (const id of this.brokenCross) {
      svgNetwork.getElementById(id).style.stroke = crossColor;
    }

    // Update the direction of each power flow arrow on the line.
    this.rotateCurrentArrows(sFlow);
  }

  /**
   * Update the direction of the power flow arrow linked to the device.
   * @param {number} pBranchFlow The new real power injection.
   */
  rotateCurrentArrows(pBranchFlow) {
    let rotationAngle = (pBranchFlow < 0) ? 180 : 0;

    // Iterate over all arrows on the line and update it.
    for (let i = 0; i < this.arrows.length; i++) {
      this.rotations[i].setRotate(rotationAngle, this.arrowCenters[i][0], this.arrowCenters[i][1]);
    }
  }
}

/**
 * A distributed energy storage unit connected to an electric network.
 */
class StorageUnit {
  /**
   * @param {Object.<string, string>} svgStorage A dictionary containing SVG tags
   * related to this storage unit.
   * @param {number} socMin Minimum state of charge of the storage unit.
   * @param {number} socMax Maximum state of charge of the storage unit.
   * @param {Array.<Object>} svgParams General parameters for the visualization.
   */
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

    // Change the minimum and maximum state of charge ticks.
    svgNetwork.getElementById(this.socMinTick).innerHTML = this.socMin;
    svgNetwork.getElementById(this.socMaxTick).innerHTML = this.socMax;
  }

  /**
   * Update the displayed information about the state of the storage unit.
   * @param {number} curValue The new state of charge of the storage unit.
   */
  update(curValue) {

    // Compute the new height of the storage level.
    let barHeight = this.rectMaxHeight * (curValue - this.socMin)
        / (this.socMax - this.socMin);

    // Update the height and color of the storage level.
    svgNetwork.getElementById(this.batteryRect).setAttribute('height', barHeight);
    svgNetwork.getElementById(this.batteryRect).style.fill = this.socRectColor;

    // Update the text displaying the state of charge.
    svgNetwork.getElementById(this.socText).innerHTML =
        curValue.toFixed(this.socTextDecN) + ' ' + this.socUnits;
  }
}


/**
 * Initialize the variables used to orientate the power flow arrow.
 * @returns {[SVGTransform, *[]]}
 */
function initArrowDirection(arrowLabel) {
      // Get the center of the arrow SVG element.
    let box = svgNetwork.getElementById(arrowLabel).getBBox();
    let arrowCenter = [box.x + (box.width / 2), box.y + (box.height / 2)];

    // Create a rotation transformation.
    let rotation = svgRoot.createSVGTransform();
    rotation.setRotate(0, arrowCenter[0], arrowCenter[1]);
    rotation = svgNetwork.getElementById(arrowLabel).transform.baseVal.appendItem(rotation);

    return [rotation, arrowCenter];
}