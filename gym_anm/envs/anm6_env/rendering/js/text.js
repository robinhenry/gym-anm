"use strict";

function addTitle(svgParams, title) {
  svgNetwork.getElementById(svgParams.title).innerHTML = title;
}

function networkCollapsed(svgParams, collapsed) {
  // Add or remove the check mark.
  if (collapsed) {
    svgNetwork.getElementById(svgParams.networkCollapsedCheckMark).innerHTML = '\u2713';
  } else {
    svgNetwork.getElementById(svgParams.networkCollapsedCheckMark).innerHTML = '';
  }

  // Make 'voltage collapse' red to attract attention if it did.
  let c;
  if (collapsed) {
    c = '#ff0000';
  } else {
    c = '#000000';
  }
  for (let i = 0; i < svgParams.networkCollapsedGroup.length; i++) {
    svgNetwork.getElementById(svgParams.networkCollapsedGroup[i]).style.fill = c;
  }
}
