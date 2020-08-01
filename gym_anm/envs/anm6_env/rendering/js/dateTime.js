"use strict";

let months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December'];

function updateDate(svgParams, month, day) {
  svgNetwork.getElementById(svgParams.date).innerHTML =
      months[month - 1] + ' ' + day;
}

function updateTime(svgParams, hour, min) {
  let new_hour;
  if (hour < 10) {
    new_hour = '0' + hour;
  } else {
    new_hour = hour;
  }

  let new_min;
  if (min < 10) {
    new_min = '0' + min;
  } else {
    new_min = min;
  }

  svgNetwork.getElementById(svgParams.time).innerHTML =
      new_hour + ':' + new_min;
}

function updateYearCount(svgParams, yearCount) {
  svgNetwork.getElementById(svgParams.yearCount).innerHTML = 'Year count: ' + yearCount;

}