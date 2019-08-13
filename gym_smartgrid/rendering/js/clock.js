"use strict";

// Constants used to place the clock in the right top corner of the screen.
let x_center = 50;
let y_center = 50;
let radius = 40;
let clock_linewidth = 3;
let hour_pointer_len = 20;
let min_pointer_len = 35;
let pointer_width = 2;
let number_radius = radius - 8;
let number_center_x = x_center - 3;
let number_center_y = y_center + 4;
let number_font = "10px Verdana";

// Extract the HTML clock element.
let clock = document.getElementById("clock");
let ctx = clock.getContext("2d");

/**
 * Update the displayed time on the clock.
 * @param {number} h The hour.
 * @param {number} m The minutes.
 */
function update_clock(h, m) {
  ctx.beginPath();
  ctx.fillStyle = "white";
  ctx.arc(x_center, y_center, radius, 0, Math.PI * 2, true);
  ctx.fill();
  ctx.strokeStyle = "black";
  ctx.lineWidth = clock_linewidth;
  ctx.stroke();

  drawNumber();

  drawPointer(360 * (h / 12) + (m / 60) * 30 - 90, hour_pointer_len, "black", pointer_width);
  drawPointer(360 * (m / 60) - 90, min_pointer_len, "black", pointer_width);
}

/** Draw the numbers on the clock. */
function drawNumber() {
  for (let n = 0; n < 12; n++) {
    let d = -60;
    let num = n + 1;
    let str = num.toString();
    let dd = Math.PI / 180 * (d + n * 30);
    let tx = Math.cos(dd) * number_radius + number_center_x;
    let ty = Math.sin(dd) * number_radius + number_center_y;

    ctx.font = number_font;
    ctx.fillStyle = "black";
    ctx.fillText(str, tx, ty);
  }
}

/** Draw the hour and minute pointers on the clock. */
function drawPointer(deg, len, color, w) {
  let rad = (Math.PI / 180 * deg);
  let x1 = x_center + Math.cos(rad) * len;
  let y1 = y_center + Math.sin(rad) * len;

  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = w;
  ctx.moveTo(x_center, y_center);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}
