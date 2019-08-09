x_center = 50;
y_center = 50;
radius = 40;
clock_linewidth = 3;
hour_pointer_len = 20;
min_pointer_len = 35;
pointer_width = 2;
number_radius = radius - 8;
number_center_x = x_center - 3;
number_center_y = y_center + 4;
number_font = "10px Verdana";


clock = document.getElementById("clock");
ctx = clock.getContext("2d");

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

function drawNumber() {
  for (n = 0; n < 12; n++) {
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
