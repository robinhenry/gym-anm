"use strict";

let months = ['January', 'February', 'March', 'April', 'Mai', 'June', 'July',
              'August', 'September', 'October', 'November', 'December'];

/**
 * Update the displayed calendar.
 * @param {number} month The index of the month to display (1-indexing).
 * @param {number} day The day number to display.
 */
function update_calendar(month, day) {
  document.getElementById('month').innerHTML = months[month - 1];
  document.getElementById('day').innerHTML = day;
}
