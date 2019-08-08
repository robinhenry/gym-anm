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
    updateBar(messageJson.new_value);
  };

  // When the connection with the WebSocket server is closed.
  ws.onclose = function() {};

  // When an error message is received from the WebSocket server.
  ws.onerror = function(e) {
    console.log(e);
  };
}

function updateBar(value) {
  document.getElementById('myBar').value = value;
}