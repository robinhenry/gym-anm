"use strict";

/** The websocket connection to the server. */
let ws;

/** The graph representing the network. */
let graph;

/** The reward visualization. */
let reward;

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


/**
 * Initialize the WebSocket connection to the server and describe how to handle
 * new clients, clients leaving, new messages, and errors.
 */
function init() {

  // Initialize the connection to Web Socket server (wsServerAddress is
  // provided in index.html).
  ws = new WebSocket(wsServerAddress);

  /**
   * What to do when the connection with the WebSocket server is established.
   */
  ws.onopen = function() {};

  /**
   * What to do when a message from the WebSocket server is received.
   * @param {Object} e The data received, where e.data contains the message in
   * JSON format.
   */
  ws.onmessage = function(e) {
    let messageJson = JSON.parse(e.data);

    switch (messageJson.messageLabel) {

      // Case 1: Initialize the web-based visualization.
      case 'init':
        graph = new Graph(svgDevices, svgLines, svgStorageUnits, svgBuses,
                          svgParams,
                          messageJson.deviceType, messageJson.pMax,
                          messageJson.qMax, messageJson.sRate,
                          messageJson.vMagnMin, messageJson.vMagnMax,
                          messageJson.socMax);
        addTitle(svgParams, messageJson.title);
        reward = new RewardSignal(svgReward, messageJson.energyLossMax,
            messageJson.penaltyMax);
        break;

      // Case 2: Update the visualization.
      case 'update':
        graph.update(messageJson.pInjections, messageJson.qInjections,
                     messageJson.sFlows, messageJson.socStorage,
                     messageJson.pPotential, messageJson.vMagn);
        updateDate(svgParams, messageJson.time[0], messageJson.time[1]);
        updateTime(svgParams, messageJson.time[2], messageJson.time[3]);
        updateYearCount(svgParams, messageJson.yearCount);
        reward.update(svgReward, messageJson.reward[0], messageJson.reward[1]);
        networkCollapsed(svgParams, messageJson.networkCollapsed);
        break;
    }
  };

  /**
   * What to do when the connection with the WebSocket server is closed.
   */
  ws.onclose = function() {};

  // When an error message is received from the WebSocket server.
  /**
   * What to do when an error message is received form the WebSocket server.
   * @param e - The error message.
   */
  ws.onerror = function(e) {
    console.log(e);
  };
}