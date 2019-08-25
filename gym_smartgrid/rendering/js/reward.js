
function initReward(svgReward) {
  svgNetwork.getElementById(svgReward.energyLossMaxTxt).innerHTML = svgReward.energyLossMax;
  svgNetwork.getElementById(svgReward.penaltyMaxTxt).innerHTML = svgReward.penaltyMax;

  svgNetwork.getElementById(svgReward.energyLossRect).setAttribute('width', 0);
  svgNetwork.getElementById(svgReward.energyLossRect).style.fill = svgReward.color;

  svgNetwork.getElementById(svgReward.penaltyRect).setAttribute('width', 0);
  svgNetwork.getElementById(svgReward.penaltyRect).style.fill = svgReward.color;
}

function updateReward(svgReward, e_loss, penalty) {
  let widthLoss = svgReward.rectWidth * e_loss / svgReward.energyLossMax;
  widthLoss = Math.min(widthLoss, svgReward.rectWidth);
  svgNetwork.getElementById(svgReward.energyLossRect).setAttribute('width', widthLoss);

  let widthPenalty = svgReward.rectWidth * penalty / svgReward.penaltyMax;
  widthPenalty = Math.min(widthPenalty, svgReward.rectWidth);
  svgNetwork.getElementById(svgReward.penaltyRect).setAttribute('width', widthPenalty);
}