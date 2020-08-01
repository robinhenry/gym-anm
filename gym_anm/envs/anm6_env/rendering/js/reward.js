/**
 * A class to display rewards.
 */
class RewardSignal {

  constructor(svgReward, energyLossMax, penaltyMax) {
    this.energyLossMax = energyLossMax;
    this.penaltyMax = penaltyMax;

    // Set the ticks for the maximum costs.
    svgNetwork.getElementById(svgReward.energyLossMaxTxt).innerHTML = energyLossMax;
    svgNetwork.getElementById(svgReward.penaltyMaxTxt).innerHTML = penaltyMax.toPrecision(svgReward.numPrecision);

    // Color the energy loss bar and set its width to 0.
    svgNetwork.getElementById(svgReward.energyLossRect).setAttribute('width', 0);
    svgNetwork.getElementById(svgReward.energyLossRect).style.fill = svgReward.negColor;

    // Color the penalty bar and set its width to 0.
    svgNetwork.getElementById(svgReward.penaltyRect).setAttribute('width', 0);
    svgNetwork.getElementById(svgReward.penaltyRect).style.fill = svgReward.negColor;
  }

  update(svgReward, e_loss, penalty) {
    // Set the width of the energy loss bar.
    let widthLoss = svgReward.rectWidth * Math.abs(e_loss) / this.energyLossMax;
    widthLoss = Math.min(widthLoss, svgReward.rectWidth);
    svgNetwork.getElementById(svgReward.energyLossRect).setAttribute('width', widthLoss);

    // Set the color of the energy loss bar.
    if (e_loss > 0) {
      svgNetwork.getElementById(svgReward.energyLossRect).style.fill = svgReward.negColor;
    } else {
      svgNetwork.getElementById(svgReward.energyLossRect).style.fill = svgReward.posColor;
    }

    // Set the width of the penalty bar.
    let widthPenalty = svgReward.rectWidth * penalty / this.penaltyMax;
    widthPenalty = Math.min(widthPenalty, svgReward.rectWidth);
    svgNetwork.getElementById(svgReward.penaltyRect).setAttribute('width', widthPenalty);

    // Set the texts.
    svgNetwork.getElementById(svgReward.energyLossTxt).innerHTML = (e_loss).toFixed(2);
    svgNetwork.getElementById(svgReward.penaltyTxt).innerHTML = (penalty).toFixed(2);
  }
}
