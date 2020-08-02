"use strict";

let colorGradient = ["#0000ff", "#0300fc","#0500fa","#0800f7","#0a00f5","#0d00f2","#0f00f0","#1200ed","#1500ea","#1700e8","#1a00e5","#1c00e3","#1f00e0","#2100de","#2400db","#2700d8","#2900d6","#2c00d3","#2e00d1","#3100ce","#3400cb","#3600c9","#3900c6","#3b00c4","#3e00c1","#4000bf","#4300bc","#4600b9","#4800b7","#4b00b4","#4d00b2","#5000af","#5200ad","#5500aa","#5800a7","#5a00a5","#5d00a2","#5f00a0","#62009d","#64009b","#670098","#6a0095","#6c0093","#6f0090","#71008e","#74008b","#760089","#790086","#7c0083","#7e0081","#81007e","#83007c","#860079","#890076","#8b0074","#8e0071","#90006f","#93006c","#95006a","#980067","#9b0064","#9d0062","#a0005f","#a2005d","#a5005a","#a70058","#aa0055","#ad0052","#af0050","#b2004d","#b4004b","#b70048","#b90046","#bc0043","#bf0040","#c1003e","#c4003b","#c60039","#c90036","#cb0034","#ce0031","#d1002e","#d3002c","#d60029","#d80027","#db0024","#de0021","#e0001f","#e3001c","#e5001a","#e80017","#ea0015","#ed0012","#f0000f","#f2000d","#f5000a","#f70008","#fa0005","#fc0003", "#ff0000"];

let svgColors = {
    'genRectColor': '#31b620',
    'loadRectColor': '#4960fb',
    'potGenRect': '#ff0000',
    'socColor':'#f17741',
    'colorGradient': colorGradient,
    'brokenLine': '#ff0000'};

let svgReward = {
    'energyLossRect': 'rect5879',
    'penaltyRect': 'rect5903',
    'energyLossMaxTxt': 'text3022',
    'energyLossTxt': 'text3028',
    'penaltyMaxTxt': 'text2992',
    'penaltyTxt': 'text3034',
    'rectWidth': 93.8,
    'negColor': "#ff0000",
    'posColor': "#31b620",
};

let svgParams = {
    'svgRootId': 'svg4693',
    'pInjHeight': 19.8,
    'iCurHeight': 24.6,
    'batteryHeight': 19.8,
    'colors': svgColors,
    'pqNumPrecision': 3,
    'iNumPrecision': 3,
    'socNumPrecision': 3,
    'vMagnNumPrecision': 3,
    'pUnits': 'MW',
    'qUnits': 'MVAr',
    'socUnits': 'MWh',
    'vMagnUnits': 'pu',
    'date': 'tspan5951',
    'time': 'tspan5955',
    'yearCount': 'tspan1071',
    'reward': svgReward,
    'title': 'text5968',
    'networkCollapsedCheckMark': 'text6301',
    'networkCollapsedGroup': ['text1128', 'text6301']  //'rect6315'
};

let svgDevices = [
    {'pMaxTick': 'text4219', 'qMaxTick': 'text4450', 'pInjRect': 'rect4223', 'qInjRect': 'rect4227', 'pInjText': 'text4267', 'qInjText': 'text4271', 'arrow': 'g2863'},
    {'pMaxTick': 'text1129', 'qMaxTick': 'text4479', 'pInjRect': 'rect1121', 'qInjRect': 'rect1094', 'pInjText': 'text1183', 'qInjText': 'text1206', 'arrow': 'g2899'},
    {'pMaxTick': 'text1223', 'qMaxTick': 'text4509', 'pInjRect': 'rect1227', 'qInjRect': 'rect1231', 'pInjText': 'text3968', 'qInjText': 'text3972', 'arrow': 'g2935', 'pPotential': 'rect4394'},
    {'pMaxTick': 'text3978', 'qMaxTick': 'text4539', 'pInjRect': 'rect3982', 'qInjRect': 'rect3986', 'pInjText': 'text4038', 'qInjText': 'text4042', 'arrow': 'g3007'},
    {'pMaxTick': 'text4048', 'qMaxTick': 'text4569', 'pInjRect': 'rect4052', 'qInjRect': 'rect4056', 'pInjText': 'text4081', 'qInjText': 'text4085', 'arrow': 'g2971', 'pPotential': 'rect4421'},
    {'pMaxTick': 'text4103', 'qMaxTick': 'text4599', 'pInjRect': 'rect4107', 'qInjRect': 'rect4111', 'pInjText': 'text4129', 'qInjText': 'text4133', 'arrow': 'g3079'},
    {'pMaxTick': 'text4177', 'qMaxTick': 'text4628', 'pInjRect': 'rect4181', 'qInjRect': 'rect4185', 'pInjText': 'text4201', 'qInjText': 'text4205', 'arrow': 'g3043'}
];

let svgLines = [
    {'iCurrentRect': 'rect1326', 'iCurrentText': 'text1338', 'brokenCross': ['path1591', 'path1593'], 'arrows': ['g3151', 'g2613']},
    {'iCurrentRect': 'rect1232', 'iCurrentText': 'text1324', 'brokenCross': ['path8986', 'path8988'], 'arrows': ['g2325', 'g3115']},
    {'iCurrentRect': 'rect1226', 'iCurrentText': 'text1270', 'brokenCross': ['path1623', 'path1625'], 'arrows': ['g2541', 'g2577']},
    {'iCurrentRect': 'rect1272', 'iCurrentText': 'text1284', 'brokenCross': ['path1615', 'path1617'], 'arrows': ['g2469', 'g2505']},
    {'iCurrentRect': 'rect1286', 'iCurrentText': 'text1298', 'brokenCross': ['path1631', 'path1633'], 'arrows': ['g2361', 'g2397', 'g2433']}
];

let svgStorageUnits = [
  {'socMaxTick': 'text1215', 'socText': 'text4215', 'batteryRect': 'rect1607', 'arrow': 'g3043'}
];

let svgBuses = [
    {'vMagnText': 'text1153', 'brokenCross': ['path1131', 'path1133']},
    {'vMagnText': 'text1159', 'brokenCross': ['path1123', 'path1125']},
    {'vMagnText': 'text1165', 'brokenCross': ['path1105', 'path1108']},
    {'vMagnText': 'text1077', 'brokenCross': ['path1098', 'path1100']},
    {'vMagnText': 'text1141', 'brokenCross': ['path1107', 'path1109']},
    {'vMagnText': 'text1147', 'brokenCross': ['path1115', 'path1117']}
];