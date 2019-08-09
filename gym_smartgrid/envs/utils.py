import os
from gym_smartgrid import RENDERING_FOLDER, WEB_FILES


def write_html(svg_data):

    s = """<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="styles.css">
    <script src="{0}"></script>
    <script src="dynamicVisualization.js"></script>
    <title>SmartGrid-gym</title>
</head>

<body onload="init();">

    <time datetime="2014-09-20" class="icon">
        <strong id="month"></strong>
        <span id="day"></span>
    </time>

    <div>
        <canvas id="clock" width="100" height="100"></canvas>
    </div>
    <script src="clock.js"></script>

    <object id="svg-network" data="{1}"
            type="image/svg+xml"></object></body>
</html>

    """.format(svg_data['labels'], svg_data['network'])

    html_file = os.path.join(RENDERING_FOLDER, WEB_FILES['index'])

    with open(html_file, 'w') as f:
        f.write(s)
