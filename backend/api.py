from neuralNetworkMultiLayer import NeuronalNetwork
from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
import numpy as np

inputs = np.array(
    [
        [0,0,0],
        [1,1,1],
        [0.5,0.5,0.5],
        [0,1,0],
        [0,0.5,0],
        [1,0.6,0.6],
        [1,0,0],
        [0,0,1],
        [0.6,0.6,1],
    ]
)

output = np.array([
    [1], 
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [1],
    [0],
])
nn = NeuronalNetwork(inputs,output,[4,4])
nn.train(50000)

app = Flask(__name__)
CORS(app)
@app.route('/api/v1/color', methods=['GET'])
def api_color():
    red = float(request.args.get('red')) / 255
    green = float(request.args.get('green')) / 255
    blue = float(request.args.get('blue')) / 255

    return jsonify({"predict": nn.predict([red,green,blue])})

app.run()