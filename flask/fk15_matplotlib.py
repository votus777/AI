from flask import Flask , render_template, send_file, make_response

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np
import keras 

app = Flask(__name__)

@app.route('/mypic')

def mypic() : 
    return render_template('mypic.html')


@app.route('/plot')
def plot():
    
    fig, axis = plt.subplot(1) 
    
    #데이터 준비 
    x = [1,2,3,4,5]
    y = [0,2,1,3,4]

    # 데이터를 캔버스에 그리자 
    axis.plt(x,y)
    canvas = FigureCanvas(fig)
    
    from io import BytesIO
    img = BytesIO() 
    fig.savefig(img)
    img.seek(0) 
    return send_file(img, mimetype = 'image/png')

if __name__ == '__main__' :
    port = 5050
    app.debug = False
    app.run(port = port) 
    
    