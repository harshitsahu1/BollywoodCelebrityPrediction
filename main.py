from flask import Flask ,request ,render_template
from datetime import datetime 
import os
import cv2
import recommend
app = Flask(__name__)

@app.route('/' ,methods=["GET"])
def index():
    return render_template('index.html')



@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        a=uploaded_file.filename
    
        
        uploaded_file.save(os.path.join("./static",a))
        print(a)
        ret="/static/{}".format(a)
        res = recommend.read_image(a)
        name = 'output' + a
        ret1 = "/static/{}".format(name)
        print(name)
        return render_template("index.html",name1=ret,name2=ret1)
        

    

if __name__=="__main__":
    app.run(debug=True)
