from flask import Flask, render_template, Response, request,  redirect, url_for, make_response,jsonify
import threading
import numpy as np
import numpy
import cv2, time
from yolo_video import yoloCamera as camera
import os
from yolo import YOLO, detect_video
import json
import tensorflow as tf
global graph,model
graph = tf.get_default_graph()
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


basedir = os.path.abspath(os.path.dirname(__file__))
 

@app.route('/', methods=['POST', 'GET'])  # 添加路由
def main():
    if request.method == 'POST':
        f = request.files['file']
 
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
 
        user_input = request.form.get("name")
 
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
 
        upload_path = os.path.join(basepath, 'static/images', f.filename)  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)
 
        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)

        src_img = "./static/images/test.jpg"

        yolocore = YOLO()
        img = cv2.imread(src_img)
        after_img, json_file = yolocore.detect_image(img)
        # youya = json.dumps(json_file, sort_keys=True, indent=4, separators=(',', ':'))
        # print(str(youya))
        cv2.imwrite(os.path.join("./static/images", "output.jpg"),after_img)

        # del yolocore
        return render_template('upload_ok.html',userinput=user_input, userJson=json_file, val1=time.time())
 
    return render_template('upload.html')


@app.route('/index')
def index():
    print('request received')
    return render_template('index.html')

def getStream(camera):
    while True:
        mainFrame = camera.getFrame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+mainFrame+b'\r\n\r\n')

@app.route('/getimg')
def getimg():
    return Response(getStream(camera(2)), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    #cmr = camera(0)
    #frameGenerator = threading.Thread(target=getFrame, args=(cmr,))
    #frameGenerator.start()
    app.run(host='0.0.0.0', debug=False, port=5000)
    #app_run = threading.Thread(target=runapp, args=())
    #app_run.start()


