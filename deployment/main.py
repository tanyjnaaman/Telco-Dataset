from flask import Flask, request, render_template, redirect, url_for
import data
import os

'''
This file contains the main logic of the program
'''

# initialize flask class and specify templates directory
app = Flask(__name__, template_folder="templates")

# create directory for saved files
uploads_dir = os.path.join(app.instance_path, 'uploads')
fileName = 'data.csv'
filePath = os.path.join(uploads_dir, fileName)
os.makedirs(uploads_dir, exist_ok  = True)

# set default route as home
@app.route('/')
def home():
    return render_template('home.html')


# route classify with method
@app.route('/classify', methods = ['GET','POST'])
def classify_user():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            uploaded_file.save(os.path.join(uploads_dir, fileName))
        return redirect(url_for('classify_user'))
    try:
        # make prediction
        pred_XGB, pred_ResNet, pred_FTT = data.classify(filePath)  
    except:
        return 'data & prediction error'
    try:

        # render the output in new html page
        return render_template('output.html', 
                                pred_XGB = pred_XGB, 
                                pred_ResNet = pred_ResNet,
                                pred_FTT = pred_FTT)
    except:
        return 'render error'



# run the flask server
if(__name__ == '__main__'):
    app.run(debug=False)