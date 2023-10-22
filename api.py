from distutils.log import debug 
from fileinput import filename 
from flask import *
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__) 
app.config['UPLOAD_FOLDER'] = 'files_for_model'


@app.route('/') 
def main(): 
    return render_template("index.html", name1 ='', name2 ='')
     

@app.route('/', methods = ['POST']) 
def success(): 
    if request.method == 'POST': 
        # Get the list of files from webpage 
        files = request.files.getlist("file") 
        for file in files:      
            files[0].save(os.path.join(app.config['UPLOAD_FOLDER'], 'cal file'))
            files[1].save(os.path.join(app.config['UPLOAD_FOLDER'], 'spending file'))
            return render_template("index.html", name1 = files[0].filename, name2 = files[1].filename)
        else:
            return render_template("index.html", name1 ='', name2 ='')
        

if __name__ == '__main__': 
	app.run(debug=True)
