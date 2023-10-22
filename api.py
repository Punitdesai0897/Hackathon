from distutils.log import debug 
from fileinput import filename 
from flask import *
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from model.parse_csv import parse_csv, reduce_months
from templates.runModel import *
from transformers import BertTokenizer, TFBertModel


app = Flask(__name__) 
app.config['UPLOAD_FOLDER'] = 'calendars'
# Path to your CSV file (replace with the correct path)
csv_file_path = 'Spendings.csv'



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

loaded_model = load_model('model.h5')
calendar_df = preprocessCals()
spending_df = preprocessSpend() 
preprocess_data(calendar_df, spending_df, tokenizer)
model_inputs, Y_TEST, next_weeks = prepare_model_inputs(calendar_df, get_data, tokenizer)
predicted_spending = loaded_model.predict(model_inputs)
for i in range(0,len(predicted_spending[0])-1):
     if predicted_spending[0][i] > 0:
          predicted_spending[0][i] = predicted_spending[0][i]
     else:
          predicted_spending[0][i] = 0
print(Y_TEST)
print(predicted_spending)
#next_weeks =  extractor()
print('next',next_weeks)

@app.route('/') 
def main(): 
    return render_template("index.html", name1 ='', name2 ='',d='',a='',m='')
     

@app.route('/', methods = ['POST']) 
def success(): 
    if request.method == 'POST': 

        parsed_data = parse_csv(csv_file_path)
        dates = parsed_data[0][0]
        amounts = parsed_data[0][2]
        reduced_data = reduce_months(dates,amounts,1)

        # Get the list of files from webpage 
        files = request.files.getlist("file") 
        for file in files:      
            files[0].save(os.path.join(app.config['UPLOAD_FOLDER'], 'cal file'))
            files[1].save(os.path.join(app.config['UPLOAD_FOLDER'], 'spending file'))
            #return render_template("index.html", name1 = files[0].filename, name2 = files[1].filename)
            return render_template("index.html", name1 = files[0].filename, name2 = files[1].filename,d=reduced_data[0], n=parsed_data[0][1],a=reduced_data[1])
        else:
            return render_template("index.html", name1 ='', name2 ='',d='', n='',a='')
        
    '''    

@app.route('/data')
def get_month():
    parsed_data = parse_csv(csv_file_path)
    dates = parsed_data[0][0]
    amounts = parsed_data[0][2]
    reduced_data = reduce_months(dates,amounts,1)
    return render_template('index.html',d=reduced_data[0], n=parsed_data[0][1],a=reduced_data[1])
 ''' 
    # Return the reduced data to your template
    #return render_template('index.html', d=reduced_data[0], n=parsed_data[0][1], a=reduced_data[1],m=months)

if __name__ == '__main__': 
	app.run(debug=True)
