from flask import Flask, render_template, jsonify
import csv
from model.parse_csv import parse_csv, reduce_months

app = Flask(__name__)

# Path to your CSV file (replace with the correct path)
csv_file_path = 'Spendings.csv'

# Route to serve the HTML page
@app.route('/')
def index():
    data = {
        'date': 'YourDateHere',
        'name': 'YourNameHere',
        'amount': 'YourAmountHere'
    }
    return render_template('test.html', data=data)

# Route to serve the parsed data as JSON
@app.route('/data')
def get_data():
    parsed_data = parse_csv(csv_file_path)
    dates = parsed_data[0][0]
    amounts = parsed_data[0][2]
    reduced_data = reduce_months(dates,amounts,1)
    return render_template('test.html',d=reduced_data[0], n=parsed_data[0][1],a=reduced_data[1])

if __name__ == '__main__':
    app.run(debug=True)
