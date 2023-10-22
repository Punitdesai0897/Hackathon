import warnings
#warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import time

#???
#!pip install transformers

from keras.layers import Input, LSTM, Dense, concatenate
from keras.models import Model
from transformers import BertTokenizer, TFBertModel
from keras.layers import Input, Dense
from keras.models import Model

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

MAX_LENGTH = 512 #can be changed bruh

next_week_events = []



input_ids = Input(shape=(MAX_LENGTH,), dtype='int32', name='input_ids')
attention_mask = Input(shape=(MAX_LENGTH,), dtype='int32', name='attention_mask')

bert_output = bert_model([input_ids, attention_mask])

MAX_TOKENS = 768

import os
import glob
from datetime import datetime
from pandas.core.frame import DataFrame
from numpy.core.multiarray import datetime_as_string
import pandas as pd
from dateutil.rrule import rrule, WEEKLY, MO, TU, WE, TH, FR, SA, SU
from datetime import datetime, date


WEEKDAYS_MAP = {
    "MO": MO,
    "TU": TU,
    "WE": WE,
    "TH": TH,
    "FR": FR,
    "SA": SA,
    "SU": SU
}

def generate_recurring_events(row, last_event_date):
    # Extract repeating patterns from the Notes column
    if 'Notes' not in row or pd.isnull(row['Notes']):
      return []

    rules = row['Notes'].split(";")
    freq_pattern = [rule for rule in rules if "FREQ=" in rule][0].split("=")[1]

    until_date = None
    if "UNTIL=" in row['Notes']:
        until_pattern = [rule for rule in rules if "UNTIL=" in rule][0].split("=")[1]
        until_date = datetime.strptime(until_pattern[:8], '%Y%m%d').date()

    if until_date and until_date < last_event_date:
        last_event_date = until_date

    byday_pattern = [rule for rule in rules if "BYDAY=" in rule]
    if byday_pattern:
        days = byday_pattern[0].split("=")[1].split(',')
    else:
        days = []

    start_date = pd.to_datetime(row['start_date']).date()

    recurring_dates = list(rrule(freq=WEEKLY, dtstart=start_date, until=last_event_date, byweekday=[WEEKDAYS_MAP[day] for day in days]))
    return [{'start_date': date, 'duration': row['duration'], 'Title': row['Title']} for date in recurring_dates]


def preprocessCals():
  # List all CSV files in the "calendars" directory
  csv_files = glob.glob("calendars/*.csv")
  # Read each CSV file into a DataFrame and store them in a list
  dfs = [pd.read_csv(file) for file in csv_files]
  # Concatenate all the DataFrames
  calendar_df = pd.concat(dfs, ignore_index=True)
  calendar_df = calendar_df.rename(columns={"Given planned earliest start": "start", "Given planned earliest end": "end"})
  #From the csv file, create another calendar_dfframe that parses events per day. and generates features
  calendar_df['start_date'] = calendar_df['start'].str.slice(0,10)
  calendar_df['start_time'] = calendar_df['start'].str.slice(10)
  calendar_df['start_time'] = calendar_df['start_time'].str.slice(0,5)
  calendar_df['start_time'] = calendar_df['start_time'].astype(str)
  calendar_df['end_date'] = calendar_df['end'].str.slice(0,10)
  calendar_df['end_time'] = calendar_df['end'].str.slice(10)
  calendar_df['end_time'] = calendar_df['end_time'].str.slice(0,5)
  calendar_df['end_time'] = calendar_df['end_time'].astype(str)
  # Function to convert time strings to total minutes
  def convert_to_minutes(time_str):
      if(time_str == 'nan'):
        return 540
      hours, minutes = map(int, time_str.split(':'))
      return hours * 60 + minutes

  # Apply the function to the column
  calendar_df['total_minutes_start'] = calendar_df['start_time'].apply(convert_to_minutes)
  calendar_df['total_minutes_end'] = calendar_df['end_time'].apply(convert_to_minutes)

  # Calculate duration taking into account time boundaries
  calendar_df['duration'] = calendar_df.apply(lambda row: row['total_minutes_end'] - row['total_minutes_start'] if row['total_minutes_end'] >= row['total_minutes_start'] else (24*60 - row['total_minutes_start']) + row['total_minutes_end'], axis=1)
  calendar_df = calendar_df.drop(columns=['start_time', 'end_time'])

  #add a repeat column for repeating events
  calendar_df['repeating'] = calendar_df['Notes'].str.slice(0,4)
  calendar_df['repeating'] = calendar_df['repeating'].apply(lambda x: 1 if x == 'FREQ' else 0)

  # Sort by ascending date
  # Convert the date_column to datetime objects
  def parse_date(date_value):
    if isinstance(date_value, pd.Timestamp):  # Check if the value is already a Timestamp
        return date_value

    if pd.isna(date_value):  # Check if the value is NaN
        return None

    try:
        # Try parsing the date with the first format
        return datetime.strptime(date_value, '%m/%d/%Y')
    except ValueError:
        try:
            # Try parsing the date with the second format
            return datetime.strptime(date_value, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            # If both fail, raise the original error
            raise ValueError(f"time data '{date_value}' does not match either format '%m/%d/%Y' or '%Y-%m-%d %H:%M:%S'")



  calendar_df['date_column_start1'] = calendar_df['start_date'].apply(parse_date)
  calendar_df['date_column_end1'] = calendar_df['end_date'].apply(parse_date)
  # Sort the calendar_dfFrame based on the date_column
  calendar_df = calendar_df.sort_values(by='date_column_start1')

  # Get the last event's date
  last_event_date = calendar_df['date_column_end1'].max().date()

  # For each row, generate recurring events
  new_rows = []
  for _, row in calendar_df.iterrows():
      if row['repeating'] == 1:
          new_events = generate_recurring_events(row, last_event_date)
          new_rows.extend(new_events)

  # Create a DataFrame from the new rows
  new_df = pd.DataFrame(new_rows)
  calendar_df = pd.concat([calendar_df, new_df], ignore_index=True)

  try:
    calendar_df = calendar_df.drop(columns=['Notes', 'Assigned Resources', 'Additional Title', 'start', 'end'])
  except:
    print("error dropping")

  calendar_df['date_column_start'] = calendar_df['start_date'].apply(parse_date)
  calendar_df['date_column_end'] = calendar_df['end_date'].apply(parse_date)

  # Sort by ascending date again
  calendar_df = calendar_df.sort_values(by='date_column_start')
  calendar_df = calendar_df.reset_index(drop=True)

  # Reset the index if needed
  calendar_df = calendar_df.reset_index(drop=True)
  #drop last row as it does not contain any useful calendar_df
  calendar_df = calendar_df.drop([len(calendar_df)-1])
  # Remove rows where all data is missing or NaN
  calendar_df = calendar_df.dropna(how='all')
  calendar_df = calendar_df.dropna(subset=['Title','date_column_start', 'repeating', 'duration'])

  return calendar_df

def custom_mse(y_true, y_pred):
    # Mean squared error for the individual outputs
    individual_mse = K.mean(K.square(y_pred - y_true), axis=-1)

    # Mean squared error for the sum of outputs
    sum_mse = K.square(K.sum(y_pred, axis=-1) - K.sum(y_true, axis=-1))

    # Combine the individual and sum MSE
    return individual_mse + 0.3 * sum_mse

def preprocessSpend():
  #NOTE: in final code replace the csv file with the input from the website
  csv_files = glob.glob("spending/*.csv")

  # Read each CSV file into a DataFrame and store them in a list
  dfs = [pd.read_csv(file) for file in csv_files]

  spending_df = pd.concat(dfs, ignore_index=True)

  # Convert the date_column to datetime objects
  spending_df['date_column_start'] = pd.to_datetime(spending_df['Date'], format='%m/%d/%Y')

  # Remove rows where all data is missing or NaN
  spending_df = spending_df.dropna(how='all')

  # Sort the DataFrame based on the date_column
  spending_df = spending_df.sort_values(by='date_column_start')

  # Reset the index if needed
  spending_df = spending_df.reset_index(drop=True)

  # Drop the last row if it does not contain any useful data
  spending_df = spending_df.drop([len(spending_df)-1])
  return spending_df



def pad_or_truncate(tokens):
    if len(tokens) < MAX_TOKENS:
        return tokens + [0] * (MAX_TOKENS - len(tokens))
    return tokens[:MAX_TOKENS]

def preprocess_data(calendar_df, spending_df, tokenizer):
    calendar_df['event_tokens'] = calendar_df['Title'].apply(lambda x: pad_or_truncate(tokenizer.encode(x, add_special_tokens=True, truncation=True)))
    spending_df['vendor_tokens'] = spending_df['Name'].apply(lambda x: pad_or_truncate(tokenizer.encode(x, add_special_tokens=True, truncation=True)))
    calendar_df['day_of_week'] = calendar_df['date_column_start'].dt.dayofweek  # This will give you numbers 0-6 for Monday-Sunday.
    return calendar_df, spending_df


def get_date_range(calendar_df):
    start_date = calendar_df['date_column_start'].iloc[9]  # 10th event
    end_date = calendar_df['date_column_start'].iloc[-102]
    return pd.date_range(start=start_date, end=end_date)



# Helper function to get the past and future 10 events/spendings and spending for next week
def get_data(date):

    # Check if 'Amount' is of type string, if not, convert it
    if spending_df['Amount'].dtype != 'object':
        spending_df['Amount'] = spending_df['Amount'].astype(str)

    # Convert 'Amount' column to float
    spending_df['Amount'] = spending_df['Amount'].str.replace('$', '').str.replace(',', '').astype(float)

    past_events = calendar_df[calendar_df['date_column_start'] < date].tail(10)
    future_events = calendar_df[calendar_df['date_column_start'] > date].head(10)
    past_spendings = spending_df[spending_df['date_column_start'] < date].tail(10)

    # Now sum up 'Amount' after conversion
    #next_week_spending = spending_df[(spending_df['date_column_start'] > date) & (spending_df['date_column_start'] <= date + timedelta(days=7))].sum()['Amount']

    # Initialize a list for daily spending
    daily_spending = []

     # Iterate over the next 7 days
    for i in range(7):
        # Calculate the start and end dates for each day
        start_date = date + timedelta(days=i)
        end_date = date + timedelta(days=i + 1)

        # Filter the spending data for the current day
        day_spending = spending_df[(spending_df['date_column_start'] > start_date) & (spending_df['date_column_start'] <= end_date)]['Amount']

        # Calculate the total spending for the day and append it to the list
        total_day_spending = day_spending.sum()
        daily_spending.append(total_day_spending)

    # Convert the list to a NumPy array for further processing
    next_week_spending = daily_spending


    # Calculate past week event density
    past_week_event_density = []
    for i in range(7):
        day_events = calendar_df[calendar_df['date_column_start'] == date - timedelta(days=i+1)]
        past_week_event_density.append(len(day_events))
    past_week_event_density = np.array(past_week_event_density[::-1]).reshape(7,1) # Reverse to have oldest day first

    # Calculate future week event density
    future_week_event_density = []
    for i in range(7):
        day_events = calendar_df[calendar_df['date_column_start'] == date + timedelta(days=i+1)]
        future_week_event_density.append(len(day_events))
    future_week_event_density = np.array(future_week_event_density).reshape(7,1)

    # Calculate past week spending density
    past_week_spending_density = []
    for i in range(7):
        day_spending = spending_df[spending_df['date_column_start'] == date - timedelta(days=i+1)]
        total_spent = day_spending['Amount'].sum()
        past_week_spending_density.append(total_spent)
    past_week_spending_density = np.array(past_week_spending_density[::-1]).reshape(7,1) # Reverse to have oldest day first

    return past_events, future_events, past_spendings, next_week_spending, past_week_event_density, future_week_event_density, past_week_spending_density



def prepare_model_inputs(calendar_df, get_data, tokenizer):
    # Now, only generate data for the second-to-last event date
    date = calendar_df['date_column_start'].iloc[-1]

    # Fetch data for that specific date
    past_events, future_events, past_spendings, next_week_spending, past_week_event_density, future_week_event_density, past_week_spending_density = get_data(date)

    dayINC = 1
    while len(future_events) < 10:
        default_event = pd.DataFrame({
            'Title': ['Default_Event'],
            'date_column_start': [date + timedelta(dayINC)],  # default date set to 1 year in the future, can be adjusted
            'repeating': [0],
            'duration': [0],
            'day_of_week': [(date + timedelta(days=dayINC)).weekday()], # Added day_of_week for default event,
            'event_tokens': [pad_or_truncate(tokenizer.encode('Default_Event', add_special_tokens=True, truncation=True))]
        })
        future_events = pd.concat([future_events, default_event], ignore_index=True)
        dayINC += 1

    if len(past_events) >= 10 and len(future_events) >= 10:
        past_events['date_column_start'] = past_events['date_column_start'].apply(lambda x: x.timestamp())
        future_events['date_column_start'] = future_events['date_column_start'].apply(lambda x: x.timestamp())
        past_spendings['date_column_start'] = past_spendings['date_column_start'].apply(lambda x: x.timestamp())

        X_TEST_past_event_tokens = np.array(past_events['event_tokens'].tolist())
        X_TEST_past_spending_tokens = np.array(past_spendings['vendor_tokens'].tolist())
        X_TEST_future_event_tokens = np.array(future_events['event_tokens'].tolist())

        X_TEST_past_event_structured = np.array(past_events[['date_column_start', 'repeating', 'duration', 'day_of_week']])
        X_TEST_past_spending_structured = np.array(past_spendings[['date_column_start', 'Amount']])
        X_TEST_future_event_structured = np.array(future_events[['date_column_start', 'repeating', 'duration', 'day_of_week']])

        X_TEST_past_week_event_density = past_week_event_density
        X_TEST_future_week_event_density = future_week_event_density
        next_week_events = X_TEST_future_week_event_density
        print('next week events', next_week_events)
        X_TEST_past_week_spending_density = past_week_spending_density

        X_TEST_current_date_encoded = np.array([date.timestamp()])
        Y_TEST = next_week_spending

    model_inputs = [X_TEST_past_event_tokens,
                    X_TEST_past_event_structured,
                    X_TEST_past_spending_tokens,
                    X_TEST_past_spending_structured,
                    X_TEST_future_event_tokens,
                    X_TEST_future_event_structured,
                    X_TEST_past_week_event_density,
                    X_TEST_future_week_event_density,
                    X_TEST_past_week_spending_density,
                    X_TEST_current_date_encoded]

    model_inputs = [np.expand_dims(x, axis=0) for x in model_inputs]

    return model_inputs, Y_TEST, next_week_events

def extractor():
    return next_week_events

from keras.models import load_model


loaded_model = load_model('model.h5')
calendar_df = preprocessCals()
spending_df = preprocessSpend() 
preprocess_data(calendar_df, spending_df, tokenizer)
model_inputs, Y_TEST, next_week = prepare_model_inputs(calendar_df, get_data, tokenizer)
predicted_spending = loaded_model.predict(model_inputs)
print(Y_TEST)
print(predicted_spending)

