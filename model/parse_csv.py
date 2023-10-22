import csv
from datetime import datetime
from collections import defaultdict
import math

# Function to parse CSV data and format it as a list of dictionaries
def parse_csv(csv_filename):
    data = []
    with open(csv_filename, 'r', newline='') as csv_file:
        csv_data = csv.reader(csv_file, delimiter=",")
        next(csv_data)  # Skip the header row if it exists
        date= []
        name=[]
        amount=[]
        for line in csv_data:
            date.append(line[0])
            name.append(line[1])
            amount.append(line[2][1:])
            data.append([date,name,amount])

    return data

def reduce_months(dts,amts,grp):
    # Sample data
    dates = dts
    amounts = amts

    # Create a dictionary to store the monthly totals
    monthly_totals = defaultdict(int)
    amounts = [float(amount.replace('$', '').replace(',', '')) for amount in amounts]

    # Iterate through the dates and amounts and consolidate by month
    for date, amount in zip(dates, amounts):
        date_obj = datetime.strptime(date, '%m/%d/%Y')
        month_key = (date_obj.month, date_obj.year%100)
        monthly_totals[month_key] += round(amount,2)

    # Round the total spending to the nearest second decimal place
    for key in monthly_totals:
        monthly_totals[key] = round(monthly_totals[key], 2)

    # Separate the consolidated data into two lists
    months = [f"{month:02}/{year:02}" for (month, year) in monthly_totals.keys()]
    total_spending = list(monthly_totals.values())

    # Set the number of months to group together
    grouping_months = grp # Change this value to group by a different number of months

    # Group the monthly data
    grouped_months = [months[i] for i in range(0, len(months), grouping_months)]
    grouped_total_spending = [sum(total_spending[i:i + grouping_months]) for i in range(0, len(total_spending), grouping_months)]

    return (grouped_months, grouped_total_spending )
    # Display the separate lists
    #print("Months:", months)
    #print("Total Spending:", total_spending)
    #print("Grouped Months:", grouped_months)
    #print("Grouped Total Spending:", grouped_total_spending)


# Specify the path to your CSV file
csv_filename = 'Spendings.csv'

# Call the function to parse the CSV data
parsed_data = parse_csv(csv_filename)

dates = parsed_data[0][0]
amounts = parsed_data[0][2]
reduced_data = reduce_months(dates,amounts,3)
print(reduced_data[1])