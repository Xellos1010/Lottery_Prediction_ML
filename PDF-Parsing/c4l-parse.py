import pandas as pd
import re

# Define the relative path to the text file
file_path = 'PDF-Parsing/data/c4l_fl-06_29_2024-02-20-2017.txt'

# Function to parse the data
def parse_lottery_data(file_path):
    with open(file_path, 'r') as file:
        content = file.read().replace('\n', ' ')

    # Define a regex pattern to match dates
    date_pattern = re.compile(r'\d{2}/\d{2}/\d{2}')
    
    # Split the content based on dates
    parts = date_pattern.split(content)
    dates = date_pattern.findall(content)
    
    data = []
    
    for i, part in enumerate(parts[1:], start=1):  # Skip the first part as it is before the first date
        numbers = part.split()
        while len(numbers) >= 7:  # Ensure there are enough numbers to parse
            num1, num2, num3, num4, num5 = [num.strip('-') for num in numbers[:5]]
            cb_number = numbers[6]  # The 8th element is the CB number
            data.append([dates[i-1], num1, num2, num3, num4, num5, cb_number])
            numbers = numbers[7:]  # Move to the next set of numbers
            
    return data

# Parse the data
data = parse_lottery_data(file_path)

# Convert the list to a DataFrame
df = pd.DataFrame(data, columns=['Date', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'CB'])

# Save the DataFrame to a CSV file
output_path = 'parsed_c4l_fl-06_29_2024-02-20-2017.csv'
df.to_csv(output_path, index=False)

print(f"Extracted data has been saved to {output_path}")
