import csv

# Open the input CSV file in read mode
with open('./alcohol/original_dataset.csv', 'r') as input_file:
    # Open the output CSV file in write mode
    with open('output.csv', 'w', newline='') as output_file:
        # Create a CSV reader
        reader = csv.reader(input_file)
        # Create a CSV writer
        writer = csv.writer(output_file)
        # Iterate over each row in the input CSV
        for row in reader:
            # Append 'beer' to the row
            str = ",beer"
            row.append(str)
            # Write the row to the output CSV
            writer.writerow(row)