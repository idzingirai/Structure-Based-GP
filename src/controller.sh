#!/bin/bash

# Generate 100 unique values between 1 and 999999999
values=(14986112 16223899 22878896 68625033 101006424 113139835 125838671 213232696 311489368 376423770)

# Loop through each value and run Python script
for val in "${values[@]}"
do
  # Run Python script with value as argument and write results to file
  python main.py $val
done