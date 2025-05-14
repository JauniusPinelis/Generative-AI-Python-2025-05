year_input = input('Enter your year of birth: ')
month_input = input('Enter your month of birth: ')
day_input = input('Enter your day of birth: ')

# Convert inputs to integers
year = int(year_input)
month = int(month_input)
day = int(day_input)

from datetime import datetime

# Get current date
current_date = datetime.now()

date_of_birth = datetime(year, month, day)

# Calculate age
age = current_date - date_of_birth

print(f"Age: {age.days // 365} years")