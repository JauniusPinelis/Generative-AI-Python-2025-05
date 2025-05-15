from datetime import datetime

# Variables and Data Types
# Variables are used to store data values
# This is a comment

a = 1 # This is an integer
b = 'hello' # This is a string
c = 3.14 # This is a float
d = True # boolean value
e = "True"

# print(b + ' Jaunius')  

# main console functions
# console_input = input('Enter your name: ')
# print('Hello, ' + console_input)

# if statements
if a > 5:
    print('a is greater than 5')
    print('a is greater than 5')
    print('a is greater than 5')
elif a == 5:
    print('a is equal to 5')
else:
    print('a is less than 5')

# for loops
for i in range(5):
    print('Iteration:', i)
    print('Iteration:', i)
    print('Iteration:', i)
    # Example: Calculate age from a date

# while loops
while i > 3:
    print('While loop iteration:', i)
    i += 1

# try catch
try:
    # Attempt to convert a string to an integer
    num = int('abc')  # This will raise a ValueError
except Exception as e:
    print("An unexpected error occurred:", str(e))

# Get current date
current_date = datetime.now()

# Example birthdate (year, month, day)
birth_date = datetime(1990, 1, 14)

# Calculate age
age = current_date - birth_date

print(f"Age: {age}")



