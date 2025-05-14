from datetime import datetime


# Variables and Data Types
# Variables are used to store data values
# This is a comment
a = 10 # This is an integer
b = 'hello' # This is a string
c = 3.14 # This is a float
d = True

# print(b + ' Jaunius')  

# main console functions
console_input = input('Enter your name: ')
print('Hello, ' + console_input)

# if statements
if a > 5:
    print('a is greater than 5')
elif a == 5:
    print('a is equal to 5')
else:
    print('a is less than 5')

# for loops
for i in range(5):
    print('Iteration:', i)

    # Example: Calculate age from a date

# Get current date
current_date = datetime.now()

# Example birthdate (year, month, day)
birth_date = datetime(1990, 5, 15)

# Calculate age
age = current_date - birth_date

print(f"Age: {age}")



