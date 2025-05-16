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
while i < 3:
    print('While loop iteration:', i)
    i += 1

# try catch
try:
    # Attempt to convert a string to an integer
    num = int('abc')  # This will raise a ValueError
except Exception as e:
    print("An unexpected error occurred:", str(e))

# objects 

# We want to store information about a person
# each person has a name, age, and city

person1_name = 'John'
person1_age = 30
person1_city = 'New York'

person2_name = 'Jane'
person2_age = 25
person2_city = 'Los Angeles'

# dictionary to store person information
person_information_object = {
    'name': "jaunius",
    'age': 35,
    'city': 'Birzai'
}

person_information_object['age'] = 36  # Update age

print("Person Information Object:", person_information_object)
print("Person Name:", person_information_object['name'])

# lists 

# A list is a collection which is ordered and changeable. Allows duplicate members.
my_list = [1, 2, 3, 4, 5, 3]

# Adding an element to the list
my_list.append(6)

# Removing an element from the list
my_list.remove(3)

# Accessing elements in the list
print("First element in the list:", my_list[1])

# filter a list, dont worry about it for now.
filtered_list = [x for x in my_list if x > 3]

# a collection of objects
# list of dictionaries
people = [
    {
        'name': 'John',
        'age': 30,
        'city': 'New York'
    },
    {
        'name': 'Jane',
        'age': 25,
        'city': 'Los Angeles'
    }
]


# # Get current date
# current_date = datetime.now()

# # Example birthdate (year, month, day)
# birth_date = datetime(1990, 1, 14)

# # Calculate age
# age = current_date - birth_date

# print(f"Age: {age}")



