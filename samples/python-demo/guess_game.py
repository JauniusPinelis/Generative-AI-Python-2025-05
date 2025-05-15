# This will be a simple number guessing game
# The user will have to guess a number between 1 and 10
# if the user guess number is incorrect, the program will tell 
# either "too high" or "too low"

# Problems: 
# Does not handle non-integer inputs
# Does not handle out of range inputs
# Does not handle multiple guesses

print("Welcome to the Number Guessing Game!")
print("I'm thinking of a number between 1 and 10.")

import random

number_to_guess = random.randint(1, 10)

user_input = input("Please enter your guess: ")

print("You guessed: " + user_input)
#print(f"You guessed: {user_input}")

parsed_guess = int(user_input)

if parsed_guess < number_to_guess:
    print("Too low!")
elif parsed_guess > number_to_guess:
    print("Too high!")
else :
    print("Congratulations! You guessed the number correctly!")

print("The number was: " + str(number_to_guess))