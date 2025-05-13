from datetime import datetime

def calculate_age(birth_date):
    today = datetime.today()
    age = today.year - birth_date.year

    # Adjust if birthday hasn't occurred yet this year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        age -= 1

    return age

def main():
    dob_input = input("Enter your date of birth (YYYY-MM-DD): ")
    try:
        birth_date = datetime.strptime(dob_input, "%Y-%m-%d")
        age = calculate_age(birth_date)

        if age < 18:
            print("Under age")
        else:
            print(f"Your age is: {age}")
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")

main()
