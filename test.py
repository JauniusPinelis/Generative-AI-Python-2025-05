string_input =  "1. Condition A\n2. Condition B\n3. Condition C"

def parse_conditions(input_string):
    list_of_strings = input_string.split("\n")

    for i, string_value in enumerate(list_of_strings):
        list_of_strings[i] = string_value[3:]  # Remove first 3 characters
        
    return list_of_strings

output_to_receive = [
    "Condition A",
    "Condition B",
    "Condition C"
]

print(parse_conditions(string_input))