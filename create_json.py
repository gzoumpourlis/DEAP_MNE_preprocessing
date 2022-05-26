import json

# DEAP account details
my_details = {
    'username': '',
    'password': ''
}

with open('login.json', 'w') as json_file:
    json.dump(my_details, json_file) 
