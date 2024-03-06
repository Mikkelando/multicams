import json
def append_to_json(file_path, data):
    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
    except FileNotFoundError:
        json_data = {}
    json_data.update(data)
    with open(file_path, 'w') as file:
        json.dump(json_data, file, indent=4)


