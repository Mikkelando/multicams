# import json
# def append_to_json(file_path, data):
#     try:
#         with open(file_path, 'r') as file:
#             json_data = json.load(file)
#     except FileNotFoundError:
#         json_data = {}
#     json_data.update(data)
#     with open(file_path, 'w') as file:
#         json.dump(json_data, file, indent=4)


def to_relative(parts, face_params):
    x, y, w, h = face_params.left(), face_params.top(), face_params.width(), face_params.height()
    for landmark in parts:
        x_abs, y_abs  = landmark.x, landmark.y
        

