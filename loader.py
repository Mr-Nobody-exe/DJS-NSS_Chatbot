import json
import os

def load_data():
    # Get absolute path to project root
    base_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_dir, "data", "nss_data.json")

    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
# ## Comment out the following code it is just for testing purposes
# if __name__ == "__main__":
#     data = load_data()
#     print(data["about"]["mission"])
#     print(data["events"][3]["title"])
#     print(data["projects"][2]["description"])
