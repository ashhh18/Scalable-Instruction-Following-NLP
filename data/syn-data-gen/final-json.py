## converted to final json format compatible with dataloader

import json

with open('train-2.json', 'r') as f:
    data = json.load(f)

formatted_data = [{"text": story} for story in data.values()]

with open('train-4.json', 'w') as f:
    json.dump(formatted_data, f, indent=4)

print("Conversion complete. Saved as 'formatted_data.json'.")
