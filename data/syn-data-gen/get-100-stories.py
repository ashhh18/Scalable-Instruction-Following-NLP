## load the first 100 stories

import json

input_file = 'train-1.json' 
output_file = 'sampled_stories.json'  

with open(input_file, 'r') as f:
    stories = json.load(f)

sampled_stories = dict(list(stories.items())[:100])

with open(output_file, 'w') as f:
    json.dump(sampled_stories, f, indent=4)

print(f"100 stories have been saved to {output_file}")
