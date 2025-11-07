# train/val text file to json format compatible with nltk wordnet

import json

with open("train.txt", "r") as file:
    stories = file.read().split("<|endoftext|>")

cleaned_stories = [story.strip() for story in stories if story.strip()]

stories_dict = {f"story_{i+1}": story for i, story in enumerate(cleaned_stories)}

with open("train-1.json", "w") as file:
    json.dump(stories_dict, file, indent=4)

print("TinyStories saved without <endoftext> tokens.")
