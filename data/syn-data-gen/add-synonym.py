## replaces certain words with synonyms

import json
import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize
from random import choice, random
from tqdm import tqdm

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def replace_with_synonyms(sentence, replacement_percentage=0.3):
    words = word_tokenize(sentence)  
    tagged_words = pos_tag(words)  
    
    new_sentence = []

    for word, tag in tagged_words:
        if tag in ['NN', 'VB', 'JJ', 'RB']:  
            if random() < replacement_percentage:
                synonyms = get_synonyms(word)
                if synonyms:
                    new_word = choice(synonyms)
                    new_word = new_word.replace("_", " ")
                else:
                    new_word = word 
            else:
                new_word = word
        else:
            new_word = word  

        new_sentence.append(new_word)

    return ' '.join(new_sentence)

input_file = 'train-1.json'  

with open(input_file, 'r') as f:
    stories = json.load(f)

def augment_first_25_percent(stories, replacement_percentage=0.3):
    total_stories = len(stories)
    end_idx = int(total_stories * 0.25)
    
    augmented_stories = {}
    for story_id, story_text in tqdm(list(stories.items())[:end_idx], desc="Augmenting first 25%", total=end_idx):
        augmented_text = replace_with_synonyms(story_text, replacement_percentage)
        augmented_stories[story_id] = augmented_text

    augmented_stories.update({k: v for k, v in list(stories.items())[end_idx:]})
    
    return augmented_stories

augmented_stories = augment_first_25_percent(stories, replacement_percentage=0.3)

output_file = 'train-2.json'  
with open(output_file, 'w') as f:
    json.dump(augmented_stories, f, indent=4)

print(f"synonym dataset saved to {output_file}")
