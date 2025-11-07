import json
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
    
file1 = 'data/store-json/FIN.json' 
file2 = 'data/store-json/FIN2.json'
    
with open(file1, 'r') as f1, open(file2, 'r') as f2:
    stories1 = json.load(f1)
    stories2 = json.load(f2)
    
assert len(stories1) == len(stories2), "Files must have the same number of stories."
    
scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    
rouge_scores = []
for story1, story2 in zip(stories1, stories2):
    score = scorer.score(story1, story2)['rouge2']
    rouge_scores.append(score.fmeasure)
    
plt.hist(rouge_scores, bins=20, color='purple', edgecolor='black')
plt.title('ROUGE-2 score 64-4-2')
plt.xlabel('ROUGE-1 F1 Score')
plt.ylabel('Frequency')
plt.show()