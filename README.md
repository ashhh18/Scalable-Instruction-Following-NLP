# To train the model run the command
`python3 run_model.py {embedding_size} {num_layers} {num_heads}`

# To train tinystories-reward run:
`python3 -m catstory.catstory-train {embedding_size} {num_layers} {num_heads} {path_where_model_saved}`

# To generate using catstory run:
`python3 -m catstory.catstory-gen {embedding_size} {num_layers} {num_heads} {path_where_model_saved}`
 
# Evaluations:
## To run any of the evaluations run:
`python3 -m evaluations.{correct_file} {embedding_size} {num_layers} {num_heads} {path_where_model_saved}`
 
# Other files/folders:
## Data:
attn-heatmap contains the heatmaaps of the model where interpretability was run
krct-results contain the results of krct evals 
store-json contains all the required files
 
# Link to the presentation :
`https://docs.google.com/presentation/d/1Zip8g5M5ntaZuOc-_sNXsvJ6mFsx1R-_bYh-CV7y_9c/edit?usp=sharing`

# Link to the pretrained models : 
`https://drive.google.com/drive/folders/1IcbGcgfmURitBhBDqirey6Dx6FBNHpox?usp=sharing`


# Directory Structrue :
```
.
├── catstory
│   ├── catstory-gen.py
│   └── catstory-train.py
├── config.py
├── data
│   ├── attn-heatmap
│   │   ├── attention_head_0_128_12_8.png
│   │   ├── attention_head_1_128_12_8.png
│   │   ├── attention_head_2_128_12_8.png
│   │   ├── attention_head_3_128_12_8.png
│   │   ├── attention_head_4_128_12_8.png
│   │   ├── attention_head_5_128_12_8.png
│   │   ├── attention_head_6_128_12_8.png
│   │   └── attention_head_7_128_12_8.png
│   ├── __init__.py
│   ├── krct-results
│   │   ├── context_tracking_stories.txt
│   │   ├── factual_knowledge_stories.txt
│   │   └── reasoning_stories.txt
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   ├── __init__.cpython-311.pyc
│   │   ├── __init__.cpython-312.pyc
│   │   ├── __init__.cpython-36.pyc
│   │   ├── tinystories.cpython-310.pyc
│   │   ├── tinystories.cpython-311.pyc
│   │   ├── tinystories.cpython-312.pyc
│   │   └── tinystories.cpython-36.pyc
│   ├── store-json
│   │   ├── 50_stories.json
│   │   ├── catstories.json
│   │   ├── context_tracking_prompts.json
│   │   ├── factual_prompts.json
│   │   ├── FIN2.json
│   │   ├── FIN.json
│   │   ├── instruct.json
│   │   ├── reasoning_prompts.json
│   │   ├── rouge.json
│   │   └── tinystories.json
│   ├── syn-data-gen
│   │   ├── add-synonym.py
│   │   ├── final-json.py
│   │   ├── get-100-stories.py
│   │   └── text-to-json.py
│   └── tinystories.py
├── evaluations
│   ├── context_tracking.py
│   ├── evaluate_gpt2_pretrained.py
│   ├── evaluate_selftrained.py
│   ├── factual_knowledge.py
│   ├── instruct_eval.py
│   ├── interpretability.py
│   ├── __pycache__
│   │   ├── context_tracking.cpython-312.pyc
│   │   ├── evaluate_gpt2_pretrained.cpython-312.pyc
│   │   ├── evaluate_selftrained.cpython-312.pyc
│   │   ├── factual_knowledge.cpython-312.pyc
│   │   ├── instruct_eval.cpython-312.pyc
│   │   ├── interpretability.cpython-312.pyc
│   │   ├── krct_prompt_eval.cpython-312.pyc
│   │   ├── reasoning.cpython-312.pyc
│   │   ├── rouge1.cpython-312.pyc
│   │   └── rouge.cpython-312.pyc
│   ├── reasoning.py
│   ├── rouge1.py
│   └── rouge2.py
├── gemini_api.py
├── __pycache__
│   ├── config.cpython-312.pyc
│   ├── gemini_api.cpython-312.pyc
│   ├── gpt2.cpython-312.pyc
│   ├── run_gpt2.cpython-312.pyc
│   ├── run_model.cpython-312.pyc
│   └── scratchmodel.cpython-312.pyc
├── README.md
├── Report.pdf
├── run_model.py
└── scratchmodel.py

10 directories, 68 files
```
