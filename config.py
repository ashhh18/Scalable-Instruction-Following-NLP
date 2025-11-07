import sys
embed = int(sys.argv[1])
layer_inp = int(sys.argv[2])
heads_inp = int(sys.argv[3])
eval_file = sys.argv[4] if len(sys.argv) > 4 else "null"

batch_size = 8
max_length = 512
model_config = {"hidden_size": embed, "layers": layer_inp  , "heads": heads_inp}
model_name = f"gpt2_{embed}_{layer_inp}_{heads_inp}_syn"
cache_dir = ".cache"
