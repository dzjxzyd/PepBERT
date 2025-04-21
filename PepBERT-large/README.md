This is the official repository for the paper: Du, Z., Li, Y. PepBERT: Lightweight language models for bioactive peptide representation represenation.


All pretrained models in this repository have been uploaded to Hugging Face for easy access. Usage tutorials are provided in the respective README.md files on each model’s Hugging Face page.

You can access the models at the following links:

[PepBERT-large-UniParc](https://huggingface.co/dzjxzyd/PepBERT-large-UniParc)
 
[PepBERT-large-UniRef100](https://huggingface.co/dzjxzyd/PepBERT-large-UniRef100)
  
[PepBERT-large-UniRef90](https://huggingface.co/dzjxzyd/PepBERT-large-UniRef90)
   
[PepBERT-large-UniRef50](https://huggingface.co/dzjxzyd/PepBERT-large-UniRef50)

PepBERT-small was available at a different repository at https://github.com/dzjxzyd/PepBERT-small.

You can also download dataset, scripts, and pretrained models, all the checkpoints from GoogleDrive at https://drive.google.com/drive/folders/19y7nsCUJyV_MLzBmFum99htQvCLo47HO?usp=drive_link

### A example tutorial for usage in your custom peptide embeddings.
```
import os
import torch
import importlib.util
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

# Function to download and load a Python module from the Hugging Face Hub.
def load_module_from_hub(repo_id, filename):
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Repository ID where your files are hosted.
repo_id = "dzjxzyd/PepBERT-large-UniParc"

# 1) Download and load model.py and config.py dynamically.
model_module = load_module_from_hub(repo_id, "model.py")
config_module = load_module_from_hub(repo_id, "config.py")
build_transformer = model_module.build_transformer
get_config = config_module.get_config

# 2) Download tokenizer.json and load the tokenizer.
tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)

# 3) Download model weights (tmodel_17.pt).
weights_path = hf_hub_download(repo_id=repo_id, filename="tmodel_17.pt")

# 4) Initialize the model structure and load the weights.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() or torch.backends.mps.is_available() else "cpu"
config = get_config()
model = build_transformer(
    src_vocab_size=tokenizer.get_vocab_size(),
    src_seq_len=config["seq_len"],
    d_model=config["d_model"]
)
state = torch.load(weights_path, map_location=torch.device(device))
model.load_state_dict(state["model_state_dict"])
model.eval()

# 5) Generate embeddings for an example amino acid sequence.
# Add special tokens [SOS] and [EOS] as in training.
sequence = "KRKGFLGI"
encoded_ids = (
    [tokenizer.token_to_id("[SOS]")]
    + tokenizer.encode(sequence).ids
    + [tokenizer.token_to_id("[EOS]")]
)
input_ids = torch.tensor([encoded_ids], dtype=torch.int64)

with torch.no_grad():
    # Create a simple attention mask (all ones, since no padding is used here)
    encoder_mask = torch.ones((1, 1, 1, input_ids.size(1)), dtype=torch.int64)
    # Forward pass through the encoder to get token embeddings
    emb = model.encode(input_ids, encoder_mask)
    # Remove first ([SOS]) and last ([EOS]) embeddings
    emb_no_first_last = emb[:, 1:-1, :]
    # Apply average pooling over the remaining tokens to get a fixed-size vector
    emb_avg = emb_no_first_last.mean(dim=1)
    
print("Shape of emb_avg:", emb_avg.shape)
print("emb_avg:", emb_avg)

```
