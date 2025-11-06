# Standard imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  
os.environ["RAYON_NUM_THREADS"] = "2" # Add this line
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from tqdm import tqdm
import plotly.express as px
# torch.set_grad_enabed(False)

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Force garbage collection
import gc
gc.collect()

hook_name = "layer_20/width_16k/canonical"
model = HookedTransformer.from_pretrained_no_processing("gemma-2-2b", device=device, center_writing_weights=False)
# model = HookedTransformer.from_pretrained("gemma-2-2b", device=device)

# the cfg dict is returned alongside the SAE since it may contain useful information for analysing the SAE (eg: instantiating an activation store)
# Note that this is not the same as the SAEs config dict, rather it is whatever was in the HF repo, from which we can extract the SAE config dict
# We also return the feature sparsities which are stored in HF for convenience.
sae = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",  # see other options in sae_lens/pretrained_saes.yaml
    sae_id=hook_name,  # won't always be a hook point
    device=device,
)

from transformer_lens.utils import tokenize_and_concatenate

dataset = load_dataset(
    # path="monology/pile-ughted",
    path="NeelNanda/pile-10k",
    split="train",
    streaming=True,
)

token_dataset = tokenize_and_concatenate(
    dataset=dataset,  # type: ignore
    tokenizer=model.tokenizer,  # type: ignore
    streaming=True,
    max_length=50, #sae.cfg.metadata.context_size,
    add_bos_token=sae.cfg.metadata.prepend_bos,
)


## running dashboard:
test_feature_idx_gpt = list(range(10)) #+ [1000]

from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner


feature_vis_config_gpt = SaeVisConfig(
    hook_point=hook_name,
    features=test_feature_idx_gpt,
    minibatch_size_features=64,
    minibatch_size_tokens=128, #256
    verbose=True,
    device=device,
)

visualization_data_gpt = SaeVisRunner(
    feature_vis_config_gpt
).run(
    encoder=sae,  # type: ignore
    model=model,
    tokens=token_dataset[:10000]["tokens"],  # type: ignore
)

from sae_dashboard.data_writing_fns import save_feature_centric_vis

filename = f"sae_gemma-2-2b-20-gemmascope-res-16k_feature_dashboards.html"
save_feature_centric_vis(sae_vis_data=visualization_data_gpt, filename=filename)

del model, sae, dataset, token_dataset

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Force garbage collection
import gc
gc.collect()