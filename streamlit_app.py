import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from t5_vae_flax.src.t5_vae import FlaxT5VaeForAutoencoding

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = T5VaeForAutoencoding.from_pretrained("flax-community/t5-vae-python")

assert model.params['t5']['shared'].shape[0] == len(tokenizer), "T5 Tokenizer doesn't match T5Vae embedding size."
