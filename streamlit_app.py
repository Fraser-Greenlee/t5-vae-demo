import pandas as pd
import streamlit as st
from transformers import AutoTokenizer
from t5_vae_flax.src.t5_vae import FlaxT5VaeForAutoencoding


st.title('T5-VAE')
st.text('''
Try interpolating between lines of Python code using this T5-VAE.
''')


def get_model():
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = FlaxT5VaeForAutoencoding.from_pretrained("flax-community/t5-vae-python")
    assert model.params['t5']['shared']['embedding'].shape[0] == len(tokenizer), "T5 Tokenizer doesn't match T5Vae embedding size."
    return model, tokenizer


def get_latent(text):
    model, tokenizer = get_model()
    import pdb
    pdb.set_trace()
    return model.get_latent(tokenizer(text))


def output_from_latent(lt):
    model, tokenizer = get_model()
    return tokenizer.decode(model.generate(latent=lt))


def decode(ratio, txt_1, txt_2):
    lt_1, lt_2 = get_latent(txt_1), get_latent(txt_2)
    lt_new = lt_1 + ratio * (lt_2 - lt_1)
    return output_from_latent(lt_new)


st.text_input("x = 3",          key="in_1")
st.text_input("y += 'hello'",   key="in_2")
r = st.slider('Interpolation Ratio')
st.write(decode(r, st.session_state.in_1, st.session_state.in_2))
