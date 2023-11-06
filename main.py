import streamlit as st
import extra_streamlit_components as stx

from upload import upload
from run import run
from review import review
from intro import intro
from visualization import visuuu
from model_jingwei.main import new_code



st.set_page_config(page_title="Continuous Learning App")

val = stx.stepper_bar(steps=["Upload", "Run", 'Visualisation','Review'])

if val == 0:
    upload()
elif val == 1:
    # new_code()
    run()
elif val == 2:
    visuuu()
elif val == 3:
    review()

