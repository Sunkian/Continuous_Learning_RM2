import streamlit as st
import extra_streamlit_components as stx

from upload import upload
from run import run
from review import review
from intro import intro
from visualization import visuuu
st.set_page_config(page_title="Continuous Learning App")

val = stx.stepper_bar(steps=["Intro", "Upload", "Run", 'Visualization','Review'])

if val == 0:
    intro()
elif val == 1:
    upload()
elif val == 2:
    run()
elif val == 3:
    visuuu()
elif val == 4:
    review()
