import streamlit as st
import extra_streamlit_components as stx

from run_ft import run_ft
import sys
sys.path.append('../')  # This adds the parent directory to the path

from upload import upload


st.set_page_config(page_title="Continuous Learning App")

val = stx.stepper_bar(steps=["Upload", "Run"])

if val == 0:
    upload()
elif val == 1:
    run_ft()