import streamlit as st
import streamlit as st
from PIL import Image
from streamlit_extras.colored_header import colored_header
from streamlit_extras.mention import mention
from streamlit_extras.badges import badge




def intro():
    colored_header(
        label="Welcome to the Continuous Learning App !",
        description=None,
        color_name="violet-70",
    )

    st.write("**Fine-tuning :** Technique used in Deep Learning where a pre-trained model is further trained to adapt "
             "to a specific task.\n\n"
             "Our continuous learning at the Edge platform empowers you to personalize models to specific needs."
             " Dive into the inference capabilities of "
             "our new state-of-the-art model and see how it performs on your unique data. Found an image or a "
             "scenario the model isn't familiar with ? No worries. Upload your dataset and fine-tune the model to enhance its "
             "understanding. By bridging the gap between pre-trained architectures and real-world scenarios, "
             "we offer a seamless, interactive experience, ensuring that our model is not just accurate but also "
             "tailor-made for your needs.")

    # st.subheader('How to use')
    st.info('**Instructions:**\n'
            '1. First, upload your images to the database.\n'
            '2. Then, select on which dataset you want to run the inference on. This will take a few '
             'minutes.\n'
            '3. Here, the model will show you how it succeeded to clusterize the results of the inference it ran. If '
             'there are classes it did not recognize, it will select a sample of representative images for you to '
             'annotate in the next step.\n'
            '4. On this section, you will be allowed to review the results of the models in more details, as well as '
             'annotate the unrecognized images. This will create a new set of data on which you can re-train the model.\n'
            '5. Once you have selected your preferred set on unknown images, you can click on the "Fine-Tune" button '
             'to fine-tune the model.')


    st.subheader('Github')

    mention(
        label="Continuous-Learning at the Eddge App",
        icon="github",
        url="https://github.com/Sunkian/Continuous_Learning",
    )

    # from streamlit_extras.mention import mention
    st.subheader('Mentions')
    st.caption("👨‍💻 Jingwei Zuo, Researcher, AI & Digital RC")
    st.caption("👩‍💻 Alice Pagnoux, Engineer, AI & Digital RC")
    badge(type="streamlit", url="https://plost.streamlitapp.com")



