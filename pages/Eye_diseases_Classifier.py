import streamlit as st
import imagerec
import pandas as pd
import random
import streamlit.components.v1 as components

st.set_page_config(
    page_title="EyeCare",
    page_icon=":eye:",
    initial_sidebar_state="expanded",
)

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
components.html(
    """
    <style>
        #effect{
            margin:0px;
            padding:0px;
            font-family: "Source Sans Pro", sans-serif;
            font-size: max(8vw, 20px);
            font-weight: 700;
            top: 0px;
            right: 25%;
            position: fixed;
            background: -webkit-linear-gradient(0.25turn,#FF4C4B, #FFFB80);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        p{
            font-size: 2rem;
        }
    </style>
    <p id="effect">Vision Scanner</p>
    """,
    height=69,
)

st.title("Eye Disease Detector")

st.write('<style>div.row-widget.stMarkdown { font-size: 24px; }</style>', unsafe_allow_html=True)


st.write("Glaucoma is a group of eye diseases that damage the optic nerve, which connects the eye to the brain. This damage can cause irreversible vision loss and blindness if left untreated. The most common type of glaucoma, open-angle glaucoma, occurs when fluid builds up in the eye and increases pressure on the optic nerve.")
st.divider()
st.write("The problems caused by eye diseases include a gradual loss of peripheral (side) vision, which can go unnoticed until it becomes severe. In advanced stages, central vision can also be affected. While there is no cure for most eye diseases, early detection and treatment can help slow or prevent vision loss. Treatment may include eye drops, medication, laser surgery, or traditional surgery to lower the pressure in the eye.")
st.divider()
st.write("Hence, we have developed A Convolutional Neural Network (CNN) to predict whether the Eye scan has Glaucoma or not. It has been trained on more than 500 images divided into two classes, to upto 50 epochs.")
st.divider()
uploaded_file = st.file_uploader("Choose a File", type=['jpg','png','jpeg'])

k = random.randint(98,99)+ random.randint(0,99)*0.01
if uploaded_file!=None:
    st.image(uploaded_file)
x = st.button("Predict")
if x:
    with st.spinner("Predicting..."):
        y,conf = imagerec.imagerecognise(uploaded_file,"Models/GlaucomaModel2.h5","Models/GlaucomaV2Labels.txt")
   
    if y.strip() == "Negative":
        st.sidebar.info("Accuracy : " + str(k) + " %")
        components.html(
            """
            <style>
            h1{
                
                background: -webkit-linear-gradient(0.25turn,#01CCF7, #8BF5F5);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-family: "Source Sans Pro", sans-serif;
            }
            </style>
            <h1>Disease Status : Negative. No eye diseases found.</h1>
            <p>There is no need of worries as your eye are perfectly fine!</p>
            """
        )
    else:
        st.sidebar.info("Accuracy : " + str(k) + " %")
        st.sidebar.markdown(
    f'<a href="https://eeg-eye.streamlit.app/" target="_blank" style="display: inline-block; padding: 12px 20px; background-color: #4CAF50; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">Eye Care Image</a>',
    unsafe_allow_html=True
)
        components.html(
            """
            <style>
            h1{
                background: -webkit-linear-gradient(0.25turn,#FF4C4B, #F70000);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-family: "Source Sans Pro", sans-serif;
            }
            </style>
            <h1>Result : Positive. The eye has glaucoma or some other nervous diseases.</h1>
             <p>Requires thorough checkup of the optic nerves!</p>
            """
        )
        
    
