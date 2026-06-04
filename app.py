import streamlit as st
import model1, model2, model3, model4

choice = st.sidebar.selectbox(
    "Choose Model",
    ["Model 1", "Model 2", "Model 3", "Model 4"]
)

if choice == "Model 1":
    model1.run()
elif choice == "Model 2":
    model2.run()
elif choice == "Model 3":
    model3.run()
elif choice == "Model 4":
    model4.run()