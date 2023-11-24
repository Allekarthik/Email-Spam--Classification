import pickle
import streamlit as st

model = pickle.load(open("spam.pkl", "rb"))
cv = pickle.load(open("vectorized.pkl", "rb"))

def main():
    st.title("Email Spam Classification")
    st.subheader("Build with Streamlit & Python")
    msg=st.text_input("Enter a text :")
    if st.button("Predict"):
        data=[msg]
        vect=cv.transform(data).toarray()
        prediction=model.predict(vect)
        result=prediction[0]
        if result==1:
            st.error("This is spam mail")
        else:
            st.success("This is ham mail")




main()
