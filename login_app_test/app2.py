import streamlit as st

def start():
    st.title("✅ Welcome to App 2")
    st.write("This is the main app screen.")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()