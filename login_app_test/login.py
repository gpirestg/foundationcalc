import streamlit as st
import app2

st.set_page_config(
    page_title="TG-Digital",  # ← this changes the browser tab title
    page_icon="💻",           # ← optional: add a custom emoji or favicon
    #layout="centered"         # or "wide"
)

# --- Set up login credentials ---
USER_CREDENTIALS = {
    "admin": "1234",
    "gilberto": "pires"
}

# --- Initialise session state ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def show_login():
    # Custom CSS for background image
    st.markdown("""
        <style>
        .stApp {
            background-image: url("https://www.tonygee.com/wp-content/uploads/2025/01/p.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        label {
        color: white !important;
        font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)

    # Logo or image on top of the page
    st.markdown("""
        <div style="text-align: center; margin-top: 10px;">
            <img src="https://www.tonygee.com/wp-content/uploads/2021/06/SocialImg.jpg"
                 style="width: 200px; max-width: 90%; height: auto; transform: translateX(-10px);">
        </div>
        """, unsafe_allow_html=True)

    # Login title
    st.markdown("<h1 style='color:white; text-align:center;'>Login Page</h1>", unsafe_allow_html=True)
    
    # Login form 
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.success("Login successful!")
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid username or password")


def show_app():
    app2.start()

# --- App logic ---
if st.session_state.authenticated:
    show_app()
else:
    show_login()
