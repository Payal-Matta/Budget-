import streamlit as st
import math

st.set_page_config(page_title="Scientific Calculator", layout="centered")

# ----------- CUSTOM CSS FOR UI DESIGN -----------
st.markdown("""
    <style>
        .calculator {
            width: 350px;
            margin: auto;
            padding: 20px;
            border-radius: 15px;
            background-color: #111827;
            box-shadow: 0px 0px 20px rgba(0,0,0,0.3);
        }
        .display {
            width: 100%;
            height: 70px;
            background-color: #1f2937;
            color: white;
            font-size: 28px;
            border-radius: 10px;
            text-align: right;
            padding: 15px;
            margin-bottom: 15px;
            border: 2px solid #374151;
        }
        .btn {
            width: 70px;
            height: 55px;
            background-color: #374151;
            color: white;
            font-size: 20px;
            border-radius: 10px;
            border: none;
            margin: 6px;
        }
        .btn:hover {
            background-color: #4b5563;
        }
        .btn-orange {
            background-color: #f59e0b;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ CALCULATOR LOGIC ------------------

if "expression" not in st.session_state:
    st.session_state.expression = ""

def press(button):
    if button == "C":
        st.session_state.expression = ""
    elif button == "=":
        try:
            st.session_state.expression = str(eval(st.session_state.expression))
        except:
            st.session_state.expression = "Error"
    else:
        st.session_state.expression += button

# ------------------ UI LAYOUT ------------------

st.markdown("<div class='calculator'>", unsafe_allow_html=True)

st.markdown(
    f"<div class='display'>{st.session_state.expression}</div>",
    unsafe_allow_html=True
)

# Button labels
buttons = [
    ["C", "(", ")", "/"],
    ["7", "8", "9", "*"],
    ["4", "5", "6", "-"],
    ["1", "2", "3", "+"],
    ["0", ".", "√", "="],
    ["sin", "cos", "tan", "^"]
]

# Render grid
for row in buttons:
    cols = st.columns(4)
    for i, btn in enumerate(row):

        # special buttons
        if btn == "=":
            style = "btn btn-orange"
        else:
            style = "btn"

        if cols[i].button(btn, key=btn + str(i), help=f"Press {btn}"):
            if btn == "√":
                st.session_state.expression = str(math.sqrt(float(st.session_state.expression or 0)))
            elif btn == "sin":
                st.session_state.expression = str(math.sin(math.radians(float(st.session_state.expression or 0))))
            elif btn == "cos":
                st.session_state.expression = str(math.cos(math.radians(float(st.session_state.expression or 0))))
            elif btn == "tan":
                st.session_state.expression = str(math.tan(math.radians(float(st.session_state.expression or 0))))
            elif btn == "^":
                st.session_state.expression += "**"
            else:
                press(btn)

st.markdown("</div>", unsafe_allow_html=True)
