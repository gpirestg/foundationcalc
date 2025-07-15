# gekko_test_app.py
import streamlit as st
import pandas as pd
import time
from gekko import GEKKO

# === UI ===
st.title("Gekko Optimisation Test")
num_cases = st.slider("Number of Load Cases", 1, 100, 10)
run_button = st.button("Run Optimisation")

# === Dummy Load Case Generator ===
def generate_dummy_loadcases(n):
    return pd.DataFrame({
        'P': [100 + i*10 for i in range(n)],
        'M': [50 + i*5 for i in range(n)]
    })

# === Dummy Gekko Function (simple parabola min) ===
def run_gekko_case_old(p, m):
    m_gekko = GEKKO(remote=False)
    x = m_gekko.Var(value=1.0, lb=0.1, ub=10)
    m_gekko.Equation(p * x**2 + m / x == 150)
    m_gekko.Obj(x**2)
    m_gekko.options.SOLVER = 1
    m_gekko.options.IMODE = 3
    m_gekko.options.MAX_TIME = 20  # limit solve time
    try:
        m_gekko.solve(disp=False)
        return x.value[0]
    except Exception as e:
        return f"Error: {e}"
    
def run_gekko_case(p, m):
    m_gekko = GEKKO(remote=False)
    
    x = m_gekko.Var(value=1.0, lb=0.1, ub=10)
    
    # Simple objective: minimise (x - target_value)^2
    target = 3 + (p + m) % 5  # varies with loadcase but always solvable
    m_gekko.Obj((x - target)**2)

    m_gekko.options.SOLVER = 1
    m_gekko.options.IMODE = 3
    m_gekko.options.MAX_TIME = 5

    try:
        m_gekko.solve(disp=False)
        return round(x.value[0], 4)
    except Exception as e:
        return f"Error: {e}"


# === RUN LOGIC ===
if run_button:
    loadcases = generate_dummy_loadcases(num_cases)
    st.info(f"Running {num_cases} load case(s)...")
    start_time = time.time()

    results = []
    progress_bar = st.progress(0)
    for i, row in loadcases.iterrows():
        x_opt = run_gekko_case(row['P'], row['M'])
        #results.append({'Case': i+1, 'x_opt': x_opt})
        results.append({'Case': i+1, 'x_opt': str(x_opt)})
        progress_bar.progress((i + 1) / num_cases)

    total_time = time.time() - start_time
    st.success(f"Completed in {total_time:.2f} seconds")
    st.dataframe(pd.DataFrame(results))
