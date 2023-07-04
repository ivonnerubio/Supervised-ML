import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import streamlit as st

# Data Formation
def generate_linear_data():
    x = np.linspace(0, 15, 15)
    y = 2 * x + 1    
    noise = np.random.normal(0, 2.5, len(y)) 
    y = y + noise
    return x,y

# Cost Function
def compute_cost(x,y,w, b):
    f_wb = w * x + b

    m = len(x)
    cost = (f_wb - y) ** 2        
    j_wb = (1/(2 * m)) * np.sum(cost)

    return j_wb



# Gradient Function
def compute_gradient(x,y,w,b):
    m = len(x)
    f_wb = w * x + b
    dj_dw = (np.sum((f_wb - y) * x)) / m
    dj_db = (np.sum(f_wb - y)) / m

    return dj_dw, dj_db


# Gradient Descent Function
def compute_gradient_descent(x,y,w_in,b_in,alpha,num_iters,compute_cost,compute_gradient):
    J_history = []
    p_history = []

    alpha = alpha
    num_iters = num_iters
    w = w_in
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x,y,w,b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 10000:
            J_history.append( compute_cost(x,y,w,b))
            p_history.append([w,b])
    return w, b, J_history, p_history


def generate_prediction_line(w_final,b_final):
    new_x = np.linspace(0, 15, 15)
    new_y = w_final * x + b_final
    return new_x, new_y



def generate_logistic_data():
    return "data"


st.title("Regression Model")
model_selection = st.radio("Regression Type",["Linear","Logistic"],horizontal=True)

if(model_selection == "Linear"): 
        #features = st.select_slider('Number of features',options=['1', '2', '3'])
        x,y = generate_linear_data()
        w_final, b_final, J_hist, p_hist = compute_gradient_descent(x ,y, 0, 0, 1.0e-2,1000, compute_cost, compute_gradient)
        new_x, new_y = generate_prediction_line(w_final,b_final)



if(model_selection == "Linear"): 
        st.button("Refresh Data")
        st.text("Data")
        fig = plt.figure()
        plt.scatter(x, y)
        plt.plot(new_x,new_y, linestyle = 'dotted',color='C1')
        st.pyplot(fig)


if(model_selection == "Logistic"): 
    st.text("Logistic")
    
    


