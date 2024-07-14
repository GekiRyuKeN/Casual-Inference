import streamlit as st
import pandas as pd
import numpy as np
from dowhy import CausalModel
import plotly.express as px
import plotly.graph_objects as go
import json

# Generate synthetic data
def generate_synthetic_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'transaction_id': range(1000),
        'transaction_fee': np.random.uniform(0.0001, 0.01, 1000),
        'transaction_size': np.random.uniform(200, 1000, 1000),
        'number_of_inputs': np.random.randint(1, 5, 1000),
        'number_of_outputs': np.random.randint(1, 5, 1000),
        'confirmation_time': np.random.uniform(1, 60, 1000)
    })
    return data

# Load data
@st.cache_data
def load_data():
    data = generate_synthetic_data()
    return data

data = load_data()

st.title("Causal Inference in Blockchain Transactions")
st.write("This app demonstrates causal inference in the blockchain domain, focusing on understanding the causal effects of different transaction features on confirmation times.")

# Show data
if st.checkbox("Show raw data"):
    st.write(data)

# Exploratory Data Analysis
st.header("Exploratory Data Analysis")

# Correlation heatmap
correlation = data.corr()
fig_corr = px.imshow(correlation, text_auto=True, aspect="auto")
fig_corr.update_layout(title="Correlation Heatmap")
st.plotly_chart(fig_corr.to_dict())

# Scatter plot matrix
fig_scatter = px.scatter_matrix(data, dimensions=['transaction_fee', 'transaction_size', 'number_of_inputs', 'number_of_outputs', 'confirmation_time'])
fig_scatter.update_layout(title="Scatter Plot Matrix")
st.plotly_chart(fig_scatter.to_dict())

# Distribution plots
st.subheader("Distribution of Variables")
for column in ['transaction_fee', 'transaction_size', 'number_of_inputs', 'number_of_outputs', 'confirmation_time']:
    fig_dist = px.histogram(data, x=column, marginal="box")
    fig_dist.update_layout(title=f"Distribution of {column}")
    st.plotly_chart(fig_dist.to_dict())

# Causal Inference Analysis
st.header("Causal Inference Analysis")
st.write("We use the `DoWhy` library to build a causal model from the data.")

# Create a causal model
model = CausalModel(
    data=data,
    treatment=['transaction_fee', 'transaction_size', 'number_of_inputs', 'number_of_outputs'],
    outcome='confirmation_time'
)

# Attempt to visualize the causal model
causal_graph = model.view_model()
if causal_graph:
    st.graphviz_chart(causal_graph)
else:
    st.write("Unable to visualize the causal model graph.")

# Identification
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
st.write("Identified Estimand:")
st.write(identified_estimand)

# Estimation
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)
st.write("Causal Estimate:")
st.write(estimate)

# Allow user to select feature for intervention
selected_feature = st.selectbox("Select a feature to intervene on:", ['transaction_fee', 'transaction_size', 'number_of_inputs', 'number_of_outputs'])

# Perform intervention
if st.button("Intervene"):
    st.write(f"Intervening on {selected_feature}. This would simulate changing the values of {selected_feature} and observing the effect on confirmation time.")

    # Example intervention: increasing the selected feature
    data_intervened = data.copy()
    data_intervened[selected_feature] *= 2

    # Create a causal model for the intervened data
    model_intervened = CausalModel(
        data=data_intervened,
        treatment=['transaction_fee', 'transaction_size', 'number_of_inputs', 'number_of_outputs'],
        outcome='confirmation_time'
    )

    # Identification
    identified_estimand_intervened = model_intervened.identify_effect(proceed_when_unidentifiable=True)

    # Estimation
    estimate_intervened = model_intervened.estimate_effect(
        identified_estimand_intervened,
        method_name="backdoor.linear_regression"
    )

    st.write("Causal Estimate after intervention:")
    st.write(estimate_intervened)

    # Visualize the effect of intervention
    fig_intervention = go.Figure()
    fig_intervention.add_trace(go.Scatter(x=data[selected_feature], y=data['confirmation_time'], mode='markers', name='Original'))
    fig_intervention.add_trace(go.Scatter(x=data_intervened[selected_feature], y=data_intervened['confirmation_time'], mode='markers', name='After Intervention'))
    fig_intervention.update_layout(title=f"Effect of Intervention on {selected_feature}", xaxis_title=selected_feature, yaxis_title='Confirmation Time')
    st.plotly_chart(fig_intervention.to_dict())

# Deployment instructions
st.header("Deployment")
st.write("To deploy this app, you can use platforms like Heroku or Azure. Make sure to include the `requirements.txt` file for dependencies.")

# Version information
st.sidebar.header("Version Information")
st.sidebar.write(f"Streamlit version: {st.__version__}")
st.sidebar.write(f"Pandas version: {pd.__version__}")
st.sidebar.write(f"NumPy version: {np.__version__}")
import dowhy
st.sidebar.write(f"DoWhy version: {dowhy.__version__}")
import plotly
st.sidebar.write(f"Plotly version: {plotly.__version__}")