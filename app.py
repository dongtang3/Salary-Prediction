import gradio as gr
import pandas as pd
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import threading

# Initialize Prometheus metrics
PREDICTIONS_TOTAL = Counter('salary_predictions_total', 'Total number of salary predictions made')
CONTRIBUTIONS_TOTAL = Counter('salary_contributions_total', 'Total number of salary data contributions')
VIEWS_TOTAL = Counter('salary_data_views_total', 'Number of times salary data was viewed')
PREDICTION_ERROR = Histogram('salary_prediction_error', 'Prediction error distribution')
DATASET_SIZE = Gauge('salary_dataset_size', 'Current size of the salary dataset')

# Start Prometheus server in a separate thread
def start_prometheus_server():
    start_http_server(8000)

threading.Thread(target=start_prometheus_server, daemon=True).start()

# Load data and initialize model
data = pd.read_csv("Salary_Data.csv")
DATASET_SIZE.set(len(data))
x = data["YearsExperience"].values.reshape(-1, 1)
y = data["Salary"].values
lr = LinearRegression()
lr.fit(x, y)

# Function for displaying the table
def show_table(show):
    if show:
        VIEWS_TOTAL.inc()
        return data
    else:
        return None

# Function for plotting graphs
def plot_graph(graph_type, val):
    filtered_data = data[data["YearsExperience"] >= val]
    if graph_type == "Non-Interactive":
        plt.figure(figsize=(10, 5))
        plt.scatter(filtered_data["YearsExperience"], filtered_data["Salary"])
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.tight_layout()
        return plt
    else:
        layout = go.Layout(
            xaxis=dict(range=[0, 20]),
            yaxis=dict(range=[0, data["Salary"].max() + 10000])
        )
        fig = go.Figure(
            data=go.Scatter(
                x=filtered_data["YearsExperience"],
                y=filtered_data["Salary"],
                mode="markers"
            ),
            layout=layout
        )
        return fig

# Function for salary prediction
def predict_salary(experience):
    PREDICTIONS_TOTAL.inc()
    experience_array = np.array([[experience]])
    predicted_salary = lr.predict(experience_array)[0]
    return f"Your predicted salary is {round(predicted_salary)}"

# Function for data contribution
def contribute_data(experience, salary):
    CONTRIBUTIONS_TOTAL.inc()
    to_add = pd.DataFrame({"YearsExperience": [experience], "Salary": [salary]})
    to_add.to_csv("Salary_Data.csv", mode='a', header=False, index=False)
    DATASET_SIZE.inc()

    # Update the model with new data
    global data, lr
    data = pd.read_csv("Salary_Data.csv")
    x = data["YearsExperience"].values.reshape(-1, 1)
    y = data["Salary"].values
    lr.fit(x, y)

    # Observe prediction error
    predicted_salary = lr.predict([[experience]])[0]
    if salary != 0:
        error_percentage = abs(predicted_salary - salary) / salary * 100
        PREDICTION_ERROR.observe(error_percentage)

    return "Your contribution has been submitted!"

# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🌟 Salary Prediction App 🌟")
    with gr.Tab("Home"):
        gr.Markdown("## Welcome to the Salary Prediction App")
        show_table_checkbox = gr.Checkbox(label="Show Table of Data")
        table_output = gr.Dataframe()
        show_table_checkbox.change(fn=show_table, inputs=show_table_checkbox, outputs=table_output)

        graph_type = gr.Radio(["Interactive", "Non-Interactive"], label="Select Graph Type")
        val_slider = gr.Slider(0, 20, label="Filter data by minimum years of experience", value=0)
        graph_output = gr.Plot()
        graph_button = gr.Button("Generate Graph")
        graph_button.click(fn=plot_graph, inputs=[graph_type, val_slider], outputs=graph_output)

    with gr.Tab("Prediction"):
        gr.Markdown("## Salary Prediction")
        experience_input = gr.Number(label="Enter your experience (in years)")
        predict_button = gr.Button("Predict Salary")
        prediction_output = gr.Textbox(label="Predicted Salary")
        predict_button.click(fn=predict_salary, inputs=experience_input, outputs=prediction_output)

    with gr.Tab("Contribution"):
        gr.Markdown("## Contribute to the Dataset")
        experience_contrib = gr.Number(label="Your Experience (in years)")
        salary_contrib = gr.Number(label="Your Salary")
        contribute_button = gr.Button("Submit Contribution")
        contrib_output = gr.Textbox()
        contribute_button.click(fn=contribute_data, inputs=[experience_contrib, salary_contrib], outputs=contrib_output)

if __name__ == "__main__":
    demo.launch()
