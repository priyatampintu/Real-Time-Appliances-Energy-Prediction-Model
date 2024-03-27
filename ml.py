import gradio as gr
import numpy as np
import pickle

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import argparse

# Function to preprocess input data
def preprocess_input(hu_build_out, Windspeed, Tdewpoint, month, weekday, hour,
                     hu_Kitchen, hu_living, hu_laundry, hu_office, hu_bath,
                     hu_ironing_room, hu_teen, hu_parent):
    # Convert month name to numeric value
    month_dict = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    month_numeric = month_dict[month]

    # Convert weekday name to numeric value
    weekday_dict = {
        "Sunday": 0, "Monday": 1, "Tuesday": 2, "Wednesday": 3,
        "Thursday": 4, "Friday": 5, "Saturday": 6
    }
    weekday_numeric = weekday_dict[weekday]

    # Scale hour to a value between 0 and 23
    hour_numeric = hour % 24

    # Calculate mean of humidity features
    mean_humidity = np.mean([hu_Kitchen, hu_living, hu_laundry, hu_office,
                             hu_bath, hu_ironing_room, hu_teen, hu_parent])

    # Calculate absolute difference of 'hu_build_out' and mean humidity
    humidity_difference = np.abs(hu_build_out - mean_humidity)

    # Convert Windspeed to np.log10
    Windspeed_log = np.log10(Windspeed)

    # Combine all features
    features = np.array([[hu_build_out, Windspeed_log, Tdewpoint, month_numeric,
                          weekday_numeric, hour_numeric, humidity_difference]])

    print('features------------------', features)
    scaled_features = scaler.transform(features)
    print('scaled_features------------------', scaled_features)

    return scaled_features


# Function to load model and make predictions
def predict_appliance_energy_consumption(hu_build_out, Windspeed, Tdewpoint, month, weekday, hour,
                                         hu_Kitchen, hu_living, hu_laundry, hu_office, hu_bath,
                                         hu_ironing_room, hu_teen, hu_parent):
    # print('scaled_features----------', hu_build_out, month, weekday, hour)
    # Preprocess input data
    scaled_features = preprocess_input(hu_build_out, Windspeed, Tdewpoint, month, weekday, hour,
                                       hu_Kitchen, hu_living, hu_laundry, hu_office, hu_bath,
                                       hu_ironing_room, hu_teen, hu_parent)


    # Make predictions
    y_pred = loaded_model.predict(scaled_features)
    actual_pred_value = 10 ** (y_pred) - 1
    consume_electricity = str(round(actual_pred_value[0], 2)) + ' wH'

    # Return predicted appliance energy consumption in wH
    return consume_electricity


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Applicances Energy Prediction", add_help=True)
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    args = parser.parse_args()

    scaler = pickle.load(open('scaler.sav', 'rb'))

    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    block = gr.Blocks()

    with block:
        gr.Markdown("# Appliances Energy Prediction Model")
        with gr.Row():

            with gr.Column():

                # Define input components for Gradio interface
                inputs = [
                    gr.Slider(minimum=0, maximum=100, label="Humidity in Outside Building(in %)",  step=0.1),
                    gr.Slider(minimum=0, maximum=100, label="Windspeed(in m/s)", step=0.1),
                    gr.Slider(minimum=-20, maximum=20, label="Tdewpoint(in Â°C)"),
                    gr.Dropdown(["January", "February", "March", "April", "May", "June",
                                 "July", "August", "September", "October", "November", "December"],
                                label="Month"),
                    gr.Dropdown(["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
                                label="Weekday"),
                    gr.Slider(minimum=0, maximum=23, label="Hour", step=1),
                    gr.Slider(minimum=0, maximum=100, label="Humidity in Kitchen(in %)", step=0.1),
                    gr.Slider(minimum=0, maximum=100, label="Humidity in living Room(in %)", step=0.1),
                    gr.Slider(minimum=0, maximum=100, label="Humidity in laundry(in %)", step=0.1),
                    gr.Slider(minimum=0, maximum=100, label="Humidity in office(in %)", step=0.1),
                    gr.Slider(minimum=0, maximum=100, label="Humidity in bathroom(in %)", step=0.1),
                    gr.Slider(minimum=0, maximum=100, label="Humidity in ironing room(in %)", step=0.1),
                    gr.Slider(minimum=0, maximum=100, label="Humidity in teenage room(in %)", step=0.1),
                    gr.Slider(minimum=0, maximum=100, label="Humidity in parent room(in %)", step=0.1)
                ]

            with gr.Column():
                final_prompt = gr.Textbox(label="Total Predicted Electricity Consumption", type="text")

                def display_gen_prompt(outputs):
                    final_prompt.value = outputs[0]  # Get the last element of the outputs, which is the user prompt
                    return outputs[0]


                generate_prompt = gr.Button(value="Run")
                generate_prompt.click(fn=predict_appliance_energy_consumption, inputs=inputs,
                                      outputs=[final_prompt])

    block.launch(server_name='0.0.0.0', server_port=8504, debug=args.debug, share=args.share)



