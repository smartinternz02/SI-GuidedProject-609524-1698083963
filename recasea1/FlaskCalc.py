from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'm1n2b3v4c5x6z7'

# Load the model and dataset
data = pd.read_csv('processed_data.csv')
model = pickle.load(open('recaseamodel.pkl', 'rb'))

# Route for the home page
@app.route("/")
def index():
    return render_template("index.html")

# Route for the calculator page
@app.route("/calculate", methods=["GET", "POST"])
def calculator():
    if request.method == "GET":
        return render_template("calculator.html")
    elif request.method == "POST":
        # Extract user inputs from the POST request
        features = ['Global_reactive_power', 'Voltage', 'Global_intensity',
                    'kitchen_cons', 'laundry_cons', 'heatcool_cons',
                    'period_e', 'is_weekend', 'other_cons']
        user_inputs = {
            'Global_reactive_power': request.form['globalReactivePower'],
            'Voltage': request.form['voltage'],
            'Global_intensity': request.form['globalIntensity'],
            'kitchen_cons': request.form['kitchenEnergy'],
            'laundry_cons': request.form['laundryEnergy'],
            'heatcool_cons': request.form['heatingCoolingEnergy'],
            'period_e': request.form['timeOfDay'],
            'is_weekend': request.form['weekdayOrWeekend'],
            'other_cons': request.form['otherAppliancesEnergy']
        }

        # Convert user inputs to DataFrame for prediction
        user_data = pd.DataFrame([user_inputs])

        # Predict Global_active_power using the trained RandomForestRegressor model
        user_features = user_data[features]
        predicted_power = model.predict(user_features)[0]

        # Filter data based on user inputs for period_e and is_weekend
        filtered_data = data[
            (data['period_e'] == int(user_inputs['period_e'])) &
            (data['is_weekend'] == int(user_inputs['is_weekend']))
            ]

        filtered_data = filtered_data.dropna(subset=['Global_active_power'])
        # Calculate the mean Global_active_power for the filtered data
        expected_power = filtered_data['Global_active_power'].mean()

        # Analyze if the obtained value is appropriate given the time of day and weekend status
        if predicted_power > (expected_power + 1.25):
            power_status = "The predicted Global active power is higher than the regular usage for this time and weekend status."
        elif predicted_power < expected_power:
            power_status = "The predicted Global active power is lower than the regular usage for this time and weekend status."
        else:
            power_status = "The predicted Global active power is in line with the regular usage for this time and weekend status."

        # Display predicted Global_active_power to the user
        predicted_power = f"Predicted Global active power: {predicted_power: .2f} kW"

        # Display the normal value of Global_active_power for the given time of day and weekend status
        normal_power = f"Normal value of Global active power for the specified time and weekend status: {expected_power:.2f} kW"

        # Visualize energy usage distribution of different appliance types based on user input

        appliance_columns = ['Kitchen appliances', 'Washing machine & Dryer', 'Fans, AC and Heating',
                             'Other appliances']
        user_appliance_usage = [user_inputs[column] for column in
                                ['kitchen_cons', 'laundry_cons', 'heatcool_cons', 'other_cons']]

        plt.figure(figsize=(8, 6))
        plt.pie(user_appliance_usage, labels=appliance_columns, autopct='%1.1f%%',
                colors=['cyan', 'yellow', 'salmon', 'maroon'])
        plt.title('Energy Usage Distribution of Different Appliances')
        plt.tight_layout()

        # Save the plot as an image file
        plot_path = 'static/user_plot.png'  # Save the plot in the 'static' folder
        plt.savefig(plot_path)
        plt.close()  # Close the plot to release memory

        '''# Pass the plot path along with other results to the template
        return render_template("result.html", predicted_power=predicted_power, power_status=power_status,
                               normal_power=normal_power, plot_path=plot_path)'''
        session['predicted_power'] = predicted_power
        session['power_status'] = power_status
        session['normal_power'] = normal_power
        session['plot_path'] = plot_path
        return redirect(url_for("show_result"))


# Route for the result page
@app.route("/result")

def show_result():
    predicted_power = session.get('predicted_power')
    power_status = session.get('power_status')
    normal_power = session.get('normal_power')
    plot_path = session.get('plot_path')
    return render_template("result.html", predicted_power=predicted_power, power_status=power_status,
                           normal_power=normal_power, plot_path=plot_path)

if __name__ == '__main__':
    app.run(debug=True)
