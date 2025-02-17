from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

fetal_distress = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
maternal_age = [25, 30, 35, 20, 28, 32, 38, 22, 27, 29]
gestational_age = [38, 39, 40, 37, 38, 41, 39, 36, 40, 38]
previous_c_section = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
blood_pressure = [120, 110, 130, 115, 125, 118, 132, 113, 128, 122]
bmi = [22, 27, 24, 23, 26, 28, 30, 22, 25, 27]

c_section = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] 

X = np.column_stack((fetal_distress, maternal_age, gestational_age, previous_c_section, blood_pressure, bmi))
y = c_section

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=500)
model.fit(X_scaled, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        fetal_distress_input = int(request.form['fetal_distress'])
        maternal_age_input = int(request.form['maternal_age'])
        gestational_age_input = int(request.form['gestational_age'])
        previous_c_section_input = int(request.form['previous_c_section'])
        blood_pressure_input = int(request.form['blood_pressure'])
        bmi_input = float(request.form['bmi'])  
        new_patient_data = np.array([[fetal_distress_input, maternal_age_input, gestational_age_input, previous_c_section_input, 
                                      blood_pressure_input, bmi_input]])

        new_patient_data_scaled = scaler.transform(new_patient_data)
        predicted_c_section = model.predict(new_patient_data_scaled)

        if predicted_c_section[0] == 1:
            prediction = "C-section"
            instructions = "For a C-section procedure, please consult with your healthcare provider. The steps typically involve anesthesia, an incision, delivery, and post-operative care to ensure both the mother and baby are healthy."
        else:
            prediction = "No C-section"
            instructions = "For a natural delivery: Focus on preparing for labor and delivery by attending prenatal classes, practicing breathing and relaxation techniques, and discussing your birth plan with your healthcare provider. It's also helpful to stay active, follow a healthy diet, and ensure you have a support system in place for labor. Your doctor can guide you on what to expect and how to manage any unexpected changes during delivery."

    except Exception as e:
        prediction = f"Error: {str(e)}"
        instructions = None

    return render_template('result.html', prediction=prediction, instructions=instructions)

if __name__ == '__main__':
    app.run(debug=True)