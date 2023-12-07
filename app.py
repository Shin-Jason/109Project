from flask import Flask, render_template, request
import pandas as pd
from Student_Performance_Predictor import load, get_all_p_x_given_y, get_p_y, get_prob_y_given_x
app = Flask(__name__)

df = load("StudentsPerformance.csv")
all_p_x_given_y = get_all_p_x_given_y("Label", df)
p_y = get_p_y("Label", df)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    #get data
    gender = request.form['gender']
    lunch = request.form['lunch']
    test_preparation_course = request.form['test_preparation_course']
    race_ethnicity = request.form['race_ethnicity']
    parental_education = request.form['parental_education']

    #create map for features
    input_data = {
        'gender': int(gender),
        'lunch': int(lunch),
        'test preparation course': int(test_preparation_course),
        'race/ethnicity': race_ethnicity,
        'parental level of education': parental_education
    }
    input_df = pd.DataFrame([input_data])
    #use one-hot encoding for 'race/ethnicity' and 'parental level of education'
    input_df = pd.get_dummies(input_df, columns=['race/ethnicity', 'parental level of education'])

    #make sure all are present/expected
    expected_columns = set(all_p_x_given_y.keys()) - {'Label'}
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    #align with data
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)
    #predict outcome
    prob_success = get_prob_y_given_x(1, input_df.iloc[0], all_p_x_given_y, p_y)
    prediction = 'Higher Speculative Outcome' if prob_success >= 0.5 else 'Lower Speculative Outcome'
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

