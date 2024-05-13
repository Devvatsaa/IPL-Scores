# from flask import Flask, render_template, request
# import pickle
# import numpy as np

# model = pickle.load(open('dect.pkl','rb'))
# app = Flask(__name__)

# @app.route('/')
# def man():
#     return render_template('front.html')

# @app.route('/predict',methods=['POST'])
# def home():
#     data1 = request.form['name1']
#     data2 = request.form['name2']
#     data3 = request.form['name3']
#     data4 = request.form['name4']
#     data5 = request.form['name5']
#     data6 = request.form['name6']
#     data7 = request.form['name7']
#     data8 = request.form['name8']
#     arr = np.array([[data1,data2,data3,data4,data5,data6,data7,data8]])
#     pred = model.predict(arr)
#     return render_template('front.html',data=pred)

# if __name__ == "__main__":
#     app.run(debug = True)




# from flask import Flask, render_template, request
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load the trained model
# with open('dect.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Function to make score prediction
# def score_predict(batting_team, bowling_team, runs, wickets, overs, runs_last_5, wickets_last_5):
#     prediction_array = []
    
#     # Batting Team
#     if batting_team == 'Chennai Super Kings':
#         prediction_array = prediction_array + [1, 0, 0, 0, 0, 0, 0, 0]
#     elif batting_team == 'Delhi Daredevils':
#         prediction_array = prediction_array + [0, 1, 0, 0, 0, 0, 0, 0]
#     elif batting_team == 'Kings XI Punjab':
#         prediction_array = prediction_array + [0, 0, 1, 0, 0, 0, 0, 0]
#     elif batting_team == 'Kolkata Knight Riders':
#         prediction_array = prediction_array + [0, 0, 0, 1, 0, 0, 0, 0]
#     elif batting_team == 'Mumbai Indians':
#         prediction_array = prediction_array + [0, 0, 0, 0, 1, 0, 0, 0]
#     elif batting_team == 'Rajasthan Royals':
#         prediction_array = prediction_array + [0, 0, 0, 0, 0, 1, 0, 0]
#     elif batting_team == 'Royal Challengers Bangalore':
#         prediction_array = prediction_array + [0, 0, 0, 0, 0, 0, 1, 0]
#     elif batting_team == 'Sunrisers Hyderabad':
#         prediction_array = prediction_array + [0, 0, 0, 0, 0, 0, 0, 1]
    
#     # Bowling Team
#     if bowling_team == 'Chennai Super Kings':
#         prediction_array = prediction_array + [1, 0, 0, 0, 0, 0, 0, 0]
#     elif bowling_team == 'Delhi Daredevils':
#         prediction_array = prediction_array + [0, 1, 0, 0, 0, 0, 0, 0]
#     elif bowling_team == 'Kings XI Punjab':
#         prediction_array = prediction_array + [0, 0, 1, 0, 0, 0, 0, 0]
#     elif bowling_team == 'Kolkata Knight Riders':
#         prediction_array = prediction_array + [0, 0, 0, 1, 0, 0, 0, 0]
#     elif bowling_team == 'Mumbai Indians':
#         prediction_array = prediction_array + [0, 0, 0, 0, 1, 0, 0, 0]
#     elif bowling_team == 'Rajasthan Royals':
#         prediction_array = prediction_array + [0, 0, 0, 0, 0, 1, 0, 0]
#     elif bowling_team == 'Royal Challengers Bangalore':
#         prediction_array = prediction_array + [0, 0, 0, 0, 0, 0, 1, 0]
#     elif bowling_team == 'Sunrisers Hyderabad':
#         prediction_array = prediction_array + [0, 0, 0, 0, 0, 0, 0, 1]
    
#     prediction_array = prediction_array + [runs, wickets, overs, runs_last_5, wickets_last_5]
#     prediction_array = np.array([prediction_array])
    
#     pred = model.predict(prediction_array)
#     return int(round(pred[0]))

# @app.route('/')
# def home():
#     return render_template('front.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Retrieve form data
#     batting_team = request.form['batting_team']
#     bowling_team = request.form['bowling_team']
#     runs = int(request.form['runs'])
#     wickets = int(request.form['wickets'])
#     overs = float(request.form['overs'])
#     runs_last_5 = int(request.form['runs_last_5'])
#     wickets_last_5 = int(request.form['wickets_last_5'])

#     # Make prediction
#     predicted_score = score_predict(batting_team, bowling_team, runs, wickets, overs, runs_last_5, wickets_last_5)

#     # Return predicted score as a response
#     return str(predicted_score)

# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Add CORS support to allow requests from different origins
model = pickle.load(open('dect.pkl', 'rb'))

@app.route('/')
def man():
    return render_template('front.html')

@app.route('/predict', methods=['POST'])  # Update this line to accept POST requests
def predict():
    data = request.form.to_dict()
    batting_team = data['batting_team']
    bowling_team = data['bowling_team']
    runs = float(data['runs'])
    wickets = float(data['wickets'])
    overs = float(data['overs'])
    runs_last_5 = float(data['runs_last_5'])
    wickets_last_5 = float(data['wickets_last_5'])

    # Map team names to one-hot encoded vectors (if needed)
    # Replace this with your actual mapping logic if required

    # Add your prediction logic here using the model
    prediction_array = []

    # Batting Team
    if batting_team == 'Chennai Super Kings':
        prediction_array += [1, 0, 0, 0, 0, 0, 0, 0]
    elif batting_team == 'Delhi Daredevils':
        prediction_array += [0, 1, 0, 0, 0, 0, 0, 0]
    # Add other team mappings here

    # Bowling Team
    if bowling_team == 'Chennai Super Kings':
        prediction_array += [1, 0, 0, 0, 0, 0, 0, 0]
    elif bowling_team == 'Delhi Daredevils':
        prediction_array += [0, 1, 0, 0, 0, 0, 0, 0]
    # Add other team mappings here

    # Add other features to the prediction array
    prediction_array += [runs, wickets, overs, runs_last_5, wickets_last_5]

    # Convert prediction array to numpy array
    prediction_array = np.array([prediction_array])

    # Make prediction
    predicted_score = model.predict(prediction_array)[0]

    return jsonify({'predicted_score': predicted_score})

if __name__ == "__main__":
    app.run(debug=True)
