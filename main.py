from flask import Flask, jsonify, request
import joblib
import numpy as np
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

app = Flask(__name__)

# Load scaler from file
scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Model
rf_model = joblib.load('rf_model.joblib')
knn_model = joblib.load('knn_model.joblib')
gnb_model = joblib.load('gnb_model.joblib')
svm_model = joblib.load('svm_model.joblib')

cnn_model = tf.keras.models.load_model('cnn_model.h5')
mlp_model =  tf.keras.models.load_model('mlp_model.h5')

all_ap_features = ['ap1',
 'ap2',
 'ap3',
 'ap4',
 'ap5',
 'ap6',
 'ap7',
 'ap8',
 'ap9',
 'ap10',
 'ap11',
 'ap12',
 'ap13',
 'ap14',
 'ap15',
 'ap16',
 'ap17',
 'ap18',
 'ap19',
 'ap20',
 'ap21',
 'ap22',
 'ap23',
 'ap24',
 'ap25',
 'ap26',
 'ap27',
 'ap28',
 'ap29',
 'ap30',
 'ap31',
 'ap32',
 'ap33',
 'ap34',
 'ap35',
 'ap36',
 'ap41',
 'ap42',
 'ap45',
 'ap46',
 'ap49',
 'ap50',
 'ap90',
 'ap91',
 'ap92',
 'ap93',
 'ap94',
 'ap95',
 'ap96',
 'ap97',
 'ap98',
 'ap99',
 'ap100',
 'ap101',
 'ap102',
 'ap103',
 'ap104',
 'ap105',
 'ap106',
 'ap107',
 'ap108',
 'ap109',
 'ap110',
 'ap111',
 'ap112',
 'ap113',
 'ap114',
 'ap115',
 'ap116',
 'ap117',
 'ap118',
 'ap119',
 'ap120',
 'ap122',
 'ap123',
 'ap124',
 'ap125',
 'ap126',
 'ap127',
 'ap128',
 'ap129',
 'ap130',
 'ap133',
 'ap134',
 'ap135',
 'ap138',
 'ap141',
 'ap143',
 'ap144',
 'ap145',
 'ap148',
 'ap153',
 'ap154',
 'ap158',
 'ap165',
 'ap166',
 'ap167',
 'ap168',
 'ap169',
 'ap171',
 'ap172',
 'ap180',
 'ap181',
 'ap182']

@app.route("/rf", methods=["POST"])
def predict_location_rf():
        # Receive JSON data from POST request
    request_body = request.get_json()
    
    new_data = pd.DataFrame(index=[0])

    # Mengisi nilai-nilai AP yang tidak terdeteksi dengan 0
    for ap_feature in all_ap_features:
        new_data[ap_feature] = 0

    # Mengisi nilai-nilai AP yang tidak terdeteksi dengan nilai dari request_body
    new_data.update(pd.DataFrame.from_dict(request_body, orient='index').transpose())

    # Normalize new data
    new_data_scaled = scaler.transform(new_data)

    # Predict location for new data
    predicted_location = rf_model.predict(new_data_scaled)
    predicted_xyz = label_encoder.inverse_transform(predicted_location)

    # Return predicted location
    return jsonify({"predicted_location": predicted_xyz[0]})

@app.route("/rfo", methods=["POST"])
def predict_location_rfo():
        # Receive JSON data from POST request
    request_body = request.get_json()
    
    new_data = pd.DataFrame(index=[0])

    # Mengisi nilai-nilai AP yang tidak terdeteksi dengan 0
    for ap_feature in all_ap_features:
        new_data[ap_feature] = 0

    # Mengisi nilai-nilai AP yang tidak terdeteksi dengan nilai dari request_body
    new_data.update(pd.DataFrame.from_dict(request_body, orient='index').transpose())

    # Normalize new data
    new_data_scaled = scaler.transform(new_data)

    predicted_probabilities = rf_model.predict_proba(new_data_scaled)
    top_three_indices = np.argsort(predicted_probabilities, axis=1)[:, ::-1][:, :3]
    top_three_probabilities = np.take_along_axis(predicted_probabilities, top_three_indices, axis=1)
    predicted_classes = label_encoder.inverse_transform(top_three_indices.flatten())
    lst = []
    for i in range(3):
        predict_object = {}
        xyz = predicted_classes[i].split(",")
        predict_object["x"] = xyz[0]
        predict_object["y"] = xyz[1]
        predict_object["z"] = xyz[2]
        predict_object["confidence"] = top_three_probabilities[0,i]
        lst.append(predict_object)

    # Return predicted location
    return jsonify({"data": lst})

@app.route("/gnb", methods=["POST"])
def predict_location_gnb():
        # Receive JSON data from POST request
    request_body = request.get_json()
    
    new_data = pd.DataFrame(index=[0])

    # Mengisi nilai-nilai AP yang tidak terdeteksi dengan 0
    for ap_feature in all_ap_features:
        new_data[ap_feature] = 0

    # Mengisi nilai-nilai AP yang tidak terdeteksi dengan nilai dari request_body
    new_data.update(pd.DataFrame.from_dict(request_body, orient='index').transpose())

    # Normalize new data
    new_data_scaled = scaler.transform(new_data)

    # Predict location for new data
    predicted_location = gnb_model.predict(new_data_scaled)
    predicted_xyz = label_encoder.inverse_transform(predicted_location)

    # Return predicted location
    return jsonify({"predicted_location": predicted_xyz[0]})

@app.route("/svm", methods=["POST"])
def predict_location_svm():
        # Receive JSON data from POST request
    request_body = request.get_json()
    
    new_data = pd.DataFrame(index=[0])

    # Mengisi nilai-nilai AP yang tidak terdeteksi dengan 0
    for ap_feature in all_ap_features:
        new_data[ap_feature] = 0

    # Mengisi nilai-nilai AP yang tidak terdeteksi dengan nilai dari request_body
    new_data.update(pd.DataFrame.from_dict(request_body, orient='index').transpose())

    # Normalize new data
    new_data_scaled = scaler.transform(new_data)

    # Predict location for new data
    predicted_location = svm_model.predict(new_data_scaled)
    predicted_xyz = label_encoder.inverse_transform(predicted_location)

    # Return predicted location
    return jsonify({"predicted_location": predicted_xyz[0]})

@app.route("/knn", methods=["POST"])
def predict_location_knn():
        # Receive JSON data from POST request
    request_body = request.get_json()
    
    new_data = pd.DataFrame(index=[0])

    # Mengisi nilai-nilai AP yang tidak terdeteksi dengan 0
    for ap_feature in all_ap_features:
        new_data[ap_feature] = 0

    # Mengisi nilai-nilai AP yang tidak terdeteksi dengan nilai dari request_body
    new_data.update(pd.DataFrame.from_dict(request_body, orient='index').transpose())

    # Normalize new data
    new_data_scaled = scaler.transform(new_data)

    # Predict location for new data
    predicted_location = knn_model.predict(new_data_scaled)
    predicted_xyz = label_encoder.inverse_transform(predicted_location)

    # Return predicted location
    return jsonify({"predicted_location": predicted_xyz[0]})

@app.route("/cnn", methods=["POST"])
def predict_location_cnn():
    # Receive JSON data from POST request
    request_body = request.get_json()
    
    new_data = pd.DataFrame(index=[0])

    # Mengisi nilai-nilai AP yang tidak terdeteksi dengan 0
    for ap_feature in all_ap_features:
        new_data[ap_feature] = 0

    # Mengisi nilai-nilai AP yang tidak terdeteksi dengan nilai dari request_body
    new_data.update(pd.DataFrame.from_dict(request_body, orient='index').transpose())


    # Normalize new data
    new_data_scaled = scaler.transform(new_data)

    predicted_location = cnn_model.predict(new_data_scaled)
    # Ambil indeks dengan nilai tertinggi dari setiap baris dalam predicted_location
    predicted_labels = np.argmax(predicted_location, axis=1)

    # Invers transform label menjadi nilai asli xyz
    predicted_xyz = label_encoder.inverse_transform(predicted_labels)

    # Return predicted location
    return jsonify({"predicted_location": predicted_xyz[0]})

@app.route("/mlp", methods=["POST"])
def predict_location_mlp():
    # Receive JSON data from POST request
    request_body = request.get_json()
    
    new_data = pd.DataFrame(index=[0])

    # Mengisi nilai-nilai AP yang tidak terdeteksi dengan 0
    for ap_feature in all_ap_features:
        new_data[ap_feature] = 0

    # Mengisi nilai-nilai AP yang tidak terdeteksi dengan nilai dari request_body
    new_data.update(pd.DataFrame.from_dict(request_body, orient='index').transpose())


    # Normalize new data
    new_data_scaled = scaler.transform(new_data)

    predicted_location = mlp_model.predict(new_data_scaled)
    # Ambil indeks dengan nilai tertinggi dari setiap baris dalam predicted_location
    predicted_labels = np.argmax(predicted_location, axis=1)

    # Invers transform label menjadi nilai asli xyz
    predicted_xyz = label_encoder.inverse_transform(predicted_labels)

    # Return predicted location
    return jsonify({"predicted_location": predicted_xyz[0]})


if __name__ == "__main__":
    app.run(debug=True)
