from flask import Flask,request
#from flask import jsonify
from py_flask import model_run, model_run_json
from flask_jsonpify import jsonpify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict(): 
    #csv_file = request.json['input']
    data = request.get_json(force=True)
    print(data)
    csv_file = data["input"]
    print(csv_file)
    predictions =  model_run(csv_file)
    df = predictions.head(10)
    df_list = list(df.values.flatten())
    JSONP_data = jsonpify(df_list)
    print(JSONP_data)
    return JSONP_data
     
@app.route('/json_predict', methods=['GET','POST'])
def json_predict(): 
    json_data = request.json['input']
    predictions =  model_run_json(json_data)
    df = predictions.head(10)
    df_list = list(df.values.flatten())
    JSONP_data = jsonpify(df_list)
    print(JSONP_data)
    return JSONP_data

if __name__ == '__main__':
    app.run(port=5000,debug=True)    
    
    
#@app.route('/api',methods=['POST'])
#def predict():
#    #data = request.get_json(force=True)
#   prediction = model.predict([[np.array(data['exp'])]])
#    output = prediction[0]
#   return jsonify(output)