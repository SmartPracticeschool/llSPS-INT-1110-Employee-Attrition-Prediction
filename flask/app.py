import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
oh=pickle.load(open('oneencoder.pkl','rb'))
lb=pickle.load(open('labelencoder.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test=[[x for x in request.form.values()]]
    x_test[0][0]=int(x_test[0][0])
    x_test[0][2]=int(x_test[0][2])
    x_test[0][4]=int(x_test[0][4])
    x_test[0][5]=int(x_test[0][5])
    x_test[0][7]=int(x_test[0][7])
    x_test[0][8]=int(x_test[0][8])
    x_test[0][9]=int(x_test[0][9])
    x_test[0][11]=int(x_test[0][11])
    x_test[0][12]=int(x_test[0][12])
    x_test[0][13]=int(x_test[0][13])
    x_test[0][15]=int(x_test[0][15])
    x_test[0][17]=int(x_test[0][17])
    x_test[0][18]=int(x_test[0][18])
    x_test[0][19]=int(x_test[0][19])
    x_test[0][22]=int(x_test[0][22])
    x_test[0][23]=int(x_test[0][23])
    x_test[0][24]=int(x_test[0][24])
    x_test[0][25]=int(x_test[0][25])
    x_test[0][26]=int(x_test[0][26])
    x_test[0][27]=int(x_test[0][27])
    x_test[0][28]=int(x_test[0][28])
    x_test[0][29]=int(x_test[0][29])
    x_test[0][30]=int(x_test[0][30])
    x_test[0][31]=int(x_test[0][31])
    x_test[0][32]=int(x_test[0][32])
    x_test[0][33]=int(x_test[0][33])

   # x_test=oh.transform(x_test).toarray()
   # x_test=lb.transform(x_test).toarray()
    x_test=oh.transform(x_test).toarray()
   # x_test=lb.transform(x_test).toarray()
    x_test = np.array(x_test)
    prediction=model.predict(x_test)
    output=prediction[0]
    if(output==0):
        return render_template('index.html',prediction_text='no')
    else:
        return render_template('index.html',prediction_text='yes')

if __name__=="__main__":
    app.run(debug=True)
