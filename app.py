from flask import Flask,render_template,request
import joblib

#intiliaze the app
app = Flask(__name__)
model = joblib.load('dib_79.pkl')
print('[INFO] model loaded')

#__name__ refers that this file main file in the module

@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/predict' , methods = ['post'])
def predict():
    preg = request.form.get('preg')
    plas = request.form.get('plas')
    pres = request.form.get('pres')
    skin = request.form.get('skin')
    test = request.form.get('test')
    mass = request.form.get('mass')
    pedi = request.form.get('pedi')
    age = request.form.get('age')

    output = model.predict([[preg,plas,pres,skin,test,mass,pedi,age]])
    if output[0]==1:
        print('dibatic')
        result = 'dibatic'
    else:
        print('not dibatic')
        result = 'not dibatic'
    return render_template('predict.html',predict=f'Your Results are {result}')

# run the app
if __name__ == '__main__':
    app.run(debug=True)


