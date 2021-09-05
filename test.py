import joblib
dib_model = joblib.load('dib_79.pkl')
output = dib_model.predict([[1 , 85, 66, 1,1,1,1,1]])
print(output)
if output[0] == 0:
	    print('not diabatic')
	
else:
	    print('dibatic')


	