import time
import random
from flask import Flask, request, render_template,redirect,session
import os

app = Flask(__name__)
app.secret_key=os.urandom(24)
from commons import get_tensor
from inference import get_recipe

@app.route('/', methods=['GET', 'POST'])
def hello_world():    
	if request.method == 'GET':
		return render_template('welcome.html', value='hi')
	if request.method == 'POST':
		print(request.files)
		if 'file' not in request.files:
			print('file not uploaded')
			return
		file = request.files['file']
		image = file.read()
		a,b,c,len1,len2 = get_recipe(image_bytes=image)
		
		return render_template('result.html', title=a, ingrs=b,recipe=c,len1=len1,len2=len2)
    
    
if __name__ == '__main__':
	app.run(debug=False)
