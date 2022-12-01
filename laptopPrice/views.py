from django.shortcuts import render
from django.http import HttpResponse
import joblib
import pandas as pd
import numpy as np
import sklearn


# Create your views here.

def index(request):
    return render(request,'index.html',context={})


def predict(request):
    if request.method == 'POST':
        company = request.POST['brands']
        typeName = request.POST['typeName']
        ram = request.POST['ram']
        weight = request.POST['weight']
        touchScreen = request.POST['touchScreen']
        ips = request.POST['ips']
        resolution = request.POST['resolution']
        cpu = request.POST['cpu']
        hdd = request.POST['hdd']
        ssd = request.POST['ssd']
        gpu = request.POST['gpu']
        os = request.POST['os']
        size = request.POST['size']

        X_resolution = int(resolution.split('x')[0])
        y_resolution = int(resolution.split('x')[1])

        ppi = ((X_resolution ** 2) + (y_resolution ** 2)) ** 0.5 / float(size)

        df = pd.DataFrame(data = [[company,typeName,int(ram),float(weight),int(touchScreen),int(ips),ppi,cpu,int(hdd),int(ssd),gpu,os]], columns=['Company','TypeName','Ram','Weight','Touchscreen','Ips','ppi','Cpu brand','HDD','SSD','Gpu brand','os'])

        model = joblib.load('laptopPrice.sav')

        y_pred = np.exp(model.predict(df))

        return render(request,'predict.html',context={'predictedValue':int(y_pred[0])})

    return render(request,'predict.html',context={})
