
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier

# Create your views here.
from Remote_User.models import ClientRegister_Model,Social_Media,detection_ratio,detection_accuracy


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')

def Find_Social_Media_News_Type_Ratio(request):
    detection_ratio.objects.all().delete()
    ratio = ""
    kword = 'Fake'
    print(kword)
    obj = Social_Media.objects.all().filter(Q(Prediction=kword))
    obj1 = Social_Media.objects.all()
    count = obj.count();
    count1 = obj1.count();
    ratio = (count / count1) * 100
    if ratio != 0:
        detection_ratio.objects.create(names=kword, ratio=ratio)

    ratio1 = ""
    kword1 = 'True'
    print(kword1)
    obj1 = Social_Media.objects.all().filter(Q(Prediction=kword1))
    obj11 = Social_Media.objects.all()
    count1 = obj1.count();
    count11 = obj11.count();
    ratio1 = (count1 / count11) * 100
    if ratio1 != 0:
        detection_ratio.objects.create(names=kword1, ratio=ratio1)


    obj = detection_ratio.objects.all()
    return render(request, 'SProvider/Find_Social_Media_News_Type_Ratio.html', {'objs': obj})

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def ViewTrendings(request):
    topic = Social_Media.objects.values('topics').annotate(dcount=Count('topics')).order_by('-dcount')
    return  render(request,'SProvider/ViewTrendings.html',{'objects':topic})

def charts(request,chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def Predict_Social_Media_News_Type(request):
    obj =Social_Media.objects.all()
    return render(request, 'SProvider/Predict_Social_Media_News_Type.html', {'list_objects': obj})

def likeschart(request,like_chart):
    charts =detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})


def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="TrainedData.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = Social_Media.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1

        ws.write(row_num, 0, my_row.News_Data, font_style)
        ws.write(row_num, 1, my_row.Prediction, font_style)

    wb.save(response)
    return response

def Train_Test_DataSets(request):
    detection_accuracy.objects.all().delete()
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")
    df_fake.head()
    df_true.head(5)
    df_fake["class"] = 0
    df_true["class"] = 1
    df_fake.shape, df_true.shape
    # Removing last 10 rows for manual testing
    df_fake_manual_testing = df_fake.tail(10)
    for i in range(23480, 23470, -1):
        df_fake.drop([i], axis=0, inplace=True)

    df_true_manual_testing = df_true.tail(10)
    for i in range(21416, 21406, -1):
        df_true.drop([i], axis=0, inplace=True)
        df_fake.shape, df_true.shape
        df_fake_manual_testing["class"] = 0
        df_true_manual_testing["class"] = 1
        df_fake_manual_testing.head(10)
        df_true_manual_testing.head(10)
        df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
        df_manual_testing.to_csv("manual_testing.csv")
        df_merge = pd.concat([df_fake, df_true], axis=0)
        df_merge.head(10)
        df_merge.columns
        df = df_merge.drop(["title", "subject", "date"], axis=1)
        df.isnull().sum()
        df = df.sample(frac=1)
        df.head()
        df.reset_index(inplace=True)
        df.drop(["index"], axis=1, inplace=True)
        df.columns
        df.head()

        def wordopt(text):
            text = text.lower()
            text = re.sub('\[.*?\]', '', text)
            text = re.sub("\\W", " ", text)
            text = re.sub('https?://\S+|www\.\S+', '', text)
            text = re.sub('<.*?>+', '', text)
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub('\n', '', text)
            text = re.sub('\w*\d\w*', '', text)
            return text

    cv = CountVectorizer()
    df["text"] = df["text"].apply(wordopt)
    x = df["text"]
    y = df["class"]

    x = cv.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    from sklearn.linear_model import LogisticRegression

    print("Logistic Regression")
    LR = LogisticRegression()
    LR.fit(x_train, y_train)
    pred_lr = LR.predict(x_test)
    LR.score(x_test, y_test)
    print("ACCURACY")
    print(accuracy_score(y_test, pred_lr) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, pred_lr))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, pred_lr))

    detection_accuracy.objects.create(names="Logistic Regression", ratio=accuracy_score(y_test, pred_lr) * 100)

    print("Decision Tree Classifier")

    from sklearn.tree import DecisionTreeClassifier
    DT = DecisionTreeClassifier()
    DT.fit(x_train, y_train)
    pred_dt = DT.predict(x_test)
    DT.score(x_test, y_test)
    print("ACCURACY")
    print(accuracy_score(y_test, pred_dt) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, pred_dt))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, pred_dt))

    detection_accuracy.objects.create(names="Decision Tree Classifier", ratio=accuracy_score(y_test, pred_dt) * 100)

    print("Random Forest Classifier")
    from sklearn.ensemble import RandomForestClassifier
    RFC = RandomForestClassifier(random_state=0)
    RFC.fit(x_train, y_train)
    pred_rfc = RFC.predict(x_test)
    RFC.score(x_test, y_test)
    print("ACCURACY")
    print(accuracy_score(y_test, pred_rfc) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, pred_rfc))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, pred_rfc))

    detection_accuracy.objects.create(names="Random Forest Classifier", ratio=accuracy_score(y_test, pred_rfc) * 100)

    obj = detection_accuracy.objects.all()
    return render(request,'SProvider/Train_Test_DataSets.html', {'objs': obj})