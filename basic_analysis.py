from sklearn.preprocessing import scale,normalize
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd

data = pd.read_csv('states.csv')
#X = data[['max_try','total_try','correct_rate']].values.astype(float)
X = data[['correct_rate']].values.astype(float)
Y = data['11612'].values


#X = scale(X,axis=0)


#accs  = []
#
#for i in range(10):
#    trainx,testx,trainy,testy = train_test_split(X,Y,test_size=.2)
#    model = LogisticRegression(C=1e-5)
#    model.fit(trainx,trainy)
#    preds = model.predict(testx)
#    acc = accuracy_score(testy,preds)
#    cfm = confusion_matrix(testy,preds)
#    print(acc)
#    print(cfm)
#    print('-------')
#    accs.append(acc)
#    
    
import matplotlib.pyplot as plt

pos_inds = Y==1
pos_x    = X[pos_inds,:]
plt.hist(pos_x)

neg_inds = Y==0
neg_x    = X[neg_inds,:]
plt.hist(neg_x)


