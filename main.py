#gerekli kütüphaneler
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#Veri Ön işleme Aşaması

#verilerin okunması ve bağımlı ve bağımsız değişkenlerin ayrılması
veriler = pd.read_excel('Pima.xlsx')
X= veriler.iloc[:,0:-1]
Y = veriler.iloc[:,[-1]]

#Veri setinin k-katlı çapraz doğrulama ile eğitim ve test olarak ayrılması
kfold1 = KFold(n_splits=10, shuffle=True, random_state=42)
yy = kfold1.split(Y)


for train,test in yy:

    #eğitim ve test verileri
    train_X = X.iloc[train]
    y_train = Y.iloc[train]
    test_X = X.iloc[test]
    y_test = Y.iloc[test]

    #Veri Ölçekleme(Normalizasyon)
    sc = StandardScaler()
    x_train = sc.fit_transform(train_X)
    x_test = sc.fit_transform(test_X)

    #Support Vector Machine yönteminin uygulanması
    clf = SVC(kernel='rbf')
    clf.fit(x_train, np.ravel(y_train, order='C'))

    #Tahmin
    pred = clf.predict(x_test)

    # Başarı Ölçümü
    acc = metrics.accuracy_score(y_test,pred)
    rec = metrics.recall_score(y_test,pred)
    pre = metrics.precision_score(y_test,pred)
    print('SVM için Doğruluk Oranı= ',acc)
    print('SVM için Duyarlılık= ', rec)
    print('SVM için Hassasiyet= ', pre)
    print('---------------------')



for train,test in yy:

    #eğitim ve test verileri
    train_X = X.iloc[train]
    y_train = Y.iloc[train]
    test_X = X.iloc[test]
    y_test = Y.iloc[test]

    #Veri Ölçekleme(Normalizasyon)
    sc = StandardScaler()
    x_train = sc.fit_transform(train_X)
    x_test = sc.fit_transform(test_X)

    #K-nn yönteminin uygulanması
    knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
    knn.fit(x_train, np.ravel(y_train, order='C'))

    #Tahmin
    pred = knn.predict(x_test)

    # Başarı Ölçümü
    acc = metrics.accuracy_score(y_test, pred)
    rec = metrics.recall_score(y_test, pred)
    pre = metrics.precision_score(y_test, pred)
    print('K-NN için Doğruluk Oranı= ', acc)
    print('K-NN için Duyarlılık= ', rec)
    print('K-NN için Hassasiyet= ', pre)
    print('---------------------')


for train,test in yy:

    #eğitim ve test verileri
    train_X = X.iloc[train]
    y_train = Y.iloc[train]
    test_X = X.iloc[test]
    y_test = Y.iloc[test]

    #Veri Ölçekleme(Normalizasyon)
    sc = StandardScaler()
    x_train = sc.fit_transform(train_X)
    x_test = sc.fit_transform(test_X)

    #Random Forest Classification yönteminin uygulanması
    rf_reg = RandomForestClassifier(random_state=0, n_estimators=10, criterion='entropy')
    rf_reg.fit(x_train, np.ravel(y_train, order='C'))

    #Tahmin
    pred = rf_reg.predict(x_test)

    # Başarı Ölçümü
    acc = metrics.accuracy_score(y_test,pred)
    rec = metrics.recall_score(y_test,pred)
    pre = metrics.precision_score(y_test,pred)
    print('RFC için Doğruluk Oranı= ',acc)
    print('RFC için Duyarlılık= ', rec)
    print('RFC için Hassasiyet= ', pre)
    print('---------------------')


