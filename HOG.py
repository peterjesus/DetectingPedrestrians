import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import os.path
from sklearn.metrics import make_scorer, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import pickle
import pprint
from astropy.io import ascii

p_path = '/Users/pj/Desktop/ECI.Practica/'

#Definicion del descriptor HOG
hog = cv2.HOGDescriptor()

#Funcion que llama a cada una de las fotos y le aplica el algoritmo HOG
#Se crea una matriz con las caracteristicas de las imagenes y un vector con la clase asociada
def hog_images(dataset,dir_path):
    data_path = dir_path + 'data/' + dataset + '/'
    pdst_path = data_path + 'pedestrians/'                              #Definicion de la localizacion de las imagenes
    back_path = data_path + 'background/'
    tot_pdst_files = len(os.listdir(pdst_path))
    tot_back_files = len(os.listdir(back_path))

    pedestrian_class = np.ones(tot_pdst_files)                          #Definicion del vector con las clases
    background_class = np.zeros(tot_back_files)                         #asociadas a cada imagen
    images_class = np.concatenate((pedestrian_class,background_class))

    for imgpdst_file in os.listdir(pdst_path):
        img = cv2.imread(pdst_path + imgpdst_file,0)                    #Bucle de las imagenes de personas en el que
        hog_pdst = hog.compute(img)                                     #toma cada foto y extrae sus caracteristicas y
        try:                                                            #las anyade a la matriz
            images_mat = np.hstack((images_mat,hog_pdst))
        except NameError:
            images_mat = hog_pdst

    for imgback_file in os.listdir(back_path):
        img = cv2.imread(back_path + imgback_file,0)                    #Bucle de las imagenes de fondos en el que
        hog_back = hog.compute(img)                                     #toma cada foto y extrae sus caracteristicas y
        try:                                                            #las anyade a la matriz
            images_mat = np.hstack((images_mat,hog_back))
        except NameError:
            images_mat = hog_back

    return np.transpose(images_mat), images_class


#Aplicacion del algoritmo para estudiar las caracteristicas de las imagenes de train
train_dat, train_class = hog_images(dataset='train',dir_path=p_path)

#Aplicacion del algoritmo para estudiar las caracteristicas de las imagenes de test
test_dat, test_class = hog_images(dataset='test',dir_path=p_path)

#Definicion de los parametros considerados en la busqueda de los mejores utilizables en el clasificador
tuned_parameters = [{'kernel': ['rbf'],'C': [0.1,1,10],'gamma': [0.01,0.1,1]},
                    {'kernel': ['linear'],'C': [0.1,1,10],'gamma': [0.01,0.1,1]},
                    {'kernel': ['poly'],'C': [0.1,1,10],'gamma': [0.01,0.1,1],'degree': [2, 3]}]

#Definicion de las medidas de precision a utilizar
scores = {'accuracy': make_scorer(accuracy_score),'f1': make_scorer(f1_score), 'AUC': make_scorer(roc_auc_score)}

#Aplicacion del clasificador SVM para buscar los mejores parametros
svc = SVC()
cv_tune = GridSearchCV(estimator=svc, param_grid=tuned_parameters, n_jobs=-1, scoring=scores,
                       cv=StratifiedKFold(n_splits=5), refit='accuracy', return_train_score=True, verbose=10)
cv_tune.fit(train_dat, train_class)


#Guardamos los resultados obtenidos
with open("hog_class.pickle", "wb") as f:
    pickle.dump(cv_tune, f)

#Obtencion en array de los resultados de la busqueda de parametros por validacion cruzada
hog_data=np.asarray(cv_tune.cv_results_['params'])
for i in range(0,len(hog_data)):
    hog_result_list = list(hog_data[i].values())
    if len(hog_result_list)<4:
        hog_result_list.insert(1,"-")
    try:
        hog_result_mat = np.vstack((hog_result_mat, hog_result_list))
    except NameError:
        hog_result_mat = hog_result_list
hog_accuracy = cv_tune.cv_results_['mean_test_accuracy']
hog_auc = cv_tune.cv_results_['mean_test_AUC']
hog_f1 = cv_tune.cv_results_['mean_test_f1']
hog_scores_sum = np.vstack((hog_accuracy,hog_auc,hog_f1))
hog_scores_mat = np.transpose(hog_scores_sum)
hog_latex_summary = np.hstack((hog_result_mat, hog_scores_mat))
ascii.write(hog_latex_summary,format='latex',names=['C','Degree','$\gamma$','Kernel','Accuracy','AUC','F1'])

#Resultados de la prediccion de los datos de test utilizando el modelo optimo
test_pred = cv_tune.predict(test_dat)
print(cv_tune.best_params_)
print(cv_tune.best_score_)
print(confusion_matrix(test_class, test_pred))