
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

#Funcion de comparacion entre pixeles vecinos: Toma dos pixeles y compara sus valores
def comp_pix(pixelref,pixelcomp):
    if pixelcomp>pixelref:
        pxlval=1                   #Si es mayor el pixel vecino que el central -> 1
    else:
        pxlval=0                   #Si es menor el pixel vecino que el central -> 0
    return pxlval


#Funcion encargada de la comparativa entre pixeles propia del metodo LBP (Un pixel con sus 8 vecinos)
def LBP_pixel(imagen, x, y):
    yprev = y - 1
    ynext = y + 1
    xprev = x - 1
    xnext = x + 1
    ul = comp_pix(imagen[y, x], imagen[yprev, xprev])   #Estudio de la comparacion entre el
    uc = comp_pix(imagen[y, x], imagen[yprev, x])       #pixel en cuestion y sus 8 vecinos
    ur = comp_pix(imagen[y, x], imagen[yprev, xnext])
    cr = comp_pix(imagen[y, x], imagen[y, xnext])
    dr = comp_pix(imagen[y, x], imagen[ynext, xnext])
    dc = comp_pix(imagen[y, x], imagen[ynext, x])
    dl = comp_pix(imagen[y, x], imagen[ynext, xprev])
    cl = comp_pix(imagen[y, x], imagen[y, xprev])
    pixlist = [ul, uc, ur, cr, dr, dc, dl, cl]          #Lista con el codigo binario asociado al pixel
    return pixlist


#Funcion encargada de aplicar el algoritmo LBP dada un cierto numero de vecinos
def LBP_basic(subimgn):
    sub_height = np.size(subimgn, 0)
    sub_width = np.size(subimgn, 1)
    lbp_subimgn = np.zeros((sub_height-2, sub_width-2), dtype=np.uint8)           #Definicion de la matriz con la solucion
    lbp_subimgn = lbp_subimgn.astype(np.float32)
    for y in range(1, sub_height-1):                                              #Bucles para estudiar todos los pixeles
        for x in range(1, sub_width-1):
            pixell = LBP_pixel(subimgn, x, y)                                     #Obtencion del nuevo valor del pixel
            for j in range(8):                                                    #pasando el codigo binario a decimal
                lbp_subimgn[y-1,x-1] = lbp_subimgn[y-1,x-1] + pixell[j]*2**(7-j)
    hist = cv2.calcHist([lbp_subimgn], [0], None, [256], [0, 255])                #Calculo del histograma de la sub-imagen
    n_hist = cv2.normalize(hist,hist)                                             #Normalizacion de dicho histograma
    return n_hist


#Funcion encargada de la definicion de las regiones y aplicar el algoritmo en ellas
def LBPbas_compute(img, wind, block, shift):
    img=np.pad(img, pad_width=1, mode='constant', constant_values=0)  #Hacemos un padding a la imagen total
    final_hist = []
    for i in range(0, wind[0]-block[0]+1, shift[0]):
        for j in range(0, wind[1]-block[1]+1, shift[1]):          #Barrido de las distintas ventanas en la imagen
              x1 = j                                              #Tomamos bloques de 18x18 para aplicar padding
              x2 = x1 + block[1] + 2
              y1 = i                                              #Barremos las celdas dentro de la ventana definida
              y2 = y1 + block[0] + 2
              img_reg = img[y1:y2, x1:x2]                         #Aplicamos sobre esa celda el algoritmo LBP
              lbp_feat = LBP_basic(img_reg)
              final_hist = np.append(final_hist,lbp_feat)         #Obtencion de la solucion con los histogramas concatenados
    sol_lbp=final_hist
    return sol_lbp


#Funcion que llama a cada una de las fotos y le aplica el algoritmo LBP
#Se crea una matriz con las caracteristicas de las imagenes y un vector con la clase asociada
def hoglbp_images(dataset, dir_path, wind, block, shift):
    data_path = dir_path + 'data/' + dataset + '/'
    pdst_path = data_path + 'pedestrians/'                              #Definicion de la localizacion de las imagenes
    back_path = data_path + 'background/'
    tot_pdst_files = len(os.listdir(pdst_path))
    tot_back_files = len(os.listdir(back_path))

    pedestrian_class = np.ones(tot_pdst_files)                          #Definicion del vector con las clases
    background_class = np.zeros(tot_back_files)                         #asociadas a cada imagen
    images_class = np.concatenate((pedestrian_class,background_class))
    print(images_class.shape)
    for imgpdst_file in os.listdir(pdst_path):
        img = cv2.imread(pdst_path + imgpdst_file,0)                     #Bucle de las imagenes de personas en el que
        lbp_pdst = np.transpose(LBPbas_compute(img,wind,block,shift))    #toma cada foto y extrae sus caracteristicas
        hog_pdst = hog.compute(img)                                      #hog y lbp y las anyade a la matriz
        feat_pdst = np.concatenate((lbp_pdst,hog_pdst),axis=None)
        print(feat_pdst.shape)
        try:
            images_mat = np.vstack((images_mat,feat_pdst))
        except NameError:
            images_mat = feat_pdst

    for imgback_file in os.listdir(back_path):
        img = cv2.imread(back_path + imgback_file,0)                     #Bucle de las imagenes de personas en el que
        lbp_back = LBPbas_compute(img,wind,block,shift)                  #toma cada foto y extrae sus caracteristicas
        hog_back = hog.compute(img)                                      #hog y lbp y las anyade a la matriz
        feat_back = np.concatenate((lbp_back,hog_back),axis=None)

        try:
            images_mat = np.vstack((images_mat,feat_back))
        except NameError:
            images_mat = feat_back
    return images_mat, images_class


#Definidicion de las caracteristicas del calculo
wind_size=(128,64)
block_size=(16,16)
shift_size=(8,8)


#Aplicacion del algoritmo para estudiar las caracteristicas de las imagenes de train
train_dat, train_class = hoglbp_images('train', p_path, wind_size, block_size, shift_size)

#Aplicacion del algoritmo para estudiar las caracteristicas de las imagenes de test
test_dat, test_class = hoglbp_images('test',p_path, wind_size, block_size, shift_size)

#Definicion de los parametros considerados en la busqueda de los mejores utilizables en el clasificador
tuned_parameters = [{'kernel': ['rbf'],'C': [0.1,1,10],'gamma': [0.01,0.1,1]},
                    {'kernel': ['linear'],'C': [0.1,1,10],'gamma': [0.01,0.1,1]},
                    {'kernel': ['poly'],'C': [0.1,1,10],'gamma': [0.01,0.1,1],'degree': [2, 3]}]


#Definicion de las medidas de precision a utilizar
scores = {'accuracy': make_scorer(accuracy_score),'f1': make_scorer(f1_score), 'AUC': make_scorer(roc_auc_score)}

#Aplicacion del clasificador SVM para buscar los mejores parametros
svc = SVC()
cv_tune = GridSearchCV(estimator=svc, param_grid=tuned_parameters, n_jobs=3, scoring=scores,
                       cv=StratifiedKFold(n_splits=5), refit='accuracy', return_train_score=True, verbose=10)

cv_tune.fit(train_dat, train_class)

#Guardamos esto en un fichero
import pickle
with open("hoglbp_class.pickle", "wb") as f:
    pickle.dump(cv_tune, f)


#Obtencion en array de los resultados de la busqueda de parametros por validacion cruzada
hoglbp_data = np.asarray(cv_tune.cv_results_['params'])
for i in range(0,len(hoglbp_data)):
    hoglbp_result_list = list(hoglbp_data[i].values())
    if len(hoglbp_result_list)<4:
        hoglbp_result_list.insert(1,"-")
    try:
        hoglbp_result_mat = np.vstack((hoglbp_result_mat, hoglbp_result_list))
    except NameError:
        hoglbp_result_mat = hoglbp_result_list
hoglbp_accuracy = cv_tune.cv_results_['mean_test_accuracy']
hoglbp_auc = cv_tune.cv_results_['mean_test_AUC']
hoglbp_f1 = cv_tune.cv_results_['mean_test_f1']
hoglbp_scores_sum = np.vstack((hoglbp_accuracy,hoglbp_auc,hoglbp_f1))
hoglbp_scores_mat = np.transpose(hoglbp_scores_sum)
hoglbp_latex_summary = np.hstack((hoglbp_result_mat, hoglbp_scores_mat))
ascii.write(hoglbp_latex_summary, format='latex', names=['C','Degree','$\gamma$','Kernel','Accuracy','AUC','F1'])

#Resultados de la prediccion de los datos de test utilizando el modelo optimo
test_pred = cv_tune.predict(test_dat)
print(cv_tune.best_params_)
print(cv_tune.best_score_)
print(confusion_matrix(test_class, test_pred))