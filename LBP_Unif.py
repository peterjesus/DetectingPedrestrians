
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

#Funcion encargada de decir si hay transición entre dos elementos consecutivos
def jump(i,j):
    if i == j:
        res = 0         #Si los dos elementos en cuestion son iguales, no hay salto
    else:
        res = 1         #Si los dos elementos no son el mismo, hay salto
    return res


#Funcion encargada de decir si una secuencia binaria es uniforme o no
def cyclic(listbin,label):                                  #Se introduce una etiqueta que sera el valor anteriormente
    cont = 0                                                #almacenado para asi tomar la inmediatamente siguiente
    for i in range(len(listbin)):                           #Sestudian las transciones en el codigo
        if i == (len(listbin)-1):
            cont = cont + jump(listbin[i], listbin[0])      #Estudio del ultimo elemento, que se enlaza con el primero
        else:
            cont = cont + jump(listbin[i], listbin[i+1])    #Estudio de los elementos centrales
    if cont <= 2:
         etiq = label + 1
         label += 1                                         #Si el codigo es uniforme, se anyade la siguiente etiqueta
    else:
         etiq = -1                                          #Si no es uniforme, se anyade como etiqueta un -1
    return etiq, label


#Funcion encargada de convertir la lista con el codigo binario en una cadena
#(Mas facil para su almacenamiento en el diccionario en el que se incorporara)
def convert(list):
    s = [str(i) for i in list]
    ress = "".join(s)
    return ress


#Funcion que recibe el numero de bits y realiza todas las combinaciones posibles de valores
def bitsDicc(bits):
    dicc=dict()
    label=-1
    for i in range(0,2**bits):                          #Toma todos los numeros decimales entre 0 y el maximo 2^^nbits
        bitlist = []
        if i == 0:                                      #A continuacion, se transforma este numero decimal en un binario
            bitlist.append(0)                           #Para ello, se anyadiran 0 o 1 segun corresponda a una lista
        while i:                                        #la cual finalmente contendra el codigo binario asoiado
            if i & 1 == 1:
                bitlist.append(1)
            else:
                bitlist.append(0)
            i //= 2
        k = bits - len(bitlist)
        for j in range(k):
            bitlist.append(0)
        bitlist.reverse()
        valbit, label = cyclic(bitlist, label)          #Se estudia si el codigo en cuestion es uniforme o no
        strnum = convert(bitlist)                       #Se pasa la lista a cadena
        dicc[strnum] = valbit                           #Se define en el diccionario una entrada y valor
    keymax = max(dicc.keys(), key=(lambda k: dicc[k]))
    for i in dicc:                                      #Adicion de la etiqueta ultima a aquellos elementos
        if dicc[i] < 0:                                 #del diccionario de codigos que no son uniformes
            dicc[i] = dicc[keymax] + 1
    return dicc


#Funcion encargada de aplicar el algoritmo U-LBP dada un cierto numero de vecinos
#(Requiere la previa construccion del diccionario de codigos dado el numero de vecinos)
def LBP_unif(subimgn,diccpxl):
    sub_height = np.size(subimgn, 0)
    sub_width = np.size(subimgn, 1)
    ulbp_subimgn = np.zeros((sub_height-2, sub_width-2), dtype=np.uint8)  #Definicion de la matriz que contendra la solucion
    for y in range(1, sub_height-1):                                      #Bucles para estudiar todos los pixeles
        for x in range(1, sub_width-1):
            pixelli = LBP_pixel(subimgn, x, y)                            #Obtencion del valor del pixel segun el metodo
            pixelstr = convert(pixelli)                                   #U-LBP, tras obtener el codigo y pasarlo a cadena
            ulbp_subimgn[y-1, x-1] = diccpxl[pixelstr]
    hist = cv2.calcHist([ulbp_subimgn], [0], None, [59], [0, 58])         #Calculo del histograma de la sub-imagen
    n_hist = cv2.normalize(hist, hist)                                    #Normalizacion de dicho histograma
    return n_hist


#Funcion encargada de la definicion de las regiones y aplicar el algoritmo en ellas
def LBPunif_compute(img, wind, block, shift,diccpxl):
    img=np.pad(img, pad_width=1, mode='constant', constant_values=0)  #Hacemos un padding a la imagen total
    final_hist = []                                          #Inicializacion de la solucion
    for i in range(0, wind[0]-shift[0], shift[0]):
        for j in range(0, wind[1]-shift[1], shift[1]):     #Barrido de las distintas ventanas en la imagen
              x1 = j
              x2 = x1 + block[1] + 2
              y1 = i                                         #Barremos las celdas dentro de la ventana definida
              y2 = y1 + block[0] + 2
              img_reg = img[y1:y2, x1:x2]                    #Aplicamos sobre esa celda el algoritmo U-LBP
              ulbp_feat = LBP_unif(img_reg,diccpxl)
              final_hist = np.append(final_hist,ulbp_feat)   #Obtencion de la solucion con los histogramas concatenados
    sol_ulbp=final_hist
    return sol_ulbp



#Funcion que llama a cada una de las fotos y le aplica el algoritmo U-LBP
#Se crea una matriz con las caracteristicas de las imagenes y un vector con la clase asociada
def ulbp_images(dataset,dir_path,wind,block,shift):
    data_path = dir_path + 'data/' + dataset + '/'
    pdst_path = data_path + 'pedestrians/'                              #Definicion de la localizacion de las imagenes
    back_path = data_path + 'background/'
    tot_pdst_files = len(os.listdir(pdst_path))
    tot_back_files = len(os.listdir(back_path))

    pedestrian_class = np.ones(tot_pdst_files)                          #Definicion del vector con las clases
    background_class = np.zeros(tot_back_files)                         #asociadas a cada imagen
    images_class = np.concatenate((pedestrian_class,background_class))

    diccpxl = bitsDicc(8)                                               #Inicializacion diccionario de 8bits

    for imgpdst_file in os.listdir(pdst_path):
        img = cv2.imread(pdst_path + imgpdst_file,0)                    #Bucle de las imagenes de personas en el que
        ulbp_pdst = LBPunif_compute(img,wind,block,shift,diccpxl)       #toma cada foto y extrae sus caracteristicas y
        try:                                                            #las anyade a la matriz
            images_mat = np.vstack((images_mat,ulbp_pdst))
        except NameError:
            images_mat = ulbp_pdst
        print(images_mat.shape)

    for imgback_file in os.listdir(back_path):
        img = cv2.imread(back_path + imgback_file,0)                    #Bucle de las imagenes de fondos en el que
        ulbp_back = LBPunif_compute(img,wind,block,shift,diccpxl)       #toma cada foto y extrae sus caracteristicas y
        try:                                                            #las anyade a la matriz
            images_mat = np.vstack((images_mat,ulbp_back))
        except NameError:
            images_mat = ulbp_back
        print(images_mat.shape)
    return images_mat, images_class


#Definidicion de las caracteristicas del calculo
wind_size=(128,64)
block_size=(16,16)
shift_size=(8,8)


#Aplicacion del algoritmo para estudiar las caracteristicas de las imagenes de train
train_dat, train_class = ulbp_images('train', p_path, wind_size, block_size, shift_size)

#Aplicacion del algoritmo para estudiar las caracteristicas de las imagenes de test
test_dat, test_class = ulbp_images('test',p_path, wind_size, block_size, shift_size)

#Definicion de los parametros considerados en la busqueda de los mejores utilizables en el clasificador
tuned_parameters = [{'kernel': ['rbf'],'C': [0.1,1,10],'gamma': [0.01,0.1,1]},
                    {'kernel': ['linear'],'C': [0.1,1,10],'gamma': [0.01,0.1,1]},
                    {'kernel': ['poly'],'C': [0.1,1,10],'gamma': [0.01,0.1,1],'degree': [2, 3]}]

#Definicion de las medidas de precision a utilizar
scores = {'accuracy': make_scorer(accuracy_score),'f1': make_scorer(f1_score), 'AUC': make_scorer(roc_auc_score)}

#Aplicacion del clasificador SVM para buscar los mejores parametros
svc = SVC()
cv_tune = GridSearchCV(estimator=svc, param_grid=tuned_parameters, n_jobs=2, scoring=scores,
                       cv=StratifiedKFold(n_splits=5), refit='accuracy', return_train_score=True, verbose=10)
cv_tune.fit(train_dat, train_class)

#Guardamos los resultados obtenidos
with open("ulbp_class.pickle", "wb") as f:
    pickle.dump(cv_tune, f)

#Obtencion en array de los resultados de la busqueda de parametros por validacion cruzada
ulbp_data = np.asarray(cv_tune.cv_results_['params'])
for i in range(0,len(ulbp_data)):
    ulbp_result_list = list(ulbp_data[i].values())
    if len(ulbp_result_list)<4:
        ulbp_result_list.insert(1,"-")
    try:
        ulbp_result_mat = np.vstack((ulbp_result_mat, ulbp_result_list))
    except NameError:
        ulbp_result_mat = ulbp_result_list
ulbp_accuracy = cv_tune.cv_results_['mean_test_accuracy']
ulbp_auc = cv_tune.cv_results_['mean_test_AUC']
ulbp_f1 = cv_tune.cv_results_['mean_test_f1']
ulbp_scores_sum = np.vstack((ulbp_accuracy,ulbp_auc,ulbp_f1))
ulbp_scores_mat = np.transpose(ulbp_scores_sum)
ulbp_latex_summary = np.hstack((ulbp_result_mat, ulbp_scores_mat))
ascii.write(ulbp_latex_summary, format='latex', names=['C','Degree','$\gamma$','Kernel','Accuracy','AUC','F1'])

#Resultados de la prediccion de los datos de test utilizando el modelo optimo
test_pred = cv_tune.predict(test_dat)
print(cv_tune.best_params_)
print(cv_tune.best_score_)
print(confusion_matrix(test_class, test_pred))