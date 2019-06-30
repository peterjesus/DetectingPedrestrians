
import math
import numpy as np
import cv2
from sklearn.svm import SVC
import os.path
import pickle
import math

p_path = '/Users/pj/Desktop/'

########################################################################################################################
###HOG
#Definicion del descriptor HOG
hog = cv2.HOGDescriptor()
pickle_hog = open("hog_class.pickle","rb")
hog_cv = pickle.load(pickle_hog)
########################################################################################################################
###LBP
#Definicion del descriptor LBP
pickle_lbp = open("lbp_class.pickle","rb")
lbp_cv = pickle.load(pickle_lbp)
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
########################################################################################################################
###U-LBP
#Definicion del descriptor u-LBP
pickle_ulbp = open("ulbp_class.pickle","rb")
ulbp_cv = pickle.load(pickle_ulbp)
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
########################################################################################################################
###HOG-LBP
#Definicion del descriptor HOG-LBP
pickle_hoglbp = open("hoglbp_class.pickle","rb")
hoglbp_cv = pickle.load(pickle_hoglbp)
########################################################################################################################


#Funcion que realiza la división de la imagen, calculando el numero maximo de ventanas en las dos direcciones
def imageSplit(imgn,wind,shift):
    height = np.size(imgn, 0)
    width = np.size(imgn, 1)
    hw_fct = math.floor(height/wind[0])
    hb_fct = math.floor((height-hw_fct*wind[0])/shift[0])
    y_max_window = int(hw_fct*wind[0]-((wind[0]/shift[0])-hb_fct)*shift[0])+1
    ww_fct = math.floor(width/wind[1])
    wb_fct = math.floor((width-ww_fct*wind[1])/shift[1])
    x_max_window = int(ww_fct*wind[1]-((wind[1]/shift[1])-wb_fct)*shift[1])+1
    if x_max_window < 0:
        x_max_window = 1
    if y_max_window < 0:
        y_max_window = 1
    return y_max_window,x_max_window

#Caracteristicas del calculo e imagen a analizar, asi como el modelo a utilizar
wind_size=(128,64)
shiftwin_size=(16,16)
block_size=(16,16)
shift_size=(8,8)
diccpxl = bitsDicc(8)

#Imagen a estudiar y metodo utilizado
Image= cv2.imread(os.path.join(p_path,'Ej1.jpg'),0)
Imagergb= cv2.imread(os.path.join(p_path,'Ej1.jpg'),1)
describe='hog'

#Bucle que barre la imagen con diferentes escalas definidas segun el sc_factor y el scale
scale = 0.1                                                        #Escalado de la imagen
start = 5
finish= 10                                                         #Intervalos de escalado
for sc_factor in range(start,finish):
    list_box = []
    scl_imgn = cv2.resize(Image, None, fx=scale * sc_factor, fy=scale * sc_factor, interpolation=cv2.INTER_LINEAR)
    height = np.size(scl_imgn, 0)
    width = np.size(scl_imgn, 1)
    y_wind_max, x_wind_max = imageSplit(scl_imgn,wind_size,shiftwin_size)  #Definicion de los maximos de ventanas
    if height>=wind_size[0] and width>=wind_size[1]:                #Condicion; Solo si es posible analizar la ventana
        for y_mov in range(0,y_wind_max, shiftwin_size[1]):
            for x_mov in range(0,x_wind_max, shiftwin_size[0]):
              x1 = x_mov
              x2 = x1 + wind_size[1]                                #Definicion de las coordenadas de la ventana
              y1 = y_mov                                            #dentro de la imagen correspondiente
              y2 = y1 + wind_size[0]
              imgn_wind = scl_imgn[y1:y2, x1:x2]
              #scl_imgn_wind = cv2.resize(imgn_wind, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
              #cv2.imshow('image', scl_imgn_wind)                    #Visualizacion de la ventana a mas tamanyo
              #cv2.waitKey(10)
              #cv2.destroyAllWindows()                               #Realizacion del metodo de analisis de los
              if describe=='hoglbp':                                 #rasgos de la ventana
                lbp_wind = np.transpose(LBPbas_compute(imgn_wind, wind_size, block_size, shift_size))
                hog_wind = hog.compute(imgn_wind)
                feat_wind = np.concatenate((lbp_wind, hog_wind), axis=None)
                feat_lbphog = np.transpose(feat_wind.reshape(-1, 1))
                predict_hoglbp = hoglbp_cv.predict(np.array(feat_lbphog))
                if predict_hoglbp > 0.5:
                    list_box.append([int(x1 / (scale * sc_factor)), int(x2 / (scale * sc_factor)),
                                 int(y1 / (scale * sc_factor)), int(y2 / (scale* sc_factor))])
              elif describe=='hog':
                feat_wind = hog.compute(imgn_wind)
                feat_hog = (np.transpose(feat_wind))
                predict_hog = hog_cv.predict(np.array(feat_hog))
                if predict_hog > 0.5:
                   list_box.append([int(x1 / (scale * sc_factor)), int(x2 / (scale * sc_factor)),
                                  int(y1 / (scale * sc_factor)), int(y2 / (scale* sc_factor))])
              elif describe=='lbp':
                feat_lbp = LBPbas_compute(imgn_wind,wind_size,block_size,shift_size)
                predict_lbp = lbp_cv.predict(np.array(feat_lbp).reshape(1, -1))
                if predict_lbp > 0.5:
                   list_box.append([int(x1/(scale*sc_factor)),int(x2/(scale*sc_factor)),
                                  int(y1/(scale*sc_factor)),int(y2/(scale*sc_factor))])
              elif describe=='ulbp':
                feat_ulbp = LBPunif_compute(imgn_wind, wind_size, block_size, shift_size, diccpxl)
                predict_ulbp = ulbp_cv.predict(np.array(feat_ulbp).reshape(1, -1))
                if predict_ulbp > 0.5:
                    list_box.append([int(x1 / (scale * sc_factor)), int(x2 / (scale * sc_factor)),
                                  int(y1 / (scale * sc_factor)), int(y2 / (scale* sc_factor))])


#Funcion que calcula el "intersection over union" de dos rectangulos
def intersect_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])                                     #Coordenadas del rectangulo de interseccion
    xB = min(boxA[1], boxB[1])                                     #definido segun los dos rectangulos considerados
    yA = max(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[1] - boxA[0] + 1) * (boxA[3] - boxA[2] + 1)   #Ares de los rectangulos y de la interseccion
    boxBArea = (boxB[1] - boxB[0] + 1) * (boxB[3] - boxB[2] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)       #Definicion "intersection over union"
    return iou

#Funcion que calcula si dos triangulos interseccionan (Devuelve True en caso afirmativo)
def rectangle_intersect(one, other):
    flag = False
    if one[1]>other[0] and one[0]<other[1] and one[3]>other[2] and one[2]<other[3]:
        flag=True
    return flag

#Funcion que dado una lista de rectangulos calcula el nuevo rectangulo
#para aquellos con una cierta area de interseccion superior a un threshold
def iou_rectangle_merge(list,index_box,threshold):
    rmv_box=[]
    for i in range(0, len(list)):
        if i != index_box:
            overl = rectangle_intersect(list[index_box], list[i])
            if overl == True:
              iou=intersect_over_union(list[index_box], list[i])
              if iou>threshold:
                 a = list[index_box]
                 b = list[i]
                 x1new = min(a[0], b[0])
                 x2new = max(a[1], b[1])
                 y1new = min(a[2], b[2])
                 y2new = max(a[3], b[3])
                 list[index_box] = [x1new, x2new, y1new, y2new]
                 rmv_box.append(i)
    rmv_box.sort(reverse=True)
    return list,rmv_box

#Dibujo de los rectangulos originales
for i in range(0,len(list_box)):
    box_data = list_box[i]
    x1_draw = box_data[0]
    x2_draw = box_data[1]
    y1_draw = box_data[2]
    y2_draw = box_data[3]
    cv2.rectangle(Imagergb, (x1_draw, y1_draw), (x2_draw, y2_draw), (0, 255, 0), 1)

#Dibujo de los rectangulos obtenidos por el metodo de clustering
rectList, weights = cv2.groupRectangles(list_box, 1, 0.2)
for i in range(0,len(rectList)):
    box_data = rectList[i]
    x1_draw = box_data[0]
    x2_draw = box_data[1]
    y1_draw = box_data[2]
    y2_draw = box_data[3]
    cv2.rectangle(Imagergb, (x1_draw, y1_draw), (x2_draw, y2_draw), (255, 0, 0), 2)

#Dibujo de los rectangulos originales
k,indxs=0,len(list_box)
while k<indxs:
    list_box,remv_box = iou_rectangle_merge(list_box,k,0.2)
    for index in remv_box:
            del list_box[index]
    indxs = len(list_box)
    k+=1

for i in range(0,len(list_box)):
    box_data = list_box[i]
    x1_draw = box_data[0]
    x2_draw = box_data[1]
    y1_draw = box_data[2]
    y2_draw = box_data[3]
    cv2.rectangle(Imagergb, (x1_draw, y1_draw), (x2_draw, y2_draw), (0, 0, 255), 2)

#Guardamos la imagen final
cv2.imwrite('/Users/pj/Desktop/imagen_ulbp.jpg',Imagergb)