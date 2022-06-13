import numpy as np 
import pandas as pd 
import random
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def SVM(X_train : list, y_train : list, x_test : list, y_test: list, regularization = None , graph = False):
    
    x_test = np.c_[x_test, np.ones(len(x_test))]
    x = X_train
    y = y_train
    #add bias to sample vectors
    x = np.c_[x,np.ones(len(x))]

    #initialize weight vector
    w = np.zeros(len(x[0]))
    
    #learning rate 
    lam = 0.001
    #array of number for shuffling
    order = np.arange(0,len(x),1)
    
    margin_current = 0
    margin_previous = -10
    
    pos_support_vectors = 0
    neg_support_vectors = 0
    
    start_time = time.time()
    not_converged = True
    t = 0 
    
    while(not_converged):
        margin_previous = margin_current
        t += 1
        pos_support_vectors = 0
        neg_support_vectors = 0
        
        eta = 1/(lam*t)
        fac = (1-(eta*lam))*w
        random.shuffle(order)
        for i in order:  
            prediction = np.dot(x[i],w)
            
            #check for support vectors
            if (round((prediction),1) == 1):
                pos_support_vectors += 1
                #pos support vec found
            if (round((prediction),1) == -1):
                neg_support_vectors += 1
                #neg support vec found
                
            #misclassification
            if (y[i]*prediction) < 1 :
                w = fac + eta*y[i]*x[i]            
            #correct classification
            else:
                w = fac
                
        if(t>10000):    
            margin_current = np.linalg.norm(w)
            if((pos_support_vectors > 0)and(neg_support_vectors > 0)and((margin_current - margin_previous) < 0.01)):
                not_converged = False

   
    
    y_pred = []
    for i in x_test:
        pred = np.dot(w,i)
        if(pred > 0):
            y_pred.append(1)
        elif(pred < 0):
            y_pred.append(-1)
    
    a = pd.DataFrame(x_test)
    a['y_pred'] = y_pred
    a['y_real'] = y_test
    
     #print running time
    print("--- %s seconds ---" % (time.time() - start_time))
    print(f' Коэффициенты регрессии = {w}')
    
    if graph :
        y_pred_labels =([])
        for i,val in enumerate(y_pred):
            
            if(y_test[i] == y_pred[i]):
                y_pred_labels.append(1)
            else:
                y_pred_labels.append(0)
                
        #group for plotting
        df_test = pd.DataFrame(dict(x=x_test[:,0], y=x_test[:,1], pred=y_pred_labels,label=y_test))
        grouped_test = df_test.groupby('label')
        grouped_pred = df_test.groupby('pred') 
        pred_colors = {1:'lime', 0:'red'}
        pred_names = {1:'correct',0:'incorrect'}   
        colors = {-1:(0,100/255,0,0.9), 1:(138/255,43/255,226/255,0.9)}
        names = {-1:'Group 1', 1:'Group 2'}
        #plot decision grid with prediction values
        fig = plt.figure(figsize=(15,11))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Classified by SVM", fontsize=20)
        ax.set_xlabel("Feature 1", fontsize=18)
        ax.set_ylabel("Feature 2", fontsize=18)
        ax.set_facecolor((245/255,245/255,245/255))
        
        for key, group in grouped_test:
            ax.scatter(group.x,group.y, label=names[key], color=colors[key],edgecolor=(0,0,0,0),s=350)
        for key, group in grouped_pred:
            ax.scatter(group.x,group.y,label=pred_names[key],color=(0,0,0,0),linewidth=2,edgecolor=pred_colors[key],s=350)
        ax.legend(markerscale=1,fontsize=20,fancybox=True)
        plt.show()
    
    
    return a, w