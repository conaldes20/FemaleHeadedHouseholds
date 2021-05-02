import numpy as np
import csv
import math
from math import floor
from random import randint
from decimal import Decimal
from datetime import datetime
from random import seed
from random import random
from numpy.linalg import pinv

class OPTTechnique:
    def __init__(self):
        self.xmax = []
        self.xmin = []
        self.ymax = 0.00
        self.ymin = 0.00

def createTrainingMatrix(self):    
        # load the dataset from the CSV file
        reader = csv.reader(open("C:/Users/CONALDES/Documents/FemaleHeadedHouseholds/train.csv", "r"), delimiter=",")
        xx = list(reader)
        xxln = len(xx)
        
        allrecs = []
        for row in range(1, xxln):
            fields = []
            recln = len(xx[row])
            if row != 1816:
                for i in range(0, recln):
                    if i != 0 and i != 16 and i != 17 and i != 42 and i != 57 and i !=58 and i != 59:
                        fields.append(xx[row][i])    
                allrecs.append(fields)
                
        x = []
        y = []
        xxln = len(allrecs)
        for row in range(1, xxln):
            temp = []
            rowln = len(allrecs[row])
            for i in range(0, rowln):
                if(i != 2):
                    temp.append(float(allrecs[row][i]))
                elif(i == 2):
                    y.append(float(allrecs[row][i]))
                    
            x.append(temp)
                       
        xln = len(x)
        x = np.array(x)
        y = np.array(y)
        
        print('@@@@@@@@@@@ Un-normalised data @@@@@@@@@@@')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('x: ' + str(x))
        print('                           ')
        print('y: ' + str(y))
        
        thh = x[:,0]
        thh = np.vstack(thh)
        tdl = x[:,1]
        tdl = np.vstack(tdl)
        
        dw2_13 = x[:,2:13]
        srow, scol = dw2_13.shape
        dw = []
        for i in range(0, srow):
            temp = []
            for j in range(0, scol):
                temp.append(dw2_13[i][j])
            dw.append(temp)
        dw = np.array(dw)
        psa14_18 = x[:,14:18]
        srow, scol = psa14_18.shape
        psa = []
        for i in range(0, srow):
            temp = []
            for j in range(0, scol):
                temp.append(psa14_18[i][j])
            psa.append(temp)
        psa = np.array(psa)
        
        stv19_20 = x[:,19:20]
        srow, scol = stv19_20.shape
        stv = []
        for i in range(0, srow):
            temp = []
            for j in range(0, scol):
                temp.append(stv19_20[i][j])
            stv.append(temp)
        stv = np.array(stv)
        
        car21_22 = x[:,21:22]
        srow, scol = car21_22.shape
        car = []
        for i in range(0, srow):
            temp = []
            for j in range(0, scol):
                temp.append(car21_22[i][j])
            car.append(temp)
        car = np.array(car)
        
        lln23_24 = x[:,23:24]
        srow, scol = lln23_24.shape
        lln = []
        for i in range(0, srow):
            temp = []
            for j in range(0, scol):
                temp.append(lln23_24[i][j])
            lln.append(temp)
        lln = np.array(lln)
        
        lan25_38 = x[:,25:38]
        srow, scol = lan25_38.shape
        lan = []
        for i in range(0, srow):
            temp = []
            for j in range(0, scol):
                temp.append(lan25_38[i][j])
            lan.append(temp)
        lan = np.array(lan)
        
        pa39_43 = x[:,39:43]
        srow, scol = pa39_43.shape
        pa = []
        for i in range(0, srow):
            temp = []
            for j in range(0, scol):
                temp.append(pa39_43[i][j])
            pa.append(temp)
        pa = np.array(pa)
        
        lgt44 = x[:,44]
        lgt = np.vstack(lgt44)
        
        pw45_51 = x[:,45:51]
        srow, scol = pw45_51.shape
        pw = []
        for i in range(0, srow):
            temp = []
            for j in range(0, scol):
                temp.append(pw45_51[i][j])
            pw.append(temp)
        pw = np.array(pw)
        
        lat52 = x[:,52]
        lat = np.vstack(lat52)
        
        lon53 = x[:,53]
        lon = np.vstack(lon53)
        
        nl54 = x[:,54]        
        nl = np.vstack(nl54)

        x = np.concatenate((thh, tdl, dw, psa, stv, car, lln, lan, pa, lgt, pw, lat, lon, nl), axis=1)
        xrow, xcol = x.shape
        
        xmax = []
        xmin = []
        for k in range(0, xcol):
            xmax.append(np.max(x[:,k]))
            xmin.append(np.min(x[:,k]))
        
        ymax = np.max(y)
        ymin = np.min(y)

        xrows, xcols = x.shape
        for c in range(xrows):
            for i in range(xcols): 
                x[c][i] = (x[c][i] - xmin[i])/(xmax[i] - xmin[i])
                
        for c in range(xln):
            y[c] = (y[c] - ymin)/(ymax - ymin)

        print('                           ')    
        print('@@@@@@@@@@@ Normalised data @@@@@@@@@@@')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('x: ' + str(x))
        print('                           ')
        print('y: ' + str(y))
       
        saveDataSetMetadata(self,xmax,xmin,ymax,ymin)
        return x, y

def createTestingMatrix(self):
        reader = csv.reader(open("C:/Users/CONALDES/Documents/FemaleHeadedHouseholds/test.csv", "r"), delimiter=",")
        xx = list(reader)
        xxln = len(xx)
        print('len(xx): ' + str(len(xx)))
        x_test = []
        wards = []
        for row in range(1, xxln):
            temp = []
            rowln = len(xx[row])
            for i in range(0, rowln):
                if((i != 0) and (i != 15) and (i != 16) and (i != 41) and (i != 56) and (i != 57) and (i != 58)):
                    temp.append(float(xx[row][i]))
                elif(i == 0):
                    wards.append(xx[row][i])
                    
            x_test.append(temp)
           
        x_test = np.array(x_test)
        wards = np.array(wards)
        #wards = np.array(wards)

        thh = x_test[:,0]
        thh = np.vstack(thh)
        tdl = x_test[:,1]
        tdl = np.vstack(tdl)
        
        dw2_13 = x_test[:,2:13]
        srow, scol = dw2_13.shape
        dw = []
        for i in range(0, srow):
            temp = []
            for j in range(0, scol):
                temp.append(dw2_13[i][j])
            dw.append(temp)
        dw = np.array(dw)
        psa14_18 = x_test[:,14:18]
        srow, scol = psa14_18.shape
        psa = []
        for i in range(0, srow):
            temp = []
            for j in range(0, scol):
                temp.append(psa14_18[i][j])
            psa.append(temp)
        psa = np.array(psa)
        
        stv19_20 = x_test[:,19:20]
        srow, scol = stv19_20.shape
        stv = []
        for i in range(0, srow):
            temp = []
            for j in range(0, scol):
                temp.append(stv19_20[i][j])
            stv.append(temp)
        stv = np.array(stv)
        
        car21_22 = x_test[:,21:22]
        srow, scol = car21_22.shape
        car = []
        for i in range(0, srow):
            temp = []
            for j in range(0, scol):
                temp.append(car21_22[i][j])
            car.append(temp)
        car = np.array(car)
        
        lln23_24 = x_test[:,23:24]
        srow, scol = lln23_24.shape
        lln = []
        for i in range(0, srow):
            temp = []
            for j in range(0, scol):
                temp.append(lln23_24[i][j])
            lln.append(temp)
        lln = np.array(lln)
        
        lan25_38 = x_test[:,25:38]
        srow, scol = lan25_38.shape
        lan = []
        for i in range(0, srow):
            temp = []
            for j in range(0, scol):
                temp.append(lan25_38[i][j])
            lan.append(temp)
        lan = np.array(lan)
        
        pa39_43 = x_test[:,39:43]
        srow, scol = pa39_43.shape
        pa = []
        for i in range(0, srow):
            temp = []
            for j in range(0, scol):
                temp.append(pa39_43[i][j])
            pa.append(temp)
        pa = np.array(pa)
        
        lgt44 = x_test[:,44]
        lgt = np.vstack(lgt44)
        
        pw45_51 = x_test[:,45:51]
        srow, scol = pw45_51.shape
        pw = []
        for i in range(0, srow):
            temp = []
            for j in range(0, scol):
                temp.append(pw45_51[i][j])
            pw.append(temp)
        pw = np.array(pw)
        
        lat52 = x_test[:,52]
        lat = np.vstack(lat52)
        
        lon53 = x_test[:,53]
        lon = np.vstack(lon53)
        
        nl54 = x_test[:,54]        
        nl = np.vstack(nl54)
        
        x_test = np.concatenate((thh, tdl, dw, psa, stv, car, lln, lan, pa, lgt, pw, lat, lon, nl), axis=1)
        
        return x_test, wards

def saveDataSetMetadata(self,xmax,xmin,ymax,ymin):        
    self.xmax = xmax
    self.xmin = xmin
    self.ymax = ymax
    self.ymin = ymin
   
def predict_targets(self, testdata, coefs):
    tdln = len(testdata)    
    xtest = []
    
    for i in range(0, tdln):        
        temp = (testdata[i] - self.xmin[i])/(self.xmax[i] - self.xmin[i])
        xtest.append(temp)
                
    predy = 0.00
    cfln = len(coefs)
    #print("len(xtest) = " + str(len(xtest)))
    #print("len(coefs) = " + str(len(coefs)))
    predy = predy + coefs[0]
    for i in range(0, (cfln - 1)):
        predy = predy + xtest[i]*coefs[i + 1]    # xtest[i] i = 0 to 46 and coefs[i] i = 1 to 47
    
    predy = (predy*(self.ymax - self.ymin)) + self.ymin
    return predy

def roundup(a, digits=0):
    #n = 10**-digits
    #return round(math.ceil(a / n) * n, digits)
    return round(a, digits)

def recsToSelect(numrecs):
    notoselect = 0
    #numrecs = 2822
    #sqrval = Decimal(numrecs).sqrt()
    numgroups = float(numrecs)/5.0
    flval = floor(numgroups)
    #print("Square root of " + str(numrecs) + " = " + str(flval))
    totalnum = numgroups*5   
    if(numrecs > (numgroups*5)):
        notoselect = flval + 1
    elif(numrecs == (numgroups*5)):
        notoselect = flval
        
    temprandomset = []
    minim = 1
    maxim = minim + 5 - 1                           
    while(maxim <= totalnum):
        tempval = randint(minim,maxim)	
        temprandomset.append(tempval)
        minim = maxim + 1
        maxim = minim + 5 - 1
        
    if(notoselect > flval):
        minim = totalnum + 1	# 48 - 59 => 11
        maxim = minim + ((numrecs - totalnum) - 1)
        tempval = randint(minim,maxim)
        temprandomset.append(tempval)
         
    temprandomset.sort()
    return temprandomset

def simulateModel(self,x,y):
    # current date and time
    print('                           ')    
    print('@@@@@@@@@@@ Model Simulation @@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    now0 = datetime.now()
    timestamp0 = datetime.timestamp(now0)
    brow, bcol = x.shape
    m = np.ones((brow,1))
    #x = np.array(x_train)            # x
    #print('x: ' + str(x))
    #print('x.shape: ' + str(x.shape))
    x = np.concatenate((m,x),axis=1)
    #c, residuals, rank, s = lstsq(x, y)
    c = pinv(x).dot(y)
    print('Coefficients: ' + str(c))
    #print('len(c): ' + str(len(c)))
    print(c)
    # predict using coefficients
    x_data, wards = createTestingMatrix(self)
    #print("x_data.shape: " + str(x_data.shape))
    #xdtrow, xdtcol = x_data.shape
    #print("xdtrow, xdtcol: " + str(xdtrow) + ", " + str(xdtcol))
    xdtln = len(x_data)
    print('len(x_data): ' + str(len(x_data)))
    print('x_data: ' + str(x_data))
    predicted_targets = np.zeros((xdtln, 1))
    #for i in range(0, xdtln):
    #    print(str(wards[i]))  
    for i in range(0, xdtln):
        pred_targets = predict_targets(optt, x_data[i], c)
        if pred_targets < 0:
            pred_targets = 0.00

        #print(str(pred_targets))    
        predicted_targets[i][0] = pred_targets
    now1 = datetime.now()
    timestamp1 = datetime.timestamp(now1)
    time_elapsed = timestamp1 - timestamp0
    print('Time elapsed for computations: ' + str(time_elapsed) + 'seconds')
    return predicted_targets, wards

optt = OPTTechnique()

dataTrainX, dataTrainY = createTrainingMatrix(optt)
predicted_targets, wardids = simulateModel(optt,dataTrainX,dataTrainY)
wardids = np.vstack(wardids)
#print("wardids.shape: " + str(wardids.shape))
#print("predicted_targets.shape: " + str(predicted_targets.shape))
wardids_targets_arr = np.concatenate((wardids, predicted_targets), axis=1)
#print("wardids_targets_arr: " + str(wardids_targets_arr.shape))
with open("C:/Users/CONALDES/Documents/FemaleHeadedHouseholds/ConaldesSubmission.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ward", "target"])

    for row in wardids_targets_arr:    
        l = list(row)    
        writer.writerow(l)

print("                                          ")
print("### C:/Users/CONALDES/Documents/FemaleHeadedHouseholds/ConaldesSubmission.csv contains results ###")    
