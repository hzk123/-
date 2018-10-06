import numpy as np
import math
x_train = np.load("x_train.npy");
y_train = np.load("y_train.npy");
x_test = np.load("x_test.npy");
y_test = np.load("y_test.npy");

K = 100; #  define topK in this place

def debug():
    print(np.size(x_train) , np.size(x_test));

    n = np.size(x_train);
    tot = 0 ;
    tot2 = 0;
    for i in range(0,n):
        tot += y_train[i];
        tot2 += y_test[i];
    print('train_postive , train_negetive'),
    print(tot,n-tot);
    print('test_postive , test_negetive'),
    print(tot2,n-tot2);

    return



def train():

    Acount={}
    Bcount={}
    count={}
    n = np.size(x_train)
    tot = [0,0]
    mx = 0
############################################
    for i in range(0,n):
        m = np.size(x_train[i])
        for j in range(0,m):
                c = x_train[i][j]
                if ( c in count): count[c] = count[c] + 1
                else: count[c] = 1;
    count = sorted(count.items(), key =lambda  d: d[1] , reverse=True)
    cnt = 0;
    topK={}
    for key,value in count:
        topK[key] = value;
        cnt = cnt + 1;
        if (cnt > K): break;
############################################ When I finish this I find the data is already sorted qaq
    for i in range(0,n):
        m = np.size(x_train[i]);
        label = y_train[i];
        for j in range(0,m):
            c = x_train[i][j];
            if not (c in topK): continue;
            tot[label] = tot[label] + 1;
            if ( label == 0 ):
                if (c in Acount): Acount[c] = Acount[c] + 1;
                else: Acount[c] = 1;
            if ( label == 1 ):
                if (c in Bcount): Bcount[c] = Bcount[c] + 1;
                else: Bcount[c] = 1;
            if (c > mx): mx = c;
    Acount = sorted(Acount.items(), key=lambda d: d[1], reverse=True)
    Bcount = sorted(Bcount.items(), key=lambda d: d[1], reverse=True)

############################################

    Pa = {} ;Pb = {}; cnt = 0;
    for key,value in Acount:
        Pa[key] = math.log (1.0 * (value + 1) / ( tot[0] + mx ) ) ;

    for key,value in Bcount:
        Pb[key] = math.log (1.0 * (value + 1) / ( tot[1] + mx ) ) ;
    f=[0,0]; g=[0,0];
    f[0] = math.log( 1.0 / ( tot[0] + mx) );    #default
    f[1] = math.log( 1.0 / ( tot[1] + mx) );    #default

    tot[0] = math.log( tot[0] * 1.0 / ( tot[0] + tot[1]))
    tot[1] = math.log( tot[1] * 1.0 / ( tot[0] + tot[1]))
    return [Pa,Pb,f,tot];



def predict( P ):
    Pa = P[0];
    Pb = P[1];
    f0 = P[2][0];
    f1 = P[2][1];


    TP = 0 ; FP = 0 ; FN = 0 ; TN = 0;
    n = np.size(x_test);
    L=[0] * n;
    for i in range(0,n):
        m = np.size(x_test[i])
        P1 = P[3][0];
        P2 = P[3][1];
        for j in range(0,m):
            c = x_test[i][j]
            label = y_test[i]
            if (c in Pa): P1 = P1 + Pa[c];
            else: P1 = P1 + f0;
            if (c in Pb): P2 = P2 + Pb[c];
            else: P2 = P2 + f1;
        if (P1 > P2):
            L[i] = 0;
        else:
            L[i] = 1;

        if (L[i] == 0 and label == 0): TP = TP + 1;
        if (L[i] == 0 and label == 1): FP = FP + 1;
        if (L[i] == 1 and label == 0): FN = FN + 1;
        if (L[i] == 1 and label == 1): TN = TN + 1;


    print('TP,FP,TN,FN = ',TP,FP,FN,TN);

    accuracy = (TP + TN) * 1.0 / ( TP + FP + FN + TN);
    precision = TP * 1.0 / (TP + FP);
    recall = TP * 1.0 / (TP + FN);
    F1 = 1.0 / precision + 1.0 / recall;
    F1 = 2.0 / F1;
    return [accuracy,precision,recall,F1,L];

def main():
    #debug()

    model = train();
    ret = predict(model);

    print('accuray =',ret[0]);
    print('precison =',ret[1]);
    print('recall =',ret[2]);
    print('F1-measure =',ret[3]);
    return

main()
