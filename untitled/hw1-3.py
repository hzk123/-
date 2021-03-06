
import random
import matplotlib.pyplot as plot
import numpy as np
import  math
c = [ [0.0]* 51 for i in range(51)]



def qp( p , b):
    ret = 1.0;
    while ( b > 0):
        if ( b & 1 != 0 ): ret = ret * p;
        b = b >> 1;
        p = p * p;

    return ret;

def init():
    c[0][0] = 1;
    for i in range(1,51):
        c[i][0] = 1;
        for j in range(1,51):
            c[i][j] = c[i-1][j-1] + c[i-1][j];

def getF(p):
    f = [0.0] * 11;
    for i in range(11):
        f[i] = c[10][i] * qp(p,i) * qp(1-p,11-i);
    return f;


def plotbar(x , y , name):
    fig = plot.figure();
    plot.bar(x,y,0.4,color='green');
    plot.xlabel("X");
    plot.ylabel("Y");
    plot.title(name);
    plot.savefig(name);
    plot.show()
    return ;

def solve():
    f=[0.0]*11

    mx = 0 ; mx_p = 0;
    for i in range(11):
        p = (i * 0.1);
        f[i] = c[10][2] * qp(p,2) * qp(1-p,8);

        if f[i] > mx :
            mx = f[i]
            mx_p = p;

    plotbar(range(11),f,"MLE-likelyhood")

    print(mx_p);
    print("MLE-likelyhood",f);
    return



def solve_2():
    likelyhood = np.zeros((11));
    #prior = [0.1] * 11;
    prior = [0.01, 0.01, 0.05, 0.08, 0.15, 0.4, 0.15, 0.08, 0.05, 0.01, 0.01];
    posterior=[0.0] * 11;
    sum = 0.0;
    for i in range(11):
        p = i * 0.1;
        likelyhood[i] = c[10][2] * qp(p,2) * qp(1-p,8);
        sum = sum + likelyhood[i] * prior[i];

    for i in range(11):
        posterior[i] = likelyhood[i] * prior[i] / sum;

    print(posterior);
    plotbar(range(11),prior,"MAP-prior");
    plotbar(range(11),likelyhood,"MAP-likelyhood");
    plotbar(range(11),posterior,"MAP-posterior");
    return;

def implementation():
    likelyhood = [0.0] * 11;
    #prior = [0.1] * 11;
    prior = [0.01, 0.01, 0.05, 0.08, 0.15, 0.4, 0.15, 0.08, 0.05, 0.01, 0.01];

    posterior = [0.0] * 11;
    sum = 0.0;
    f = [0] * 2;
    g = [];
    for _ in range(1,51):
        r = 1 if random.uniform(0,1000) > 500 else 0;
        f[r] = f[r] + 1;
        if ( _ % 10 == 0 ):
            sum = 0;
            print(f[0], f[1]);
            for i in range(11):
                p = i * 0.1;
                likelyhood[i] = c[f[0]+f[1]][f[0]] * qp(p, f[0] ) * qp(1-p, f[1]);
                sum = sum + likelyhood[i] * prior[i];
            for i in range(11):
                posterior[i] = likelyhood[i] * prior[i] / sum;
            posterior = prior;
            if ( _ % 10 == 0):
                #plotbar(range(11),posterior, 'posterior of '+ str(_));
                entropy = 0;
                for j in range(11):
                    if (posterior[j] > 0):
                        entropy = entropy - posterior[j] * math.log(posterior[j])
                g.append(entropy);
    print(g);
    plot.plot(range(len(g)),g);
    plot.savefig("line");
    plot.show();
    return ;


def main():
    init();
    #solve();
    #solve_2();
    implementation();
    return;

main();
