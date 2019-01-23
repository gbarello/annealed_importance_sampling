import numpy as np

def hastings_step(f,x,step,p):

    xp = x + step
    
    start = -f(x)
    end = -f(xp)

    alpha = end-start

    out = np.copy(x)

    i = (p < alpha)

    out[i] = xp[i]

    frac = np.mean(i)

    return [out,f(out),frac]

def ham_hastings_step(f,g,x,mom,r,eps,L):

    Q = np.copy(x)
    P = np.copy(mom)

    P = P - eps * g(Q)/2
    for i in range(L):

        Q = Q + eps*P
        if i != L-1:
            P = P - eps*g(Q)
    P = P - eps*g(Q)/2

    P = -P

    fU = f(Q)
    fK = (P**2).sum(axis = 1)/2
    iU = f(x)
    iK = (mom**2).sum(axis = 1)/2

    out = np.copy(x)

    i = (r < (fU - iU +fK - iK))

    frac = np.mean(i)
    
    out[i] = Q[i]

    return out,f(out),frac

def hastings(f,init,nstep,eps = .1,grad = -1,L = 10):

    #HASTINGS TAKES NEG LG LIKELIHOOD
    
    u = np.log(np.random.uniform(0,1,[nstep,init.shape[0]]))
    s = np.random.randn(nstep,init.shape[0],init.shape[1])

    xt = np.copy(init)
    ft = f(xt)
    fro = []
    
    for k in range(nstep):
        if grad == -1:
            xt,ft,fr = hastings_step(f,xt,s[k]*eps,u[k])
        else:
            xt,ft,fr = ham_hastings_step(f,grad,xt,s[k],u[k],eps,L)
        fro.append(fr)
    return xt,ft,np.mean(fro)

def AIS(f1,f2,f1sam,shape,n_samp,n_AIS_step,nhstep,eps = .1,grad = -1,L = 10,PRINT = False):
    #THESE ARE NEG LOG LIKELIHOODS
    beta = np.linspace(0,1,n_AIS_step + 1)
    
    X = f1sam([n_samp,shape])
    F = []
    fro = []
    for k in range(1,len(beta)):
        if PRINT:
            print(k)

        fa = lambda y:(1.-beta[k-1])*f1(y) + beta[k-1]*f2(y)
        fb = lambda y:(1.-beta[k])*f1(y) + beta[k]*f2(y)

        if grad != -1:
            g = lambda y:(1.-beta[k])*grad[0](y) + beta[k]*grad[1](y)
        else:
            g = -1
                        
        F.append([-fa(X),-fb(X)])

        X,f,fr = hastings(fb,X,nhstep,eps,g,L)

        if PRINT:
            G = np.array(F)
            print((G[:,1] - G[:,0]).sum(axis = 0).mean())
        fro.append(fr)
        
    #F.append([-fa(X),-fb(X)])

    F = np.array(F)
    lW = (F[:,1] - F[:,0]).sum(axis = 0)
    
    return X,lW,fro

if __name__ == "__main__":

    def prior(x,N):
        return - np.abs(x).sum(axis = 1) - N * np.log(2)

    def poste(x,N):
        #UNNORMALIZED PSOTERIOR!
        return - ((x)*(x)/(.25**2)).sum(axis = 1)/2

    def true_norm(x,N):
        #UNNORMALIZED PSOTERIOR!
        return - (N/2)*np.log(2*np.pi*.25*.25)

    def g_prior(x,N):
        return - np.sign(x)

    def g_poste(x,N):
        return  - (x)/(.25**2)

    N = 200
    nsamp = 50
    n_AIS_step = 10000
    nhstep = 1
    eps = .05
    L = 10

    fa = lambda x: -prior(x,N)
    fb = lambda x: -poste(x,N)# + prior(x,N)

    ga = lambda x: -g_prior(x,N)
    gb = lambda x: -g_poste(x,N)# + prior(x,N)

    norm = lambda x: -true_norm(x,N)# + prior(x,N)

    def prior_samp(n):
        return np.random.laplace(0,1,[n[0],n[1]])

    #f1,f2,f1sam,shape,n_samp,n_AIS_step,nhstep,eps = .1,grad = -1,L = 10)
    import time
    t1 = time.time()
    XO,W,fro = AIS(fa,fb,prior_samp,N,n_samp = nsamp,n_AIS_step = n_AIS_step,nhstep = nhstep,eps = eps,grad = [ga,gb],L = L,PRINT = False)
    t2 = time.time()
    print("time: {}".format(t2 - t1))
    print("AIS norm",np.log(np.mean(np.exp(W))))
    print("AIS norm",np.mean(W))
    print("true norm",norm(1))
    print(np.mean(fro))
