'''
regMVMT

var_name -> notation : dimension : use/discription
or
var_name -> use/discription

Define:
T -> # of Tasks
V -> # of Views
t -> task index
v -> view index
N -> # of labeled samples
M -> # of unlabeled samples
D -> # of features

Input:
y -> y_t : [T] : y[t] -> [N_t x 1] label column vector
X -> X_t : [T] : X[t] -> [N_v x D_v] labeled feature matrix
U -> U_t : [T] : U[t] -> [M_v x D_v] unlabeled feature matrix
lambda ->  
mu -> 
iterations -> N_it
epsilson -> 

Algorithm:
W0 -> W_0 :
Omega0 -> Omega_v0 : [V] : Omega0[v] -> 
I -> I_d : [T x V] : I[t, v] -> 1 for labeled samples and 0 for unlabeled samples
A -> A_tv -> [T x V] : A[t, v] -> [D_v x D_v] matrix
B -> B_vv'_t -> [T x V] : B[t, v, v'] -> [D_v x D_v] matrix
C -> C_tv : [T x V] : C[t', v] = [D_v x D_v] matrix
E -> E_t_v : [T x V] : E[t, v] -> [D_v x 1] column matrix
L -> L : [TD x TD] : L[i, j] -> value
R -> R : [TD x 1] column matrix
w -> w_t_v : [T x V] : w[t, v] -> [D_v x 1] column vector of weights
w_t -> w_t : [D x 1] : [D x 1] column vector of weights

Output:
W -> W_t : [D x T] : weights matrix
Omega -> Omega_v : [V] : Omega[v] -> [T x T] similarity matrix

'''

import numpy as np

class Reg_MVMT(object):
    def __init__(self, task_views, tasks, views):
        self.task_views = task_views # {task_key : view_keys}
        self.tasks = tasks # {task_key : example_values}
        self.views = views # {view_key : feature_matrix}
        
    def run_mvmt(self, iterations, lambda_var, mu, gamma):
        T = len(self.tasks.keys())
        V = len(self.views.keys())
        D = sum([x.shape[1] for x in self.views.values()])
        y = {}
        X = {}
        U = {}
        I = np.matrix(np.ones((T, V)))
        L = np.matrix((T * V, T * V))
        W = np.matrix(np.zeros((D, T)))
        Omega = {}

        # build y, X, U, and I
        for (t, v) in [(x, self.views.keys()) for x in self.tasks.keys()]:
            if v in self.task_views[t]:
                I[t, v] = 1
                X[t, v] = self.views[v]
            else:
                I[t, v] = 0
                U[t, v] = self.views[v]
            y[t] = np.matrix(self.tasks[t]).T

        # initialize W0
        W0 = np.matrix(np.zeros((D, T)))

        # initialize Omega0
        Omega0 = {}
        for v in range(V):
            I_T = np.matrix(np.identity(T))
            Omega0[v] = (1 / T) * I_T

        for iteration in range(iterations):
            A = {}
            B = {}
            C = {}
            E = {}

            for (t, v) in range(T), range(V):
                # construct A[t, v]
                A[t, v] = lambda_var + (mu * (V - 1) * U[t, v].T * U[t, v]) + \
                          ((X[t, v].T * X[t, v]) / (V ** 2))

                # construct E[t, v]
                E[t, v] = (X[t, v].T * y[t]) / V

                # construct B[t, v, v']
                for v` in range(V):
                    if v != v`:
                        B[t, v, v`] = ((X[t, v].T * X[t, v`]) / (V ** 2)) - \
                                      (mu * U[t, v].T * U[t, v]))

                # construct C[t', v]
                for t` in range(T):
                    if t != t`:
                        I_Dv = np.matrix(np.identity(self.views[v].shape[1]))
                        C[t`, v] = gamma * Omega[v][t, t`] * I_Dv

            # construct L
            L = np.zeros((task_count * view_count, task_count * view_count))
            i_offset = 0
            j_offset = 0
            for t in range(T):
                for t` in range(T):
                    if t == t`:
                        for v in range(V):
                            for v` in range(V):
                                if v == v`:
                                    for i in range(A[t, v].shape[0]):
                                        for j in range(A[t, v].shape[1]):
                                            L[i + i_offset, j + j_offset] = A[t, v][i, j]
                                    j_offset += A[t, v].shape[1]
                                else:
                                    for i in range(B[t, v, v`].shape[0]):
                                        for j in range(B[t, v, v`].shape[1]):
                                            L[i + i_offset, j + j_offset] = B[t, v, v`][i, j]
                                    j_offset += B[t, v, v`].shape[1]
                    else:
                        for v in range(V):
                            for v` in range(V):
                                if v == v`:
                                    for i in range(C[t, v].shape[0]):
                                        for j in range(C[t, v].shape[1]):
                                            L[i + i_offset, j + j_offset] = C[t, v][i, j]
                                    j_offset += C[t, v].shape[1]
                                else:
                                    for i in range(C[t, v].shape[0]):
                                        for j in range(C[t, v].shape[1]):
                                            L[i + i_offset, j + j_offset] = 0
                                    j_offset += C[t, v].shape[1]
                i_offset += B[t, 1, 2].shape[0]

            # construct R -> column vector
            R = []
            for t in range(T):
                for v in range(V):
                    R.extend(E[(t, v)].T.tolist()[0])
            R = np.matrix(R).T

            # compute W
            W = L.I * R

            # construct W_v
            W_v = []
            for t in range(T):
                for v in range(V):
                    W_v.extend(W[(t, v)].T.tolist()[0])
            W_v = np.matrix(W_v).T

            # update Omega[v]
            for v in range(V):
                Omega[v] = ((W_v.T * W_v) ** (1 / 2)) / (sum(np.diag((W_v.T * W_v) ** (1 / 2)).tolist()))

            if (() < epsilson) && (() < epsilson):
                break
            else:
                W0 = W
                for v in range(V):
                    Omega0[v] = Omega[v]

        return (W, Omega)


