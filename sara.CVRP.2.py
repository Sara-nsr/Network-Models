import numpy as np
import matplotlib.pyplot as plt
import math

def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]] + item
            yield item

def Gamma(s):

    gsum=0
    for t in r[s]:
        gsum = gsum + q[t]
    out = math.ceil(gsum/Q)
    print ("c",(22+s),":    s= ",s,"      ","r[s]= ",r[s],"     out= ", out)
    return  out

rnd = np.random
rnd.seed(0)
# n = 10
# Q = 20

n = 10
Q = 20


N = [i for i in range(1, n + 1)]
print(N)
V = [0] + N
q = {i: rnd.randint(1, 10) for i in N}
print("q====",q)
print(q)
loc_x = rnd.rand(len(V)) * 200
loc_y = rnd.rand(len(V)) * 100

# N = [1,2,3,4,5,6,7,8,9,10,11,12]
# V = [0] + N
# q = {1:1,2:1,3:1,4:1,5:3,6:2,7:2,8:1,9:1,10:3,11:1,12:2}
# loc_x = [276,84,224,303,111,304,331,595,659,386,425,601,410]
# loc_y = [700-627,700-457,700-135,700-79,700-371,700-277,700-380,700-520,700-156,700-315,700-288,700-208,700-382]

r = [x for x in powerset(N)]
print(r)
superSet=range(1, pow(2, len(N))-1)
print("lenR=",len(r))
print("superser=",[m for m in superSet])



plt.scatter(loc_x[1:], loc_y[1:], c='b')
for i in N:
    plt.annotate('$q_%d=%d$' % (i, q[i]), (loc_x[i] + 2, loc_y[i]))
plt.plot(loc_x[0], loc_y[0], c='r', marker='s')
plt.axis('equal')
plt.show()
A = [(i, j) for i in V for j in V if i != j]
c = {(i, j): np.hypot(loc_x[i] - loc_x[j], loc_y[i] - loc_y[j]) for i, j in A}
from docplex.mp.model import Model

mdl = Model('CVRP')
x = mdl.binary_var_dict(A, name='x')
u = mdl.continuous_var_dict(N, ub=Q, name='u')
mdl.minimize(mdl.sum(c[i, j] * x[i, j] for i, j in A))
mdl.add_constraints(mdl.sum(x[i, j] for j in V if j != i) == 1 for i in N)
mdl.add_constraints(mdl.sum(x[i, j] for i in V if i != j) == 1 for j in N)
# mdl.add_indicator_constraints(mdl.indicator_constraint(x[i, j], u[i] + q[j] == u[j]) for i, j in A if i != 0 and j != 0)
# mdl.add_constraints(u[i] >= q[i] for i in N)
K=3

mdl.add_constraint(mdl.sum(x[i, 0] for i in N) == K)
mdl.add_constraint(mdl.sum(x[0, j] for j in N) == K)

mdl.add_constraints(mdl.sum(x[i, j] for i in V if i not in r[s] for j in r[s]) >= Gamma(s) for s in superSet)






mdl.parameters.timelimit = 15
solution = mdl.solve(log_output=True)
print(solution)
print(solution.solve_status)
active_arcs = [a for a in A if x[a].solution_value > 0.9]
plt.scatter(loc_x[1:], loc_y[1:], c='b')
for i in N:
    plt.annotate('$q_%d=%d$' % (i, q[i]), (loc_x[i] + 2, loc_y[i]))
for i, j in active_arcs:
    plt.plot([loc_x[i], loc_x[j]], [loc_y[i], loc_y[j]], c='g', alpha=0.3)
plt.plot(loc_x[0], loc_y[0], c='r', marker='s')
plt.axis('equal')
plt.show()