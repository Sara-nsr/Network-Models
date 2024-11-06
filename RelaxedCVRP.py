import numpy as np
import matplotlib.pyplot as plt
import math
import random
from docplex.mp.model import Model


def Gamma(ali):
    gsum=0
    for t in ali:
        gsum = gsum + q[t]
    out = math.ceil(gsum/Q)
    return  out


def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

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

def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )


rnd = np.random
rnd.seed(0)
n = 13
K = 4

Q = 10
N = [1,2,3,4,5,6,7,8,9,10,11,12]
V = [0] + N
q = {1:1,2:1,3:1,4:1,5:3,6:2,7:2,8:1,9:1,10:3,11:1,12:2}
q.update({0:0})
loc_x = [276,84,224,303,111,304,331,595,659,386,425,601,410]
loc_y = [700-627,700-457,700-135,700-79,700-371,700-277,700-380,700-520,700-156,700-315,700-288,700-208,700-382]

# N = [i for i in range(1, n + 1)]
# V = [0] + N
# q = {i: rnd.randint(1, 10) for i in N}
# loc_x = rnd.rand(len(V)) * 200
# loc_y = rnd.rand(len(V)) * 100


A = {(i, j) for i in V for j in V if i != j}
c = {(i, j): np.hypot(loc_x[i] - loc_x[j], loc_y[i] - loc_y[j]) for i, j in A}

m=n
S=partition(V, m)
try:
    S.remove([])
except:
    print("no null in intial S")
counter=0
break_flag=0
LB=0
while(1):
    for pp in range(0, 100):
        try:
            S[pp].index(0)
            break
        except:
            continue
    S[0], S[pp] = S[pp], S[0]
    print("S", "_", counter, "= ", S)
    counter=counter+1
    ########################################################################################################################
    #init plot:
    ########################################################################################################################

    # plt.scatter(loc_x[1:], loc_y[1:], c='b')
    # for i in N:
    #     plt.annotate('$q_%d=%d$' % (i, q[i]), (loc_x[i] + 2, loc_y[i]))
    # plt.plot(loc_x[0], loc_y[0], c='r', marker='s')
    # plt.axis('equal')
    # plt.show()


    ########################################################################################################################
    #A2 calc
    ########################################################################################################################
    A3=set()
    for P in S:
         A1={(i, j) for i in P for j in P if i != j}
         A3=A3.union(A1)
    A2=A-A3
    ########################################################################################################################
    #Model init
    ########################################################################################################################

    mdl = Model('CVRP')
    x = mdl.binary_var_dict(A, name='x')
    mdl.minimize(mdl.sum(c[i, j] * x[i, j] for i, j in A2))

    mdl.add_constraints(mdl.sum(x[i, j] for (i,j) in A2 if j==nn) <= 1 for nn in N)
    mdl.add_constraints(mdl.sum(x[i, j] for (i,j) in A2 if i==nn) <= 1 for nn in N)
    mdl.add_constraint(mdl.sum(x[i, j] for (i,j) in A2 if j==0) <= K)
    mdl.add_constraint(mdl.sum(x[i, j] for (i,j) in A2 if i==0) <= K)

    mdl.add_constraints(mdl.sum(x[i, j] for j in S[z] for i in V if i not in S[z]) >= Gamma([w for w in S[z]]) for z in range(1,m))
    mdl.add_constraint(mdl.sum(x[i, j] for j in S[0] for i in V if i not in S[0]) >= Gamma([w for w in V if w not in S[0]]))
    mdl.add_constraints(mdl.sum(x[i, j] for j in S[z] for i in V if i not in S[z]) == mdl.sum(x[i, j] for i in S[z] for j in V if j not in S[z]) for z in range(1, m))
    mdl.add_constraint(mdl.sum(x[i, j] for j in S[0] for i in V if i not in S[0]) == mdl.sum(x[i, j] for i in S[0] for j in V if j not in S[0]))

    ########################################################################################################################
    #Model Run
    ########################################################################################################################

    mdl.parameters.timelimit = 15
    mdl.log_output=0
    solution = mdl.solve(log_output=False)
    # print(solution)
    print(solution.solve_status)

    ########################################################################################################################
    #Final plots:
    ########################################################################################################################
    active_arcs = [a for a in A if x[a].solution_value > 0.9]
    plt.scatter(loc_x[1:], loc_y[1:], c='b')
    for i in N:
        plt.annotate('$q_%d=%d$' % (i, q[i]), (loc_x[i] + 2, loc_y[i]))
    # for i in V:
    #     for j in range(i+1,n+1):
    #         t = np.linspace(loc_x[i], loc_x[j], 100)
    #         y = np.linspace(loc_y[i], loc_y[j], 100)
    #         line = plt.plot(t, y, c='k', alpha=0.03)[0]
    #         plt.annotate('$c_%d_%d=%d$' % (i, j,c[(i,j)]), ((loc_x[i]+loc_x[j])/2 + 2, (loc_y[i]+loc_y[j])/2))
    for i, j in active_arcs:
        t = np.linspace(loc_x[i], loc_x[j], 100)
        y = np.linspace(loc_y[i], loc_y[j], 100)
        line = plt.plot(t, y, c='g', alpha=0.3)[0]
        add_arrow(line)
        # plt.arrow(loc_x[i], loc_y[i], loc_x[j]-loc_x[i], loc_y[j]-loc_y[i], color='g', alpha=0.3, head_width=3, head_length=3)
    plt.plot(loc_x[0], loc_y[0], c='r', marker='s')
    plt.axis('equal')
    ########################################################################################################################
    # C update:
    ########################################################################################################################
    for yy in A3:
        print("c[",yy,"]= ",c[yy])
        c[yy]=0
        print("c[",yy,"]= ",c[yy])

    #########################################################################################################################
    # LB update:
    LB=LB+solution.objective_value
    print("*******************LB:",LB)

    ########################################################################################################################
    # plt.show()
    ########################################################################################################################
    # S update:
    ########################################################################################################################
    update_flag=0
    SSS=powerset(S)
    allsubs = [a for a in SSS]
    allsubs.sort()
    forCounter=-1
    v_0=[]
    v_1=[]
    temp = []
    for unflattedSub in allsubs:
        forCounter=forCounter+1
        flattedSub = [j for x in unflattedSub for j in x]
        try:
            flattedSub.index(0)
            if sum([x[i, j].solution_value for j in flattedSub for i in V if i not in flattedSub]) < Gamma([w for w in V if w not in flattedSub]) or sum([x[i, j].solution_value for j in flattedSub for i in V if i not in flattedSub])!=sum([x[i, j].solution_value for i in flattedSub for j in V if j not in flattedSub]):
                v_0.append(unflattedSub)
        except:
            if sum(x[i, j].solution_value for j in flattedSub for i in V if i not in flattedSub) < Gamma([w for w in flattedSub]) or sum(x[i, j].solution_value for j in flattedSub for i in V if i not in flattedSub)!=sum(x[i, j].solution_value for i in flattedSub for j in V if j not in flattedSub):
                v_1.append(unflattedSub)

    v_1.sort(key=len)
    v_0.sort(key=len)
    f_v_1 = [j for x in v_1[0] for j in x]
    f_v_0 = [j for x in v_0[0] for j in x]

    if len(v_1)==0 and len(v_0)!=0:
        temp=v_0[0]
    elif  len(v_1)!=0 and len(v_0)==0:
        temp = v_1[0]
    elif len(v_1)==0 and len(v_0)==0:
        break_flag=1
    elif len(f_v_1)<=len(f_v_0):
        temp=v_1[0]
    else:
        temp=v_0[0]
    if temp == []:
        # print("v_1:",v_1)
        # print("v_0",v_0)
        # print("finalS:",S)
        print("DDOOOOOONNNEE")
        plt.show()
        break
    flattedtemp = [j for x in temp for j in x]
    for i in temp:
        S.remove(i)
    S.append(flattedtemp)
    m = len(S)

    # print("super=",allsubs)
    # print("merged",unflattedSub)
    # print("merge_index",forCounter)
    # print("flag",update_flag)
    # print("v_1=",v_1)
    # print("v_0=",v_0)
    if break_flag==1:
        plt.show()
        print("no violation")
        break
    if m==1:
        plt.show()
        break
    plt.show()