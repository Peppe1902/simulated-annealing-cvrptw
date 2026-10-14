#import gurobipy as gp
#from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import random
import time
import itertools


'''
Function that performs a simple read of a Solomon instance file
'''
def leggi_file_solomon(filename):

    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_veicoli = 0
    capacita = 0
    deposito = None
    clienti = []
    
    # search VEHICLE section
    for i, line in enumerate(lines):
        if 'VEHICLE' in line:
            vehicle_line = lines[i+2].strip().split()
            num_veicoli = int(vehicle_line[0])
            capacita = int(vehicle_line[1])
            break
    
    #search CUSTOMER section
    inizio_clienti = False
    for line in lines:
        if 'CUST NO.' in line:
            inizio_clienti = True
            continue
        
        if inizio_clienti and line.strip():
            parti = line.strip().split()
            if len(parti) >= 7:
                cliente = {
                    'id': int(parti[0]),
                    'x': float(parti[1]),
                    'y': float(parti[2]),
                    'demand': int(parti[3]),
                    'ready_time': int(parti[4]),
                    'due_date': int(parti[5]),
                    'service_time': int(parti[6])
                }
                
                if cliente['id'] == 0:
                    deposito = cliente
                else:
                    clienti.append(cliente)
    
    return num_veicoli, capacita, deposito, clienti


'''
Function that computes the distance matrix between all the couples (i, j)
'''
def distances_matrix(deposito,clienti) :

    # clienti.insert(0, deposito)  # 0 is the depot index
    n = len(clienti)
    print(n)

    D = np.zeros((n,n))

    # distance between the node and itself
    for i in range(0,n) :
        D[i][i] = 0

    # distance between the depot and all the nodes
    for i in range(0,n) :
        D[0][i] = np.sqrt((deposito['x']-clienti[i]['x'])**2 + (deposito['y']-clienti[i]['y'])**2)
        D[i][0] = D[0][i]
    
    
    for i in range(1,n) :
        for j in range(i+1,n) :
            D[i][j] = np.sqrt((clienti[i]['x']-clienti[j]['x'])**2 + (clienti[i]['y']-clienti[j]['y'])**2)
            D[j][i] = D[i][j]
    
    return D


'''
Function that shows the routes on the display
'''
def plot_rotte(rotte, clienti, deposito=0):

    
    plt.rcParams.update({       
    'font.family': 'serif',         
    'font.size': 10,                
    'savefig.dpi': 300,            
    'lines.linewidth': 2,         
    })
    

    plt.figure(figsize=(8, 6))
    
    # Draw deposit
    depot_x = clienti[deposito]['x']
    depot_y = clienti[deposito]['y']
    plt.scatter(depot_x, depot_y, c='red', s=150, marker='s', label='Deposito') 

    # Draw client
    for i, c in enumerate(clienti):
        if i != deposito:
            plt.scatter(c['x'], c['y'], c='black', s=40)
            plt.text(c['x'] + 0.8, c['y'] + 0.8, str(i), fontsize=8)
         
    
    # Draw routes
    for k, route in enumerate(rotte):
        color = (random.random(), random.random(), random.random())
        x = [clienti[i]['x'] for i in route]
        y = [clienti[i]['y'] for i in route]
        plt.plot(x, y, marker='o', color=color, linewidth=2, label=f'Veicolo {k+1}')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Rotte trovate (VRPTW)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return


'''
Function that compute the initial solution by adopting TimeOrientedNeirestNeighbor Approach
'''
def TimeOrientedNN(clienti,D,capacita,num_veicoli,delta1=0.4,delta2=0.3,delta3=0.3) :

    N = len(clienti)
    deposito = 0

    demands = [0] # the demand of the first route is 0 
    rotte = [[deposito]] # the route initially contains only the depot
    a = np.zeros(N) # a_i = service start time of node i
    s = [clienti[i]['service_time'] for i in range(N)] # service times of the clients
    
    a[0] = clienti[deposito]['ready_time'] # base case
    k = 0

    nodo_curr = None
    nodo_prec = deposito
    U = clienti[1:N] # unrouted clients vector (initially it contains all the clients except the depot)

    while True :

        nodo_curr = None
        min_c_ij = np.inf
        start_service_time = None

        for client in U :
            j = client['id'] #current client ID

            a_j = max(clienti[j]['ready_time'],a[nodo_prec]+s[nodo_prec]+D[nodo_prec][j])
            T_ij = a_j  - (a[nodo_prec] + s[nodo_prec])
            v_ij = clienti[j]['due_date'] - (a[nodo_prec]+s[nodo_prec]+D[nodo_prec][j])
            c_ij = delta1 * D[nodo_prec][j] + delta2 * T_ij + delta3 * v_ij

            if (a_j <= clienti[j]['due_date'] and ((demands[k] + clienti[j]['demand']) <= capacita)) :

                if c_ij < min_c_ij :
                    min_c_ij = c_ij
                    start_service_time = a_j
                    nodo_curr = j
 
        if nodo_curr == None :
            rotte[k].append(deposito) # it closes the current route 
            k = k + 1

            if k < num_veicoli and len(U) > 0 :
                rotte.append([deposito])  # open a new route
                demands.append(0)
                nodo_prec = deposito # starting again from the depot
            else :
                break
        else :

            rotte[k].append(nodo_curr)
            a[nodo_curr] = start_service_time
            demands[k] = demands[k] + clienti[nodo_curr]['demand']
            nodo_prec = nodo_curr

            U = [c for c in U if c['id'] != nodo_curr] #it removes nodo_curr from unrouted clients
            
    
    ultima_rotta = rotte[-1] #-1 represents the last element of the vector
    if len(rotte) > 0 and ultima_rotta[-1] != deposito : #if the last route has not been closed (the last element is not the depot) 
        
        rotte[-1].append(deposito)

    return rotte,U,a


'''
Function that computes the objective function F(X) (total distance travelled)
'''
def objective_function(rotte,D) :
    f = 0
    for route in rotte :
        for i in range(len(route)-1) :
            f = f + D[route[i]][route[i+1]]
    return f


'''
Function that computes the best parameter values for the TimeOrientedNN
'''
def get_best_params(clienti, D, capacita, num_veicoli, passo=0.1):

    migliori_parametri = None
    miglior_punteggio = float('inf')
    
    print(f"Inizio calibrazione parametri con passo={passo}...")

    # Iterate on delta1 from 0 to 1
    for d1 in np.arange(0, 1.01, passo):
      # iterate on delta2 from 0 to the remaining amount up to 1
        for d2 in np.arange(0, 1.01 - d1, passo):

            d3 = abs(1.0 - d1 - d2)
            
            # round to avoid floating point inaccuracies
            params = (round(d1, 2), round(d2, 2), round(d3, 2))
            
            # execute the heuristic with the current parameter values
            rotte, clienti_non_serviti, a = TimeOrientedNN(clienti, D, capacita, num_veicoli, params[0], params[1], params[2])
            
            # the solution is considered valid only if all clients have been served
            if len(clienti_non_serviti) == 0:
                punteggio_attuale = objective_function(rotte, D)
                
                if punteggio_attuale < miglior_punteggio:
                    miglior_punteggio = punteggio_attuale
                    migliori_parametri = params
                    print(f"  -> Nuovo miglior punteggio: {miglior_punteggio:.2f} con parametri {migliori_parametri}")

    if migliori_parametri is None:
        print("ATTENZIONE: Nessuna combinazione di parametri ha prodotto una soluzione valida (tutti i clienti serviti).")
        return (0.0, 0.0, 0.0), float('inf') #return the default values
        
    print("Calibrazione completata.")
    return migliori_parametri, miglior_punteggio


'''
Cross Exchange move
'''
def cross_exchange_move(clienti,rotte,D,a,capacita) :

    num_rotte = len(rotte)

    if num_rotte < 2 : #A swap can't be performed with only one route
        return rotte,a,False

    # randomly generates the two route indices
    r1 = random.randint(0,num_rotte-1)
    r2 = random.choice([i for i in range(0,num_rotte) if i != r1])

    rotte_new = [r[:] for r in rotte]
    a_new = [a[i] for i in range(len(a))]
    
    if len(rotte_new[r1]) <= 2 and len(rotte_new[r2]) <= 2 :
        return rotte,a,False

    # randomly generates the two client indices of the two routes affected by the swap move
    if len(rotte_new[r1]) > 3 :
        p = random.randint(1,len(rotte_new[r1])-2)
    else :
        p = 1

    if len(rotte_new[r2]) > 3  :
        q = random.randint(1,len(rotte_new[r2])-2)
    else :
        q = 1

    segment_r1 = rotte[r2][q+1:]
    segment_r2 = rotte[r1][p+1:]
    rotte_new[r1] = rotte[r1][0:p+1] + segment_r1
    rotte_new[r2] = rotte[r2][0:q+1] + segment_r2

    #1) check the vehicle capacity for both routes
    domanda_totale_r1 = sum([clienti[rotte_new[r1][i]]['demand'] for i in range(0,len(rotte_new[r1]))])
    
    if domanda_totale_r1 > capacita :
        return rotte,a,False
    
    domanda_totale_r2 = sum([clienti[rotte_new[r2][i]]['demand'] for i in range(0,len(rotte_new[r2]))])

    if domanda_totale_r2 > capacita :
        return rotte,a,False

    #2) check time constraints on route r1
    a_new,fattibile = check_finestre_temporali(clienti,rotte_new,r1,p+1,a_new,D)

    if fattibile == False :
        return rotte,a,False

    #3) check time constraints on route r2
    a_new,fattibile = check_finestre_temporali(clienti,rotte_new,r2,q+1,a_new,D)

    if fattibile == False :
        return rotte,a,False
    
    f_new = objective_function(rotte_new,D)
    f_curr = objective_function(rotte,D)

    # Objective function improved
    if f_new < f_curr :
            return rotte_new,a_new,True
    
    return rotte,a,False



'''
Function that checks in a route route_idx, starting from a position pos, if all time-window constraints are met
'''
def check_finestre_temporali(clienti,rotte,route_idx,pos,a,D) :

    a_new = a.copy()
    a_prec = a_new[rotte[route_idx][pos-1]]

    for i in range(pos,len(rotte[route_idx])) :
        prec_id = rotte[route_idx][i-1]
        curr_id = rotte[route_idx][i]
        a_new[curr_id] = max(a_prec + clienti[prec_id]['service_time'] + D[prec_id][curr_id], clienti[curr_id]['ready_time'])
        if a_new[curr_id] > clienti[curr_id]['due_date'] :
            return a,False
        a_prec = a_new[curr_id]

    return a_new,True
    

'''
Best insertion move
'''
def best_insertion_move(clienti, rotte, D, a, capacita):

    num_rotte = len(rotte)

    if num_rotte < 2 :
        return rotte,a,False

    a_base = a.copy()
    rotte_new = [r[:] for r in rotte]

    r1 = random.randint(0,num_rotte-1) 
    r2 = random.choice([i for i in range(len(rotte_new)) if i!=r1])


    if len(rotte_new[r1]) <= 2 :
        return rotte,a,False

    k = 1
    if len(rotte_new[r1]) > 3 :    
        k = random.randint(1,len(rotte_new[r1])-2)

    new_client = rotte_new[r1][k]
    rotte_new[r1].remove(new_client) # remove the client with ID "new_client" from the first route

    # after the client removal, check the time constraints
    a_base,fattibile = check_finestre_temporali(clienti,rotte_new,r1,k,a_base,D)

    if fattibile == False :
        return rotte,a,False
    
    #-------------destination route----------------
    f_best = objective_function(rotte,D) # objective function value computed before move execution
    pos_best = -1 
    a_best = a_base.copy() # arrival times vector of the best solution
    rotte_best = [r[:] for r in rotte_new] # best route depending on the best insertion position

    # all the possible insertion positions are tested and the one with the best objective function value is selected
    for curr_pos in range(1,len(rotte_new[r2])) :

        # a temporary structure  (rotte_tmp) is used to test the move feasibiliy
        a1 = a_base.copy()
        rotte_tmp = [r[:] for r in rotte_new]
        
        # insertion at position curr_pos in route_tmp
        rotte_tmp[r2].insert(curr_pos,new_client) 
        
        # check time constraints 
        a1,fattibile = check_finestre_temporali(clienti,rotte_tmp,r2,curr_pos,a1,D)

        if fattibile == False :
            continue

        # check vehicle capacity
        demand = sum(clienti[rotte_tmp[r2][i]]['demand'] for i in range(len(rotte_tmp[r2])))

        if demand > capacita :
            continue

        f_tmp = objective_function(rotte_tmp,D)

        # track the position with the best value of F
        if f_tmp < f_best :
            pos_best = curr_pos
            f_best = f_tmp
            a_best = a1.copy()
            rotte_best = [r[:] for r in rotte_tmp] # routes updated

    if pos_best == -1 :
            return rotte,a,False
    
    if len(rotte_best[r1]) == 2 : 
        del rotte_best[r1] # route r1 is closed (if it contains only one client)
    
    return rotte_best,a_best,True


'''
Randomized 2-opt move: tries to reverse the segment (i+1, j) in a randomly selected route
'''
def randomized_move_2_opt(clienti, rotte, D, a):
    
    num_rotte = len(rotte)

    if num_rotte == 0:
        return rotte, a,False

    #Randomly chosen route (it must not be empty)
    candidate_routes = [idx for idx, r in enumerate(rotte) if len(r) > 4] # Minimum 5 nodes: Dep-C1-C2-C3-Dep per 2-opt
    if not candidate_routes:
        return rotte, a, False
    
    id_rotta = random.choice(candidate_routes)
    
    n = len(rotte[id_rotta]) # route length (include the depot)

    # route too small for a 2-opt
    # in order to perform a 2-opt move (removal of two non-adjacent arcs),
    # the route must have at least 5 nodes: Dep(0) - C1(1) - C2(2) - C3(3) - Dep(4)

    if n < 5:
        return rotte, a, False

    # 2. selection of the cut points (i and j)
    # index i is the node BEFORE the reversed segment
    # index j is the LAST node of the reversed segment
    # n-5: the last client node valid as i is n-5 (so that j is at most n-2)
    i = random.randint(1, n - 4)  # 1 <= i <= n-4 (jmax-1)
    j = random.randint(i + 2, n - 2) # i+2 <= j <= n-2 (there must be at least one node between i and j, and one node after j)
    
    # 3. New solution preparation
    a_new = a.copy()
    rotte_new = [r[:] for r in rotte]

    segment_inv = [] # inverted egment 
    r_idx = j    # read index (old route), starts from j
    
    # 4. Implementation of the segment inversion [i+1, j]
    # inversion cycle: copy nodes [i+2...j] in reverse order
    while r_idx >= (i + 1):
        
        #Add node r_idx (frome the old route) in segment
        segment_inv.append(rotte[id_rotta][r_idx]) # read the segment backwards
        
        r_idx -= 1
        
    # [a, b]: b is the end index and is excluded (+ : list concatenation)
    rotte_new[id_rotta]= rotte[id_rotta][:i+1] + segment_inv + rotte[id_rotta][j+1:]

    a_new, fattibile = check_finestre_temporali(clienti, rotte_new,id_rotta, i + 1, a_new, D)

    if fattibile == False:
        return rotte,a, False
    
    f_new = objective_function(rotte_new,D)
    f_curr = objective_function(rotte,D)

    #objective function improved
    if f_new < f_curr :
            return rotte_new,a_new,True
    
    return rotte,a,False



'''SimulatedAnnealing Algorithm with Adaptive Probabilites'''
def SimulatedAnnealingDynamic(T0, Tf, alfa, Lk, X_0, a, clienti, D, capacita, w0) :

    X = [r[:] for r in X_0]
    X_best = [r[:] for r in X_0]
    f_best = objective_function(X_0, D)
    a_best = a.copy()
    T = T0

    num_moves = 3  # an index is assigned to every move: 0=INSERTION, 1=2-OPT, 2=CROSS_EXCHANGE
    weights = w0  # initial weigths
    K_interval = 10
    scores = np.zeros(num_moves) 
    usage = np.zeros(num_moves)  # usage counter
    gamma = 0.1 # reaction factor (gamma): how fast the weights adapt (typically 0.1)

    # Computation of initial probabilites (normalization)
    Probs = weights / np.sum(weights)

    temp_cycle = 0 # counter of temperature cycle
    
    #print('T,BestInsertionP,Randomized2optP,CrossExchangeP,f_best')
    
    while T > Tf :
        
        temp_cycle += 1

        for repeat in range(0, Lk) :
            
            a1 = a.copy()
            
            # random.choice select move index depeding on probabilities P
            move= np.random.choice(num_moves, p=Probs) 
            usage[move] += 1 # Conto l'utilizzo

            fattibile = False
            
            if move == 0: # INSERTION (Move 0)
                X1, a1, fattibile = best_insertion_move(clienti, X, D, a1, capacita)
            elif move == 1 : # 2-OPT (Move 1)
                X1, a1, fattibile = randomized_move_2_opt(clienti, X, D, a1)
            else : # CROSS_EXCHANGE (Move 2)
                X1, a1, fattibile = cross_exchange_move(clienti, X, D, a1, capacita)
            
            if fattibile == False :
                continue

            f_X1 = objective_function(X1, D)
            f_X  = objective_function(X, D)
            delta_f = f_X1 - f_X
            
            reward = 0

             # 1. Successful move : improve the current solution 
            if delta_f < 0 :
                X = [r[:] for r in X1]
                a = a1.copy()
                reward = 50  # S2 score (local success)

                
           # 2. Acceptance move: the solution gets worse but is accepted by SA (exploration)
            else  :
                r = random.random()
                if r < np.exp(-(delta_f) / T) :
                    X = [r[:] for r in X1]
                    a = a1.copy()
                    if delta_f > 0:
                        reward = 5

            # 3. Global update: update the global best
            if f_X1 < f_best:
                #print(f"T={T:.2f} | **NUOVO BEST**:  {f_X1:.4f}")
                X_best = [r[:] for r in X1]
                f_best = f_X1
                a_best = a.copy()
                reward = 100 # S1 score (maximum success)

            if reward > 0 :
                scores[move] += reward 

        if temp_cycle % K_interval == 0:
       
            for i in range(num_moves):

                if usage[i] > 0:
                
                    # average success rate" computation ( score / usage)
                    success_rate = scores[i] / usage[i]
        
                    # weights update (exponential moving average)
                    weights[i] = weights[i] * (1-gamma) + (gamma) * success_rate

            # Normalization to get the new probabilites 
            Probs = weights / np.sum(weights)

            print(f"T={T:.2f} | Nuove Probabilità: {Probs} | f_best : {f_best}")
            #print(f"{T:.2f},{Probs[0]},{Probs[1]},{Probs[2]},{f_best}")

            scores.fill(0) 
            usage.fill(0)

        T = T * alfa

    return X_best, f_best, a_best


'''SimulatedAnnealing Algorithm with Fixed Probabilites'''
def SimulatedAnnealingFixed(T0, Tf, alfa, Lk, X_0, a, clienti, D, capacita, w0) :

    X = [r[:] for r in X_0]
    X_best = [r[:] for r in X_0]
    f_best = objective_function(X_0, D)
    a_best = a.copy()
    T = T0

    num_moves = 3  # an index is assigned to every move: 0=INSERTION, 1=2-OPT, 2=CROSS_EXCHANGE
    weights = w0  # initial weigth
    scores = np.zeros(num_moves) 
    usage = np.zeros(num_moves) 
    # computation of initial probabilities  PROBABILITA iniziali (normalization)
    Probs = weights / np.sum(weights)
    
    print(Probs)

    while T > Tf :
    

        for repeat in range(0, Lk) :
            
            a1 = a.copy()
            
            # random.choice select move index depending on probabilites P
            move= np.random.choice(num_moves, p=Probs) 

            fattibile = False
            
            if move == 0: # INSERTION (Move 0)
                X1, a1, fattibile = best_insertion_move(clienti, X, D, a1, capacita)
            elif move == 1 : # 2-OPT (Move 1)
                X1, a1, fattibile = randomized_move_2_opt(clienti, X, D, a1)
            else : # CROSS_EXCHANGE (Move 2)
                X1, a1, fattibile = cross_exchange_move(clienti, X, D, a1, capacita)
            
            if fattibile == False :
                continue

            f_X1 = objective_function(X1, D)
            f_X  = objective_function(X, D)
            delta_f = f_X1 - f_X
            
            reward = 0

            # 1. successful move : improve the current solution 
            if delta_f < 0 :
                X = [r[:] for r in X1]
                a = a1.copy()
                
            # 2. acceptance move: the solution gets worse but is accepted by SA (exploration)
            else  :
                r = random.random()
                if r < np.exp(-(delta_f) / T) :
                    X = [r[:] for r in X1]
                    a = a1.copy()

            # 3. global update: update the global best
            if f_X1 < f_best:
                #print(f"T={T:.2f} | **NUOVO BEST**:  {f_X1:.4f}")
                X_best = [r[:] for r in X1]
                f_best = f_X1
                a_best = a.copy()
       
        T = T * alfa

    return X_best, f_best, a_best





if __name__ == '__main__' :

   # filename = "solomon100\In\c101.txt"
    #filename = "solomon100\In\\r101.txt"
    #filename = "solomon100\In\\rc101.txt"
    #filename = "solomon100\In\\c203.txt"
   # filename = "solomon100\In\c204.txt" #da provare
    #filename = "solomon100\In\c108.txt"
    #filename = "solomon100\In\c208.txt"
    #filename = "solomon100\In\\rc208.txt"  #da provare (TESINA)
    #filename = "solomon100\In\\rc208.txt"  #da provare
    #filename = "solomon100\In\\r208.txt"  #da provare (TESINA)
  # filename = "solomon100\In\\rc104.txt" 
    #filename = "solomon200\C2_2_1.TXT" #da provare
    #filename = "solomon200\C2_2_3.TXT" #da provare
    #filename = "solomon200\RC2_2_8.TXT" #da provare
   # filename = "solomon200\RC2_2_5.TXT" #da provare (TESINA)
    #filename = "solomon200\R2_2_7.TXT" #da mettere su tesina è interessante
    #filename = "solomon200\R1_2_5.TXT"
    filename = "solomon200\R1_2_1.TXT"
   # filename = "solomon200\RC2_2_9.TXT" #da mettere nella tesina è bello
    #filename = "solomon200\RC2_2_4.TXT" #da mettere nella tesina è bello
    #filename = "solomon200\R1_2_8.TXT" #istanza difficile
    #filename = "solomon200\R2_2_4.TXT" #da mettere su tesina è 
    #filename = "solomon200\R2_2_8.TXT" #da mettere su tesina è interessante
    #filename = "solomon200\RC1_2_4.TXT" 
    #filename = "solomon200\R2_2_3.TXT"  #(TESINA)
    #metti il doppio \ se il nome inizia per r (non confondere con \r)
    # Leggi il file
    num_veicoli, capacita, deposito, clienti = leggi_file_solomon(filename)

    clienti.insert(0,deposito)

    #print(clienti)
    D = distances_matrix(deposito,clienti)

    #r208	2	726.82
    #rotte,U,a = TimeOrientedNN(clienti,D,capacita,num_veicoli,0.4,0.3,0.3)

    #print(a)
    delta,miglior_punteggio = get_best_params(clienti,D,capacita,num_veicoli,passo=0.1)

    #print(delta)
    #print(miglior_punteggio)
    
    rotte,U,a = TimeOrientedNN(clienti,D,capacita,num_veicoli,delta[0],delta[1],delta[2])
    print(len(U))
    #rotte,U,a = TimeOrientedNN(clienti,D,capacita,num_veicoli,0.4,0.6,0.0)
    #print(len(U))
    #plot_rotte(rotte,clienti)

    f0 = objective_function(rotte,D)

    for route in rotte :
        print(route)

    print(f"Funzione obiettivo iniziale (f0): {f0}")
    
    #T0 = 10 *abs(0.5*f0)
    T0 = 10 *abs(0.5*f0)
    Tf = abs(f0)*10**(-4)
    alfa = 0.99
    Lk =  (len(clienti)-1)
    #w0 = [3.0,4.0,3.0]
    #w0 = [5.0,3.0,2.0]
    #w0 = [2.0,3.0,5.0]
    w0 = [1.0,7.0,2.0]

    # --- Inizio misurazione SA ---
    start_time_sa = time.time()

    #X,f,a_best = SimulatedAnnealingDynamic(T0,Tf,alfa,Lk,rotte,a,clienti,D,capacita,w0)

    X,f,a_best = SimulatedAnnealingFixed(T0,Tf,alfa,Lk,rotte,a,clienti,D,capacita,w0)

    end_time_sa = time.time()
    
    #execution time required
    duration_sa = end_time_sa - start_time_sa

    #show duration and final objective function value
    print(f"\nFunzione obiettivo finale (f): {f}")
    print(f"Durata Simulated Annealing: {duration_sa:.4f} secondi")

    #print(f)

    for route in X :
       print(route)

    plot_rotte(X,clienti)
    
