import gurobipy as gp
from gurobipy import GRB
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

    #clienti.insert(0, deposito)  # 0 is the depot index
    n = len(clienti)
    print(n)

    D = np.zeros((n,n))

    #distance between the node and itself
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
    
    #Draw deposit
    depot_x = clienti[deposito]['x']
    depot_y = clienti[deposito]['y']
    plt.scatter(depot_x, depot_y, c='red', s=150, marker='s', label='Deposito') 

    #Draw client
    for i, c in enumerate(clienti):
        if i != deposito:
            plt.scatter(c['x'], c['y'], c='black', s=40)
            plt.text(c['x'] + 0.8, c['y'] + 0.8, str(i), fontsize=8)
         
    
    #Draw routes
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

    demands = [0] #the demand of the first route is 0 
    rotte = [[deposito]] # the route initially contains only the depot
    a = np.zeros(N) #a_i = initial service temp of node i
    s = [clienti[i]['service_time'] for i in range(N)] #service times of the clients
    
    a[0] = clienti[deposito]['ready_time'] #base case
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
            rotte[k].append(deposito) # close the current route 
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

            U = [c for c in U if c['id'] != nodo_curr] #remove nodo_curr from unrouted clients
            
    
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


'''Funzione di calibrazione che ricava migliori parametri per TimeOrientedNN
'''
def get_best_params(clienti, D, capacita, num_veicoli, passo=0.1):

    migliori_parametri = None
    miglior_punteggio = float('inf')
    
    print(f"Inizio calibrazione parametri con passo={passo}...")

    # Itera su delta1 da 0 a 1
    for d1 in np.arange(0, 1.01, passo):
        # Itera su delta2 da 0 fino a quel che resta per arrivare a 1
        for d2 in np.arange(0, 1.01 - d1, passo):

            d3 = abs(1.0 - d1 - d2)
            
            # Arrotondiamo per evitare imprecisioni del floating point
            params = (round(d1, 2), round(d2, 2), round(d3, 2))
            
            # Esegui l'euristica con i parametri correnti
            rotte, clienti_non_serviti, a = TimeOrientedNN(clienti, D, capacita, num_veicoli, params[0], params[1], params[2])
            
            # Consideriamo la soluzione valida SOLO se tutti i clienti sono stati serviti
            if len(clienti_non_serviti) == 0:
                punteggio_attuale = objective_function(rotte, D)
                
                if punteggio_attuale < miglior_punteggio:
                    miglior_punteggio = punteggio_attuale
                    migliori_parametri = params
                    print(f"  -> Nuovo miglior punteggio: {miglior_punteggio:.2f} con parametri {migliori_parametri}")

    if migliori_parametri is None:
        print("ATTENZIONE: Nessuna combinazione di parametri ha prodotto una soluzione valida (tutti i clienti serviti).")
        return (0.0, 0.0, 0.0), float('inf') # Ritorna valori di default
        
    print("Calibrazione completata.")
    return migliori_parametri, miglior_punteggio


'''
Cross Exchange move
'''
def cross_exchange_move(clienti,rotte,D,a,capacita) :

    num_rotte = len(rotte)

    if num_rotte < 2 : #Non si puo fare uno swap tra due rotte se ne esiste solo una
        return rotte,a,False

    #genero randomicamente gli indici delle due rotte
    r1 = random.randint(0,num_rotte-1)
    r2 = random.choice([i for i in range(0,num_rotte) if i != r1])

    #saranno rispettivamente le njuove rotte e il nuovo vettore dei cos
    rotte_new = [r[:] for r in rotte]
    a_new = [a[i] for i in range(len(a))]
    
    if len(rotte_new[r1]) <= 2 and len(rotte_new[r2]) <= 2 :
        return rotte,a,False

    #genero randomicamente gli indici dei due clienti delle due rotte da scambiare
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

    #1)verifica capacita veicolo per ambedue le rotte
    domanda_totale_r1 = sum([clienti[rotte_new[r1][i]]['demand'] for i in range(0,len(rotte_new[r1]))])
    
    if domanda_totale_r1 > capacita :
        return rotte,a,False
    
    domanda_totale_r2 = sum([clienti[rotte_new[r2][i]]['demand'] for i in range(0,len(rotte_new[r2]))])

    if domanda_totale_r2 > capacita :
        return rotte,a,False

    #2)verifica vincoli temporali sulla rotta r1
    a_new,fattibile = check_finestre_temporali(clienti,rotte_new,r1,p+1,a_new,D)

    if fattibile == False :
        return rotte,a,False

    #3)verifica vincoli temporali sulla rotta r2
    a_new,fattibile = check_finestre_temporali(clienti,rotte_new,r2,q+1,a_new,D)

    if fattibile == False :
        return rotte,a,False
    
    f_new = objective_function(rotte_new,D)
    f_curr = objective_function(rotte,D)

    #se funzione obiettivo migliorata
    if f_new < f_curr :
            return rotte_new,a_new,True
    
    return rotte,a,False



'''Funzione che verifica in una rotta route_idx, partendo da una posizione pos in poi che tutti rispettino
i vincoli temporali sulle finestre '''
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
    

'''Mossa di best insertion di un cliente da una rotta in un altra entrambe scelte casualmente
'''
def best_insertion_move(clienti, rotte, D, a, capacita):

    num_rotte = len(rotte)

    if num_rotte < 2 :
        return rotte,a,False

    a_base = a.copy()
    rotte_new = [r[:] for r in rotte]

    #--------------rotta di partenza controllo che il cliente possa essere spostato-------------------
    r1 = random.randint(0,num_rotte-1) #scelta randomica rotta sorgente
    r2 = random.choice([i for i in range(len(rotte_new)) if i!=r1])


    if len(rotte_new[r1]) <= 2 :
        return rotte,a,False

    k = 1
    if len(rotte_new[r1]) > 3 :    
        k = random.randint(1,len(rotte_new[r1])-2)

    new_client = rotte_new[r1][k]
    rotte_new[r1].remove(new_client) #rimuovo il cliente id new_cliente dalla prima rotta

    #dopo la rimozione controllo la fattibilita temporale
    a_base,fattibile = check_finestre_temporali(clienti,rotte_new,r1,k,a_base,D)

    if fattibile == False :
        return rotte,a,False
    
    #-------------rotta di destinazione----------------
    f_best = objective_function(rotte,D) #valore della funzione obiettivo calcolata sulle rotte prima della mossa
    pos_best = -1 
    a_best = a_base.copy() #vettore dei tempi di arrivo ai della soluzione migliori
    rotte_best = [r[:] for r in rotte_new] #rotte migliori in base alla miglior posizione di inserimento

    #provo le possibili posizioni di inserimento e cerco la migliore in termini di minimizzazione dell'obiettivo
    for curr_pos in range(1,len(rotte_new[r2])) :

        #uso queste strutture dati di appoggio in modo da fare l'inserimento
        #non definitivamente e testare la fattibilita
        a1 = a_base.copy()
        rotte_tmp = [r[:] for r in rotte_new]
        
        #inserisco nella posizione curr_pos di rotte_tmp
        rotte_tmp[r2].insert(curr_pos,new_client) 
        
        #verifico validita vincoli temporali
        a1,fattibile = check_finestre_temporali(clienti,rotte_tmp,r2,curr_pos,a1,D)

        if fattibile == False :
            continue

        #verifico validitia vincoli di capienza dei veicoli
        demand = sum(clienti[rotte_tmp[r2][i]]['demand'] for i in range(len(rotte_tmp[r2])))

        if demand > capacita :
            continue

        f_tmp = objective_function(rotte_tmp,D)

        #tengo traccia della posizione di inserimento piu efficiente
        if f_tmp < f_best :
            pos_best = curr_pos
            f_best = f_tmp
            a_best = a1.copy()
            rotte_best = [r[:] for r in rotte_tmp] #aggiorno le rotte

    if pos_best == -1 :
            return rotte,a,False
    
    if len(rotte_best[r1]) == 2 : #avendo rimosso l'unico cliente nella rotta r1
        del rotte_best[r1] #la rotta r1 puo essere chiusa
    
    return rotte_best,a_best,True


'''
Randomized 2opt move. Tenta di invertire un segmento (i+1, j) in una rotta scelta casualmente (o specificata).
'''
def randomized_move_2_opt(clienti, rotte, D, a):
    
    num_rotte = len(rotte)

    if num_rotte == 0:
        return rotte, a,False

    # 1.Scelta della Rotta scegli una rotta a caso che non sia vuota (dopo i depositi)
    candidate_routes = [idx for idx, r in enumerate(rotte) if len(r) > 4] # Minimo 5 nodi: Dep-C1-C2-C3-Dep per 2-opt
    if not candidate_routes:
        return rotte, a, False
    
    id_rotta = random.choice(candidate_routes)
    
    n = len(rotte[id_rotta]) # Lunghezza della rotta, include i depositi iniziali e finali

    # Caso Limite: La rotta è troppo corta per un'inversione 2-opt
    # Per eseguire una 2-opt (rimozione di due archi non adiacenti)
    # la rotta deve avere almeno 5 nodi: Dep(0) - C1(1) - C2(2) - C3(3) - Dep(4)
    if n < 5:
        return rotte, a, False

    # 2.Scelta dei Punti di Taglio (i e j)
    #L'indice i è il nodo PRECEDENTE al segmento invertito
    #L'indice j è l'ULTIMO nodo del segmento invertito
    # n-5: ultimo nodo cliente valido come i è n-5 (così j è al massimo n-2)
    i = random.randint(1, n - 4)  # 1 <= i <= n-4 (jmax-1)
    j = random.randint(i + 2, n - 2) # i+2 <= j <= n-2 (deve esserci almeno un nodo tra i e j, e uno dopo j)
    
    # 3.Preparazione della Nuova Soluzione
    # Usiamo una copia profonda di tutta la struttura per sicurezza
    a_new = a.copy()
    rotte_new = [r[:] for r in rotte]

    segment_inv = [] #segmento invertito
    r_idx = j    # Indice di lettura (Vecchia Rotta), parte da j
    
    # 4. Implementazione dell'Inversione del Segmento [i+1, j]
    # Ciclo di Inversione: copia i nodi [i+2...j] in ordine inverso
    while r_idx >= (i + 1):
        #Aggiungi il nodo r_idx (dalla vecchia rotta) in segment
        segment_inv.append(rotte[id_rotta][r_idx]) #leggo il segmento al contrario
        
        r_idx -= 1
        
    #[a,b] b indica l'indice finale ed è escluso (+ : concatenazione di liste)
    rotte_new[id_rotta]= rotte[id_rotta][:i+1] + segment_inv + rotte[id_rotta][j+1:]

    #NOTA:I segmenti rotte_new[rotta_idx][0:i] e rotte_new[rotta_idx][j+1:] sono rimasti invariati grazie alla copia iniziale.
    #altro modo : rotta_new = rotta[:i+1] + rotta[i+1:j+1][::-1] + rotta[j+1:]

    a_new, fattibile = check_finestre_temporali(clienti, rotte_new,id_rotta, i + 1, a_new, D)

    if fattibile == False:
        return rotte,a, False
    
    f_new = objective_function(rotte_new,D)
    f_curr = objective_function(rotte,D)

    #se funzione obiettivo migliorata
    if f_new < f_curr :
            return rotte_new,a_new,True
    
    return rotte,a,False



'''Funzione chiave : SimulatedAnnealing'''
# Supponiamo che le funzioni 'objective_function', 'best_insertion_move', 
# 'randomized_move_2_opt', 'cross_exchange_move' e le variabili 'clienti', 
# 'D', 'capacita' siano definite altrove.
def SimulatedAnnealingDynamic(T0, Tf, alfa, Lk, X_0, a, clienti, D, capacita, w0) :

    X = [r[:] for r in X_0]
    X_best = [r[:] for r in X_0]
    f_best = objective_function(X_0, D)
    a_best = a.copy()
    T = T0

    num_moves = 3  # Assegna un indice a ogni mossa: 0=INSERTION, 1=2-OPT, 2=CROSS_EXCHANGE
    weights = w0  # Pesi iniziali
    K_interval = 10
    scores = np.zeros(num_moves) 
    usage = np.zeros(num_moves)  # Contatore di utilizzo nell'intervallo K
    gamma = 0.1 # Fattore di reazione (gamma): quanto rapidamente i pesi si adattano (0.1 è tipico)

    # Calcola le PROBABILITA iniziali (semplice normalizzazione)
    Probs = weights / np.sum(weights)

    temp_cycle = 0 # Contatore del ciclo di temperatura
    
    #print('T,BestInsertionP,Randomized2optP,CrossExchangeP,f_best')
    
    while T > Tf :
        
        temp_cycle += 1

        for repeat in range(0, Lk) :
            
            a1 = a.copy()
            
            # random.choice seleziona l'indice in base alle probabilità P
            move= np.random.choice(num_moves, p=Probs) 
            usage[move] += 1 # Conto l'utilizzo

            fattibile = False
            
            if move == 0: # INSERTION (Mossa 0)
                X1, a1, fattibile = best_insertion_move(clienti, X, D, a1, capacita)
            elif move == 1 : # 2-OPT (Mossa 1)
                X1, a1, fattibile = randomized_move_2_opt(clienti, X, D, a1)
            else : # CROSS_EXCHANGE (Mossa 2)
                X1, a1, fattibile = cross_exchange_move(clienti, X, D, a1, capacita)
            
            if fattibile == False :
                continue

            f_X1 = objective_function(X1, D)
            f_X  = objective_function(X, D)
            delta_f = f_X1 - f_X
            
            reward = 0

            # 1. Mossa di successo: Migliora la soluzione corrente
            if delta_f < 0 :
                X = [r[:] for r in X1]
                a = a1.copy()
                reward = 50  # Punteggio S2 (Successo locale)

                
            # 2. Mossa di accettazione: Peggiora ma viene accettata da SA (esplorazione)
            else  :
                r = random.random()
                if r < np.exp(-(delta_f) / T) :
                    X = [r[:] for r in X1]
                    a = a1.copy()
                    if delta_f > 0:
                        reward = 5

            # 3. Aggiornamento globale: Aggiorna il best globale
            if f_X1 < f_best:
                #print(f"T={T:.2f} | **NUOVO BEST**:  {f_X1:.4f}")
                X_best = [r[:] for r in X1]
                f_best = f_X1
                a_best = a.copy()
                reward = 100 # Punteggio S1 (Massimo successo)

            if reward > 0 :
                scores[move] += reward 

        if temp_cycle % K_interval == 0:
       
            for i in range(num_moves):

                if usage[i] > 0:
                
                    # Calcolo del "successo medio" in questo intervallo: Punti / Utilizzo
                    success_rate = scores[i] / usage[i]
        
                    # Aggiornamento dei pesi (Media mobile smorzata)
                    weights[i] = weights[i] * (1-gamma) + (gamma) * success_rate

            # Normalizzazione per ottenere le nuove probabilità
            Probs = weights / np.sum(weights)


            print(f"T={T:.2f} | Nuove Probabilità: {Probs} | f_best : {f_best}")
            #print(f"{T:.2f},{Probs[0]},{Probs[1]},{Probs[2]},{f_best}")


            scores.fill(0) 
            usage.fill(0)

        T = T * alfa

    return X_best, f_best, a_best


def SimulatedAnnealingFixed(T0, Tf, alfa, Lk, X_0, a, clienti, D, capacita, w0) :

    X = [r[:] for r in X_0]
    X_best = [r[:] for r in X_0]
    f_best = objective_function(X_0, D)
    a_best = a.copy()
    T = T0

    num_moves = 3  # Assegna un indice a ogni mossa: 0=INSERTION, 1=2-OPT, 2=CROSS_EXCHANGE
    weights = w0  # Pesi iniziali
    scores = np.zeros(num_moves) 
    usage = np.zeros(num_moves)  # Contatore di utilizzo nell'intervallo K
    # Calcola le PROBABILITA iniziali (semplice normalizzazione)
    Probs = weights / np.sum(weights)
    
    print(Probs)

    while T > Tf :
    

        for repeat in range(0, Lk) :
            
            a1 = a.copy()
            
            # random.choice seleziona l'indice in base alle probabilità P
            move= np.random.choice(num_moves, p=Probs) 

            fattibile = False
            
            if move == 0: # INSERTION (Mossa 0)
                X1, a1, fattibile = best_insertion_move(clienti, X, D, a1, capacita)
            elif move == 1 : # 2-OPT (Mossa 1)
                X1, a1, fattibile = randomized_move_2_opt(clienti, X, D, a1)
            else : # CROSS_EXCHANGE (Mossa 2)
                X1, a1, fattibile = cross_exchange_move(clienti, X, D, a1, capacita)
            
            if fattibile == False :
                continue

            f_X1 = objective_function(X1, D)
            f_X  = objective_function(X, D)
            delta_f = f_X1 - f_X
            
            reward = 0

            # 1. Mossa di successo: Migliora la soluzione corrente
            if delta_f < 0 :
                X = [r[:] for r in X1]
                a = a1.copy()
                
            # 2. Mossa di accettazione: Peggiora ma viene accettata da SA (esplorazione)
            else  :
                r = random.random()
                if r < np.exp(-(delta_f) / T) :
                    X = [r[:] for r in X1]
                    a = a1.copy()

            # 3. Aggiornamento globale: Aggiorna il best globale
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

    # --- Fine misurazione SA ---
    end_time_sa = time.time()
    
    # Calcola la durata
    duration_sa = end_time_sa - start_time_sa

    # Stampa i risultati e il tempo
    print(f"\nFunzione obiettivo finale (f): {f}")
    print(f"Durata Simulated Annealing: {duration_sa:.4f} secondi")

    #print(f)

    for route in X :
       print(route)

    plot_rotte(X,clienti)
    
