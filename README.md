# simulated-annealing-cvrptw

The **Capacitated Vehicle Routing** Problem is one of the most famous combinatorial optimization problems. 
Given $m$ vehicles, which can be homogeneous or not (e.g. same capacity, same
service time and so on), and $n$ clients, the purpose is to compute, for each
vehicle, the route that starts from a depot, serves the assigned clients and
returns to the initial point, optimizing an objective function $F$.
The problem can be seen as a merge of two combinatorial problems:
- Travel Salesman Problem.
- Clustering.

There are different versions of this problem. This project focuses on the variant
where each client is associated with a time window bounded below by the ready time
$l_i$ (the instant when the service of the client can start) and bounded above by
the due time $L_i$ (the last instant when the client can be served).

The following assumptions are adopted:
- each client is served by a single vehicle.
- each graph arc is crossed by a single vehicle.
- the vehicles are homogeneous, which means they have the same capacity and the same features.

## Simulated Annealing

It is one of the most used metaheuristics to solve combinatorial optimization problems. It belongs to the family of algorithms that simulate natural phenomena.
The phenomena is **annealing process of a solid**:
- The solid starts at an initial high temperature (close to the **melting point**) and the molecules within the solid follow a **random distribution**
- Reducing the temperature slowly, the particles will try to position themselves in an equilibrium state
- If the cooling process were too fast, the particles would not have time to position themselves and this would cause the loss of the initial structure

The **Metropolis algorithm** allows the simulation of this process and is shown in the pseudocode reported below:

```
SimulatedAnnealing(T0, Tf, alpha, Lk)

    X      = InitialSolution()
    X_best = X
    F_best = F(X)
    T      = T0

    while T > Tf:
        for repeat = 1 to Lk:
            X', feasible = apply_move(X)
            if feasible == False:
                continue

            delta = F(X') - F(X)

            if delta > 0:
                R = Uniform(0, 1)
                if R < exp(-delta / T):
                    X = X'
            else:
                X = X'

            if F(X') < F(X_best):
                F_best = F(X')
                X_best = X

        T = alpha * T        # or another cooling schedule

    return X_best, F_best
```

The specific parameters to tune carefully are:
- $T_0$ : initial temperature that must be sufficiently high.
- $T_f$ : final temperature at which the cooling process ends
- $\alpha$ : parameter used in the exponential cooling law to compute the next temperature value (e.g. $T_{k+1} = \alpha \cdot T_k$)
- $L_k$ : number of transitions (or moves) to perform at the same temperature value (the value should be chosen according to the $\alpha$ parameter)

If solution $X'$ produced a worse value of the objective function $F(X')$, this worsening would be accepted, but with a probability equal to $e^{-\Delta/T}$,
where $\Delta = F(X') - F(X)$. 
- This idea can be implemented by simply generating a random number $r$ in $[0,1]$ and checking if it is lower than $e^{-\Delta/T}$.

## Initial solution generation (TONN approach)

Tale sezione si propone di presentare l’euristica utilizzata per produrre la soluzione iniziale 𝑆 da cui poi partire per l’applicazione della meta-euristica Simulated Annealing. Il modo in cui la soluzione iniziale è stata determinata è di base molto semplice, e si basa su una euristica greedy
Partendo dal deposito, di volta in volta, viene considerato come successivo cliente da servire quello piu vicino, dove il concetto di vicinanza deve tenere in conto sia la distanza geografica che temporale. Naturalmente l’aggiunta di un nodo nella rotta corrente non deve provocare né la violazione delle finestre temporali né i limiti di capienza massima dei veicoli. Al verificarsi di una situazione del genere si puo creare una nuova rotta (quindi usare un veicolo in piu) e aggiungere il nuovo cliente.
Se 𝑖 è l’ultimo cliente servito nella rotta corrente e 𝑗 un generico cliente non ancora servito, cioè, appartenente all’insieme dei clienti “unrouted” 𝑼 definiamo con 𝑐𝑖𝑗 la “metrica” di vicinanza tra i due nodi. Tale metrica è combinazione lineare di tre parametri: 𝒄𝒊𝒋=𝜹𝟏𝒅𝒊𝒋+𝜹𝟐𝑻𝒊𝒋+𝜹𝟑𝒗𝒊𝒋
- 𝑫𝒊𝒋 : indica la distanza fisica tra i due nodi.
- 𝑻𝒊𝒋 : indica la distanza temporale che possiamo espressa come differenza tra l’istante di inizio servizio del nodo 𝑗 e la fine del servizio del nodo 𝑖 cioè, in termini matematici : 𝑻𝒊𝒋=𝒂𝒋 − (𝒂𝒊+𝒔𝒊).
- 𝒗𝒊𝒋 : indica il tempo rimanente affinché il servizio del cliente 𝑗 possa iniziare senza eccedere la finestra temporale (cioè, l’estremo superiore 𝐿𝑖), in termini matematici : 𝒗𝒊𝒋=𝑳𝒊− (𝒂𝒊+𝒔𝒊+𝒕𝒊𝒋).
I tre parametri 𝛿1,𝛿2,𝛿3 sono non negativi e la loro somma deve essere 1 (𝛿1 + 𝛿2+𝛿3 = 1). La loro scelta va effettuata empiricamente sulla base delle istanze che si hanno a disposizione.

Tale semplice euristica è stata proposta da Solomon e prende il nome di Time Oriented Nearest Neighbor Heuristic

## Implemented moves
`cross_exchange_move()` : inter-route move that randomly chooses two routes $r_1$
and $r_2$ and two segments $r_1[p:]$ and $r_2[q:]$, and performs the exchange,
checking whether the derived solution meets the constraints.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/d1d55620-a8f5-46a2-bf81-ffa39f1ad4a6" />

`best_insertion_move()` : inter-route move that randomly chooses two routes $r_1$
and $r_2$, removes a client from one and inserts it in the best position in the
other route (the position that gives the best value of the objective function $F$).

<img width="800" alt="image" src="https://github.com/user-attachments/assets/b3a18ce7-fca9-4e35-a28b-ce669c9a27a9" />

`randomized_2opt()` — famous intra-route move that selects two non-adjacent arcs
$(u_i, u_{i+1})$ and $(u_j, u_{j+1})$ and links $u_i$ with $u_j$ and $u_{i+1}$ with
$u_{j+1}$, then reverses the direction of the arcs, orienting them from $j$ to
$i+1$. The two arcs are chosen randomly in this move implementation.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/fd9ab489-323f-41b2-988a-4dde6c898818" />

## Testing with Adaptive Probabilites

The instances used during testing are available at
[SINTEF's Solomon benchmark page](https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/).

**Instance R208 (100 customers):**
- Final objective function value: $f = 776.60$
- Simulated annealing duration: $t = 27.2881$ s (with $\alpha = 0.98$)
- Initial weights: $w_0 = [0.3, 0.4, 0.3]$ (initial probabilities = [30%, 40%, 30%])

<img width="800" alt="image" src="https://github.com/user-attachments/assets/16fddcd6-6a46-4a95-a6bb-b58fa670c30e" />

<img width="800" alt="image" src="https://github.com/user-attachments/assets/9f7971c0-1a2a-4e48-870e-f30c228fd8bc" />

<img width="800" alt="image" src="https://github.com/user-attachments/assets/ef223a47-8fe6-46d5-805c-077449ec1817" />

**Instance R2_2_3 (200 customers):**
- Final objective function value 𝑓 = 2865.8053590536174
- Simulated annealing duration: t = 77.2340 s (𝛼=0.99)
- Initial weights 𝑤0 = [3.0,4.0,3.0] (initial probabilites = [30%,40%,30%])

<img width="800" src="https://github.com/user-attachments/assets/98f01032-723a-4cc6-9703-42badabad4f6" />

<img width="800" alt="image" src="https://github.com/user-attachments/assets/3babf063-ddbd-4c32-830e-9332abc2dfa1" />

<img width="800" alt="image" src="https://github.com/user-attachments/assets/e6e668ca-47c0-4ac6-b6f1-5a0add801ba8" />


