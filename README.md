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
- each client is served by a single vehicle;
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

