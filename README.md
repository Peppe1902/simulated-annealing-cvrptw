# simulated-annealing-cvrptw
Simulated annealing applied to the Capacitated Vehicle Routing Problem with Time Windows

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

