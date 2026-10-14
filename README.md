# simulated-annealing-cvrptw
Simulated annealing applied to the Capacitated Vehicle Routing Problem with Time Windows

It is one of the most used metaheuristics to solve combinatorial optimization problems. It belongs to the family of algorithms that simulate natural phenomena.
The phenomena is annealing process of a solid:
- physical process where the solid starts at an initial high temperature (close tothe melting point) and the molecules within the solid follow a random distribution
- Reducing the temperature slowly, the particles will try to position themselves in an equilibrium state
- If the cooling process were too fast, the particles would not have time to position themselves and this would cause the loss of the initial structure
- The **Metropolis algorithm** allows the simulation of this process and is shown in the pseudocode reported below.
  
## Pseudocode
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
