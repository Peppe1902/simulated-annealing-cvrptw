# simulated-annealing-cvrptw
Simulated annealing applied to the Capacitated Vehicle Routing Problem with Time Windows


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
