import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__' : 
    
    file_path = "DatiRaccolti/LogSARC2_2_5.txt"  # replace with the correct path of your file
    df = pd.read_csv(file_path)

    plt.figure()

    plt.style.use('seaborn-v0_8-darkgrid')

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'savefig.dpi': 300,
        'lines.linewidth': 2,
    })

    # Plot di f_best
    plt.subplot(1,2,1)
    plt.plot(range(len(df.loc[:,'f_best'])), df.loc[:,'f_best'], label='f_best', color='blue', linewidth=2)

    plt.title('Andamento f_best')
    plt.xlabel('Campioni')
    plt.ylabel('f_best')
    plt.legend()
    plt.grid(axis='y',linestyle='--')


    plt.subplot(1,2,2)
    plt.plot(range(len(df.loc[:,'T'])), df.loc[:,'T'], label='T', color='orange', linewidth=2)
    plt.title('Andamento T')
    plt.xlabel('Campioni')
    plt.ylabel('T')
    plt.legend()
    plt.grid(axis='y',linestyle='--')

    
    plt.style.use('seaborn-v0_8-darkgrid')

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'savefig.dpi': 300,
        'lines.linewidth': 2,
    })

    plt.figure()
    plt.plot(range(len(df.loc[:,"BestInsertionP"])),df.loc[:,'BestInsertionP'],label='Best Insertion P',color='tab:blue',linewidth=2)
    plt.plot(range(len(df.loc[:,"Randomized2optP"])),df.loc[:,'Randomized2optP'],label='Randomized 2opt P',color='tab:green',linewidth=2)
    plt.plot(range(len(df.loc[:,"CrossExchangeP"])),df.loc[:,'CrossExchangeP'],label='Cross Exchange P',color='tab:red',linewidth=2)
    plt.title('Evoluzione delle Probabilità di Selezione delle Mosse')
    plt.ylabel('Probabilità')
    plt.xlabel('Campioni')
    plt.ylim(0, 1.05)

    plt.grid(linestyle='--',axis='y')
    plt.legend()
