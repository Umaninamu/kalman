import libreria_tesi as lib
import numpy as np


def datiEKI(f, y_prev, t_prev, dt, A, b, c):
    s = len(A)  # Numero di stadi
    d = len(y_prev)  # Dimensione del sistema
    N = 15  # Numero di ensemble
    Ntmax = 200  # Numero massimo di iterazioni
    # gamma = 1e-4 * dt**s
    gamma = dt**s
    # gamma = 1e-12 * dt**s # No per case==10

    alpha, beta = 0.7, -1  # Parametri di EKI accelerato
    eta = np.random.normal(0, gamma, d)  # Rumore se IGamma è diagonale
    IGamma = gamma**2 * np.eye(d)  # Matrice di covarianza del rumore
    y = y_prev + eta  # Dati osservati y=G(Y)+eta
    return y, eta, Ntmax, IGamma, s, d, N, alpha, beta


def datiRK():
    case = 9  # Caso f da risolvere
    y0 = np.array([1, 2])  # Dati iniziali al tempo 0 e Dimensione del sistema
    t0, T = 0, 30  # Tempo
    lam, Jf, coefficienti_m = lib.autovalori(case)
    # Vettore dei passi temporali degli autovalori. Se l'autovalore==0, fa 1 iterazione
    Dt = 1 / np.where(lam == 0, 1e-8, np.abs(lam))
    dt = np.min(Dt)  # Passo temporale, 1/autovalore
    # dt=0.02
    metodo = 5  # EulerImplicit, RK4, Cranknicolson, Dirk22, Dirk33, TrapezoidalRule, RadauIIA3, GaussLegendre4
    A, b, c = lib.TableauRK(metodo)
    calcolaOrdine = False
    # calcolaOrdine = True  # Solo per case==9
    f = lambda y, t: lib.ff(y, t, case, coefficienti_m)
    return f, y0, t0, T, dt, A, b, c, calcolaOrdine, case, Jf


def comincia():
    f, y0, t0, T, dt, A, b, c, calcolaOrdine, case, Jf = datiRK()
    t, y_Newton, y_EKI = lib.odei.ode(f, y0, t0, T, dt, A, b, c, Jf)
    # STAMPA
    print("")
    lib.autovalori(case)[0]  # stampa autovalori
    print(f"dt = {dt}, autovalore usato = {1/dt}")
    print(f"norm(y_Newton - y_EKI) = {np.linalg.norm(y_Newton[-1] - y_EKI[-1])}")
    for i in range(len(y_Newton[-1])):
        print(f"Componente {i}: {np.linalg.norm(y_Newton[-1][i] - y_EKI[-1][i])}")
    # GRAFICO
    lib.grafico_ode(
        y0, y_Newton, y_EKI, t, case, calcolaOrdine, dt, t0, T, f, A, b, c, Jf
    )


if __name__ == "__main__":
    comincia()

    # lib.my_line_profiler(comincia)
    # lib.my_pstats_profiler(comincia)
    # lib.cProfile.run('comincia()')
