import numpy as np
import einops
import ode_implicite as odei
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import EKI as eki


def datiEKI(f, y_prev, t_prev, dt, A, b, c):
    d = len(A)  # Numero di stadi
    K = len(y_prev)  # Dimensione del sistema
    N = 20  # Numero di ensemble
    Ntmax = 200  # Numero massimo di iterazioni
    gamma = 1e-2
    # gamma = dt**3
    # gamma = 1e-12 * dt**3 # No per case==10

    def G(Y):
        u = einops.rearrange(Y, "n (d k) -> n d k", k=K, d=d)
        Gu = u - y_prev[np.newaxis, np.newaxis, :] - dt * A @ f(u, t_prev + c * dt)
        Gu = einops.rearrange(Gu, "n d k -> n (d k)")
        # Gu = np.zeros_like(Y)
        # for j in range(N):
        #     u = einops.rearrange(Y[j], '(d k) -> d k', k=K, d=d)
        #     Kk = np.zeros_like(u)
        #     for i in range(d):
        #         #Kk[i] = f(y_prev + dt * A[i] @ u, t_prev + c[i] * dt)
        #         Kk[i] = u[i] - y_prev - dt * sum(A[i, l] * f(u[l], t_prev + c[l] * dt) for l in range(d))
        #     Gu[j] = einops.rearrange(Kk, 'd k -> (d k)')
        return Gu

    u0 = controllo_iniziale(y_prev, d, K, N, dt, A, b, c, f, t_prev)
    u0 = einops.rearrange(u0, "d n k -> n (d k)")
    eta = np.random.normal(0, gamma, K)  # Rumore se IGamma è diagonale
    ETA = einops.repeat(eta, "k -> (rep k)", rep=d)
    IGamma = gamma**2 * np.eye(d * K)  # Matrice di covarianza del rumore
    y = ETA  # Dati osservati y=G(Y)+eta
    U = np.zeros(d)  # INUTILE
    return u0, U, y, G, ETA, Ntmax, IGamma, d, K, N


def datiRK():
    case = 13  # Caso f da risolvere
    y0 = np.array([1, 1])  # Dati iniziali al tempo 0 e Dimensione del sistema
    t0, T = 0, 100  # Tempo
    lam, Jf = odei.autovalori(case)
    # Vettore dei passi temporali degli autovalori. Se l'autovalore==0, fa 1 iterazione
    Dt = 1 / np.where(lam == 0, 1e-8, np.abs(lam))
    dt = np.max(Dt)  # Passo temporale, 1/autovalore
    # dt=100
    metodo = 5  # EulerImplicit, RK4, Cranknicolson, Dirk22, Dirk33, TrapezoidalRule, RadauIIA3, GaussLegendre4
    A, b, c = odei.TableauRK(metodo)
    calcolaOrdine = False
    # calcolaOrdine = True  # Solo per case==9
    f = lambda y, t: odei.ff(y, t, case)
    return f, y0, t0, T, dt, A, b, c, calcolaOrdine, case, Jf


def controllo_iniziale(y, d, K, N, dt, A, b, c, f, t, contr=2):
    if contr == 1:
        """
        1) Controlli iniziali uguali ma traslati
        Funziona discretamente per case == 9, 11: tende sempre alla convergenza ma la raggiunge raramente
        Funziona male per case == 10: non converge o (CGG + IGamma) non è invertibile
        Funziona male per case == 12: se T grande (50) non vede i massimi.
        """
        u0 = np.array(
            [[y + 0.25 + np.random.randn() for _ in range(N)] for _ in range(d)]
        )
    elif contr == 2:
        """
        2) Controlli iniziali a media y_prev e deviazione standard 1
        Funziona bene per case == 9 : Migliore
        Funziona male per case == 10: in una iterazione si perde la convergenza
        Funziona bene per case == 11: più lento del 5
        Funziona male per case == 12: se T grande (50) non vede i massimi
        """
        u0 = np.random.normal(loc=y, scale=2 * dt, size=(d, N, K))
    elif contr == 3:
        """
        3) Controlli iniziali a media y_prev e deviazione standard > c_k-c_(k-1) * dt
        Funziona bene per case == 9
        Funziona male per case == 10: per Ntmax == 200 converge a sol sbagliata
        Per case == 11 non vede bene i massimi (Ntmax = 10000)
        Per case == 12 se T grande (50) non vede bene i massimi
        """
        u0 = np.random.normal(
            loc=y, scale=dt * np.max(np.diff(c)) * 1.5, size=(d, N, K)
        )
    elif contr == 4:
        """
        4) Controlli iniziali a media y_next (RK implicito) e deviazione standard 1
        Funziona bene per case == 9, 11: più lento del 2
        Funziona male per case == 10: non converge o (CGG + IGamma) non è invertibile
        Funziona bene per case == 12: meglio del 2
        """
        u0 = np.random.normal(
            loc=odei.RungeKuttaNewton(f, y, t, dt, A, b, c), scale=1, size=(d, N, K)
        )
    elif contr == 5:
        """
        5) Controlli iniziali a media y_next (RK 4 esplicito) e deviazione standard 1
        Funziona bene per case == 9
        Funziona male per case == 10: (CGG + IGamma) non è invertibile
        Funziona bene per case == 11: più veloce del 2
        Funziona bene per case == 12: (CGG + IGamma) non è invertibile
        """
        DT = dt / 1
        tab = odei.TableauRK(2)
        y_next = y.copy()
        k = np.zeros((len(tab[1]), len(y)))  # Initialize k array
        for i in range(len(tab[1])):
            k[i] = f(y_next + DT * np.dot(tab[0][i, :], k), t + tab[2][i] * DT)
        y_next += DT * np.dot(tab[1], k)
        u0 = np.random.normal(loc=y_next, scale=dt, size=(d, N, K))
    elif contr == 6:
        """ """
        u0 = np.random.normal(loc=y_next, scale=dt, size=(d, N, K))

    return u0


def comincia():
    f, y0, t0, T, dt, A, b, c, calcolaOrdine, case, Jf = datiRK()
    t, y_Newton, y_EKI = odei.ode(f, y0, t0, T, dt, A, b, c, Jf)
    # STAMPA
    print("")
    odei.autovalori(case)[0]  # stampa autovalori
    print(f"dt = {dt}, autovalore usato = {1/dt}")
    print(f"norm(y_Newton - y_EKI) = {np.linalg.norm(y_Newton[-1] - y_EKI[-1])}")
    for i in range(len(y_Newton[-1])):
        print(f"Componente {i}: {np.linalg.norm(y_Newton[-1][i] - y_EKI[-1][i])}")
    # GRAFICO
    odei.grafico(y0, y_Newton, y_EKI, t, case, calcolaOrdine, dt, t0, T, f, A, b, c, Jf)


if __name__ == "__main__":
    comincia()
