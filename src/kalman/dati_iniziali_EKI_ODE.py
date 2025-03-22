import libreria_tesi as lib
import numpy as np


def datiEKI(f, y_prev, t_prev, dt, A, b, c):
    d = len(A)  # Numero di stadi
    K = len(y_prev)  # Dimensione del sistema
    N = 20  # Numero di ensemble
    Ntmax = 200  # Numero massimo di iterazioni
    # gamma = 1e-2
    gamma = dt**3
    # gamma = 1e-12 * dt**3 # No per case==10

    def G(Y):
        u = lib.einops.rearrange(Y, "n (d k) -> n d k", k=K, d=d)
        Gu = (
            u
            - y_prev[lib.np.newaxis, lib.np.newaxis, :]
            - dt * A @ f(u, t_prev + c * dt)
        )
        Gu = lib.einops.rearrange(Gu, "n d k -> n (d k)")
        return Gu

    u0 = lib.controllo_iniziale(y_prev, d, K, N, dt, A, b, c, f, t_prev, 2)
    u0 = lib.einops.rearrange(u0, "d n k -> n (d k)")
    eta = np.random.normal(0, gamma, K)  # Rumore se IGamma è diagonale
    ETA = lib.einops.repeat(eta, "k -> (rep k)", rep=d)
    IGamma = gamma**2 * np.eye(d * K)  # Matrice di covarianza del rumore
    y = ETA  # Dati osservati y=G(Y)+eta
    U = np.zeros(d)  # INUTILE
    return u0, U, y, G, ETA, Ntmax, IGamma, d, K, N


def datiRK():
    case = 10  # Caso f da risolvere
    y0 = np.array([1, 0, 0])  # Dati iniziali al tempo 0 e Dimensione del sistema
    t0, T = 0, 10  # Tempo
    lam, Jf = lib.autovalori(case)
    # Vettore dei passi temporali degli autovalori. Se l'autovalore==0, fa 1 iterazione
    Dt = 1 / np.where(lam == 0, 1e-8, np.abs(lam))
    dt = np.min(Dt)  # Passo temporale, 1/autovalore
    # dt=100
    metodo = 5  # EulerImplicit, RK4, Cranknicolson, Dirk22, Dirk33, TrapezoidalRule, RadauIIA3, GaussLegendre4
    A, b, c = lib.TableauRK(metodo)
    calcolaOrdine = False
    # calcolaOrdine = True  # Solo per case==9
    f = lambda y, t: lib.ff(y, t, case)
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
