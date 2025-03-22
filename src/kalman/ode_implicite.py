import libreria_tesi as lib
import numpy as np
import time


def RungeKuttaEKI(f, y_prev, t_prev, dt, A, b, c):
    u0, U, y, G, ETA, Ntmax, IGamma, d, K, N = lib.dati.datiEKI(
        f, y_prev, t_prev, dt, A, b, c
    )
    u = lib.eki.EKI(u0, U, y, G, ETA, Ntmax, IGamma, d, K)[0][
        -1
    ]  # u==u_tempo_finale, return [Y, teta, res]
    um = np.mean(u, axis=0)
    um = lib.einops.rearrange(um, "(d k) -> d k", k=K, d=d)
    F = np.array([f(um[i], t_prev + c[i] * dt) for i in range(d)])
    y_next = y_prev + dt * b @ F
    return y_next


def RungeKuttaNewton(f, y_prev, t_prev, dt, A, b, c, Jf):
    s = len(c)  # Numero di stadi
    d = len(y_prev)  # Dimensione del sistema
    Y = np.zeros((s, d))  # Valori intermedi

    def F(Y, i):
        return (
            Y[i]
            - y_prev
            - dt * sum(A[i, k] * f(Y[k], t_prev + c[k] * dt) for k in range(i + 1))
        )

    for i in range(s):

        def newton(F, Y, tol=1e-10, max_iter=100):
            for iter in range(max_iter):
                J = np.eye(len(Y[i])) - dt * A[i, i] * Jf(Y[i])  # Jacobiano di F
                dy = np.linalg.solve(J, -F(Y, i))
                Y[i] += dy
                if np.linalg.norm(dy) < tol:
                    break
            return Y[i]

        Y[i] = newton(F, Y)
    y_next = y_prev + dt * sum(b[i] * f(Y[i], t_prev + c[i] * dt) for i in range(s))
    return y_next


def ode(f, y0, t0, T, dt, A, b, c, Jf):
    t = np.arange(t0, T + dt, dt)
    y_EKI = np.zeros((len(t), len(y0)))
    y_Newton = np.zeros((len(t), len(y0)))
    y_EKI[0] = y_Newton[0] = y0

    t_Newton = t_EKI = 0
    for n in range(1, len(t)):
        # print(f"{n}/{len(t)}")
        if len(t) < 100 or n % (len(t) // 100) == 0:
            print(f"{n / len(t) * 100:.2f}%")
        t1 = time.time()
        y_Newton[n] = RungeKuttaNewton(f, y_Newton[n - 1], t[n - 1], dt, A, b, c, Jf)
        t2 = time.time()
        y_EKI[n] = RungeKuttaEKI(f, y_EKI[n - 1], t[n - 1], dt, A, b, c)
        t3 = time.time()

        t_Newton += t2 - t1
        t_EKI += t3 - t2
    print(f"TEMPO TOTALE: \n\tNewton: {t_Newton} \n\tEKI: {t_EKI}")
    # f = lambda t, y: ff(y, t, case)
    # Y = solve_ivp(f, (t0, T), y0, method='Radau', t_eval=t)
    # f = lambda y, t: ff(y, t, case)
    return t, y_Newton, y_EKI


if __name__ == "__main__":
    lib.dati.comincia()
    # case = 11 # Caso f da risolvere
    # y0 = np.array([2,0]) # Dati iniziali al tempo 0 e Dimensione del sistema
    # t0, T = 0, 75 # Tempo
    # lam = np.abs(autovalori(case))
    # # Vettore dei passi temporali degli autovalori. Se l'autovalore==0, fa 1 iterazione
    # Dt = 1 / np.where(lam == 0, 1/(T-t0), lam)
    # dt = Dt[0] # Passo temporale, 1/autovalore
    # print(f"dt = {dt}")
    # metodo = 5 # EulerImplicit, Heun, Cranknicolson, Dirk22, Dirk33, TrapezoidalRule, RadauIIA3, GaussLegendre4
    # A, b, c = TableauRK(metodo)
    # #calcolaOrdine = True
    # calcolaOrdine = False #Solo per case==9
    # f = lambda y, t: ff(y, t, case)

    # f, y0, t0, T, dt, A, b, c, calcolaOrdine, case = dati.datiRK()
    # t, y_Newton, y_EKI = ode(f, y0, t0, T, dt, A, b, c)
    # print(f"norm(y_Newton - y_EKI) = {np.linalg.norm(y_Newton[-1] - y_EKI[-1])}")
    # grafico(y0, y_Newton, y_EKI, t, case, calcolaOrdine, dt, t0, T, f, A, b, c)
