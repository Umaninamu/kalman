import libreria_tesi as lib
import numpy as np
import time


def RungeKuttaEKI(f, y_prev, t_prev, dt, A, b, c, y_N):
    u0, U, y, G, ETA, Ntmax, IGamma, s, d, N, Nlam = lib.dati.datiEKI(
        f, y_prev, t_prev, dt, A, b, c
    )

    # COUPLED ##############################################################################
    # u = lib.eki.EKI_Coupled(u0, U, y, G, ETA, Ntmax, IGamma, s, Nlam)[0]  # u==u_tempo_finale, return [Y, teta, res]
    # um = np.mean(u, axis=0)
    ##um = lib.einops.rearrange(um, "(s d) -> s d", d=d, s=s)
    # PER STADI ############################################################################
    um = np.zeros((s, d))
    u_i = u0[:, 0]  # Partenza dal passo precedente
    # y_i = y
    # print("")
    for i in range(s):
        # Partenza da NEWTON
        u0 = lib.controllo_iniziale(y_N[i], s, d, N, dt, A, b, c[i], f, t_prev, 7)

        u_i = u0[:, i]
        y_i = y + dt * sum(A[i, k] * f(um[k], t_prev + c[k] * dt) for k in range(i))

        def G(u):
            Gu = u - dt * A[i, i] * f(
                u, t_prev + c[i] * dt
            )  # G(u)=u-dt*sum(A[i,k]*f(Y[k],t+c[k]*dt))
            return Gu

        u_i = lib.eki.EKI(u0[:, i], U, y_i, G, ETA, Ntmax, IGamma)[0][-1]
        um[i] = np.mean(u_i, axis=0)
    #######################################################################################

    F = np.array([f(um[i], t_prev + c[i] * dt) for i in range(s)])
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

    def Jacobiano(Y, i, F):
        # Calcolo lo jacobiano Jf di f con le differenze finite
        d = len(Y[i])
        Jf = np.zeros((d, d))
        h = np.eye(d) * 1e-8
        for p in range(d):
            Jf[:, p] = (F(Y + h[p], i) - F(Y - h[p], i)) / (2 * h[p][p])
        J = np.eye(d) - dt * A[i, i] * Jf  # Jacobiano di F
        return J

    for i in range(s):

        def newton(F, Y, tol=1e-10, max_iter=200):
            for iter in range(max_iter):
                # Calcolo lo jacobiano Jf di f con le differenze finite
                # J = Jacobiano(Y, i, F)
                # Calcolo lo jacobiano Jf di f a mano
                J = np.eye(len(Y[i])) - dt * A[i, i] * Jf(Y[i])  # Jacobiano di F
                dy = np.linalg.solve(J, -F(Y, i))
                Y[i] += dy
                if np.linalg.norm(dy) < tol:
                    # print("Newton converge in ", iter, "iterazioni")
                    break
                elif iter == max_iter - 1:
                    print("Newton-Non converge in ", i, "iterazioni")
                    # raise ValueError("Newton non converge")
            return Y[i]

        Y[i] = newton(F, Y)
    y_next = y_prev + dt * sum(b[i] * f(Y[i], t_prev + c[i] * dt) for i in range(s))
    return y_next, Y


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
        y_Newton[n], Y = RungeKuttaNewton(f, y_Newton[n - 1], t[n - 1], dt, A, b, c, Jf)
        t2 = time.time()
        y_EKI[n] = RungeKuttaEKI(f, y_EKI[n - 1], t[n - 1], dt, A, b, c, Y)
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
