import numpy as np
import matplotlib.pyplot as plt
import EKI as eki
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import sympy as sp
import time
import einops
import dati_iniziali_EKI_ODE as dati

def RungeKuttaEKI(f, y_prev, t_prev, dt, A, b, c):
    u0, U, y, G, ETA, Ntmax, IGamma, d, K, N = dati.datiEKI(
        f, y_prev, t_prev, dt, A, b, c
    )
    u = eki.EKI(u0, U, y, G, ETA, Ntmax, IGamma)[0][
        -1
    ]  # u==u_tempo_finale, return [Y, teta, res]
    um = np.mean(u, axis=0)
    um = einops.rearrange(um, "(d k) -> d k", k=K, d=d)
    F = np.array([f(um[i], t_prev + c[i] * dt) for i in range(d)])
    y_next = y_prev + dt * b @ F
    return y_next


def RungeKuttaNewton(f, y_prev, t_prev, dt, A, b, c):
    def newton(F, y_guess, tol=1e-8, max_iter=100):
        """
        Metodo di Newton per risolvere F(y) = 0
        F: funzione da risolvere (da RK), dim = n
        y_guess: guess iniziale, dim = n
        """
        for nn in range(max_iter):
            J = jacobiano(F, y_guess)
            # J = jacobiano_sym(F, y_guess)
            dy = np.linalg.solve(J, -F(y_guess))

            y_guess += dy
            # print(f'{np.linalg.norm(dy)} -- {tol}')
            if np.linalg.norm(dy) < tol:
                #print("Newton-Converge in ", nn, "iterazioni")
                break
        return y_guess

    # def jacobiano(F, y, eps=1e-8): # Ord 1
    #     n = len(y)
    #     y_eps = np.tile(y, (n, 1)) + eps * np.eye(n)
    #     F0 = F(y)
    #     J = (np.array(list(map(F, y_eps))) - F0[np.newaxis, :]) / eps
    #     return J
    def jacobiano(F, y, eps=1e-8):  # Ord 2
        n = len(y)
        h = eps * np.eye(n)
        J = np.zeros((n, n))
        for i in range(n):
            J[i] = (F(y + h[i]) - F(y - h[i])) / (2 * eps)
        return J.T

    def jacobiano_sym(F, y):  # Esatto ma lento
        y_sym = np.array(sp.symbols("y0:%d" % len(y)))
        F_sym = sp.Matrix(F(y_sym))
        J_sym = F_sym.jacobian(y_sym)
        J = sp.lambdify(y_sym, J_sym, "numpy")(*y)
        return J

    def F(Y):
        Y = Y.reshape((len(A), len(y_prev)))
        K = np.zeros_like(Y)
        for i in range(len(A)):
            K[i] = f(y_prev + dt * np.dot(A[i], Y), t_prev + c[i] * dt)
        return (Y - K).flatten()

    Y_guess = np.zeros((len(A), len(y_prev))).flatten()
    Y = newton(F, Y_guess).reshape((len(A), len(y_prev)))
    y_next = y_prev + dt * np.dot(b, Y)
    return y_next


def ode(f, y0, t0, T, dt, A, b, c):
    t = np.arange(t0, T + dt, dt)
    y_EKI = np.zeros((len(t), len(y0)))
    y_Newton = np.zeros((len(t), len(y0)))
    y_EKI[0] = y_Newton[0] = y0

    t_Newton = 0
    t_EKI = 0
    for n in range(1, len(t)):
        # print(f"{n}/{len(t)}")
        if len(t) < 100 or n % (len(t) // 100) == 0:
            print(f"{n / len(t) * 100:.2f}%")
        t1 = time.time()
        try:
            y_Newton[n] = RungeKuttaNewton(f, y_Newton[n - 1], t[n - 1], dt, A, b, c)
        except:
            print("Newton ha dato ERRORE all'iterazione:", n, "/", len(t))
            raise ()
        t2 = time.time()
        RungeKuttaEKI(f, y_EKI[n-1], t[n-1], dt, A, b, c)
        try:
            y_EKI[n] = RungeKuttaEKI(f, y_EKI[n - 1], t[n - 1], dt, A, b, c)
        except:
            print("EKI ha dato ERRORE all'iterazione:", n, "/", len(t))
            raise ()
        t3 = time.time()
        t_Newton += t2 - t1
        t_EKI += t3 - t2
    print(f"TEMPO TOTALE: \n\tNewton: {t_Newton} \n\tEKI: {t_EKI}")
    # f = lambda t, y: ff(y, t, case)
    # Y = solve_ivp(f, (t0, T), y0, method='Radau', t_eval=t)
    # f = lambda y, t: ff(y, t, case)
    return t, y_Newton, y_EKI


def TableauRK(metodo):
    if metodo == "EulerImplicit" or metodo == 1:
        A = np.array([[1]])
        b = np.array([1])
        c = np.array([1])
    elif metodo == "RK4" or metodo == 2:
        A = np.array([[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]])
        b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
        c = np.array([0, 1 / 2, 1 / 2, 1])
        # HEUN
        # A = np.array([[1e-8, 0],
        #               [1, 1e-8]])
        # b = np.array([1/2, 1/2])
        # c = np.array([1e-8, 1])
    elif metodo == "CrankNicolson" or metodo == 3:
        A = np.array([[0, 0], [1 / 2, 1 / 2]])
        b = np.array([1 / 2, 1 / 2])
        c = np.array([0, 1])
    elif metodo == "Dirk22" or metodo == 4:
        l = 1 - np.sqrt(2) / 2
        A = np.array([[l, 0], [1 - 2 * l, l]])
        b = np.array([1 / 2, 1 / 2])
        c = np.array([l, 1 - l])
    elif metodo == "Dirk33" or metodo == 5:
        l = 0.4358665215
        A = np.array(
            [
                [l, 0, 0],
                [(1 - l) / 2, l, 0],
                [-3 / 2 * l**2 + 4 * l - 1 / 4, 3 / 2 * l**2 - 5 * l + 5 / 4, l],
            ]
        )
        b = np.array([-3 / 2 * l**2 + 4 * l - 1 / 4, 3 / 2 * l**2 - 5 * l + 5 / 4, l])
        c = np.array([l, (1 + l) / 2, 1])
    elif metodo == "TrapezoidalRule" or metodo == 6:
        A = np.array([[1 / 2, 0], [1 / 2, 1 / 2]])
        b = np.array([1 / 2, 1 / 2])
        c = np.array([1 / 2, 1])
    elif metodo == "RadauIIA3" or metodo == 7:
        A = np.array([[5 / 12, -1 / 12], [3 / 4, 1 / 4]])
        b = np.array([3 / 4, 1 / 4])
        c = np.array([1 / 3, 1])
    elif metodo == "GaussLegendre4" or metodo == 8:
        A = np.array([[1 / 4, 1 / 4 - np.sqrt(3) / 6], [1 / 4 + np.sqrt(3) / 6, 1 / 4]])
        b = np.array([1 / 2, 1 / 2])
        c = np.array([1 / 2 - np.sqrt(3) / 6, 1 / 2 + np.sqrt(3) / 6])
    else:
        raise ValueError("Metodo non riconosciuto")
    return A, b, c


# Dati
def ff(y, t, case):
    if case == 0:  # y' = y^2
        Y = np.array(y) ** 2
    elif case == 1:  # y'' + y = 0
        Y = np.array([y[1], -y[0]])
    elif case == 2:  # y'' + t = 0
        Y = np.array([y[1], -t])
    elif case == 3:  # y'' + y'- t = 0
        Y = np.array([y[1], -y[1] + t])
    elif case == 4:  # y''' + y'' + y' + y = 0
        Y = np.array([y[1], y[2], -y[2] - y[1] - y[0]])
    elif case == 5:  # y''' + y'' + y' + y = 1
        Y = np.array([y[1], y[2], -y[2] - y[1] - y[0] + 1])
    elif (
        case == 6
    ):  # y^(10) + y^(9) + y^(8) + y^(7) + y^(6) + y^(5) + y^(4) + y^(3) + y^(2) + y + 1 = 0
        Y = np.array(
            [
                y[1],
                y[2],
                y[3],
                y[4],
                y[5],
                y[6],
                y[7],
                y[8],
                y[9],
                -y[9]
                - y[8]
                - y[7]
                - y[6]
                - y[5]
                - y[4]
                - y[3]
                - y[2]
                - y[1]
                - y[0]
                - 1,
            ]
        )
    elif case == 7:  # y' = t-1/t * y/1-2y
        Y = np.array([(t - 1) / t * y[0] / (1 - 2 * y[0])])
    elif case == 8:  # y' = 3*x/y + y/x
        Y = np.array([3 * t / y[0] + y[0] / t])
    # STIFF
    elif case == 9:  # Test y' = Ay, Separabile
        # Calcolo Jacobiano
        J = np.array([[-1, 0.01], 
                      [0.01, -16]])
        if len(y.shape) == 1:  # y=y1,y2. Per Newton
            Y = J @ y
        else:  # y: N x d x K. Per EKI
            Y = np.einsum("ij,ghj->ghi", J, y)
    elif case == 10:  # Robertson Chemical Reaction, Separabile
        # T=1e11, y0=1,0,0, Nt=1e3
        if len(y.shape) == 1:  # y=y1,y2,y3. Per Newton
            Y = np.array(
                [
                    -ma * y[0] + mb * y[1] * y[2],
                    ma * y[0] - mb * y[1] * y[2] - mc * y[1] ** 2,
                    mc * y[1] ** 2,
                ]
            )
        else:  # y: N x d x K. Per EKI
            Y = np.array(
                [
                    -ma * y[:, :, 0] + mb * y[:, :, 1] * y[:, :, 2],
                    ma * y[:, :, 0]
                    - mb * y[:, :, 1] * y[:, :, 2]
                    - mc * y[:, :, 1] ** 2,
                    mc * y[:, :, 1] ** 2,
                ]
            ).transpose(1, 2, 0)
    elif case == 11:  # Electrical Circuits / Van der Pol
        # T=75, y0=2,0
        if len(y.shape) == 1:  # y=y1,y2. Per Newton
            Y = np.array([y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]])
        else:  # y: N x d x K. Per EKI
            Y = np.array(
                [y[:, :, 1], mu * (1 - y[:, :, 0] ** 2) * y[:, :, 1] - y[:, :, 0]]
            ).transpose(1, 2, 0)
    elif case == 12:  # Oregonator
        # T=360, y0=1,2,3, Nt=1e4
        if len(y.shape) == 1:  # y=y1,y2,y3. Per Newton
            Y = np.array(
                [
                    77.27 * (y[1] + y[0] * (1 - 8.375e-6 * y[0] - y[1])),
                    1 / 77.27 * (y[2] - (1 + y[0]) * y[1]),
                    0.161 * (y[0] - y[2]),
                ]
            )
        else:  # y: N x d x K. Per EKI
            Y = np.array(
                [
                    77.27
                    * (
                        y[:, :, 1]
                        + y[:, :, 0] * (1 - 8.375e-6 * y[:, :, 0] - y[:, :, 1])
                    ),
                    1 / 77.27 * (y[:, :, 2] - (1 + y[:, :, 0]) * y[:, :, 1]),
                    0.161 * (y[:, :, 0] - y[:, :, 2]),
                ]
            ).transpose(1, 2, 0)
    elif case == 13: #Caso Affine y' = Ay + f(t), Separabile
        # T=20, y0=9.9,0
        if len(y.shape) == 1:
            Y  = np.array([-41*y[0] + 59*y[1] + -2*t**3 * (t**2 - 50*t - 2) * np.exp(-t**2),
                        40*y[0] - 60*y[1] + 2*t**3 * (t**2 - 50*t - 2) * np.exp(-t**2)])
        else: # y: N x d x K. Per EKI
            Y = np.array([-41*y[:, :, 0] + 59*y[:, :, 1] + -2*t**3 * (t**2 - 50*t - 2) * np.exp(-t**2),
                        40*y[:, :, 0] - 60*y[:, :, 1] + 2*t**3 * (t**2 - 50*t - 2) * np.exp(-t**2)]
                         ).transpose(1, 2, 0)
    elif case == 14: #Preda-Predatore
        # T=15, y0=1,0.1
        if len(y.shape) == 1:
            Y = np.array([ma*y[0] - mb*y[0]*y[1],
                          mc*y[0]*y[1] - md*y[1]])
        else:
            Y = np.array([ma*y[:, :, 0] - mb*y[:, :, 0]*y[:, :, 1],
                          mc*y[:, :, 0]*y[:, :, 1] - md*y[:, :, 1]]
                         ).transpose(1, 2, 0)
    return Y


# Grafico
def grafico(y0, y_Newton, y_EKI, t, case, calcolaOrdine, dt, t0, T, f, A, b, c):
    plt.figure()
    for i in range(len(y0)):
        plt.plot(t, y_Newton[:, i], ".-", label=f"y{i+1} Newton")
        plt.plot(t, y_EKI[:, i], "--", label=f"y{i+1} EKI")

    if case == 9:  # Sol esatta per caso test
        M = np.array([[-1, 0.01], [0.01, -16]])
        # Calcolo Autovalori
        l1 = 0.5 * (
            M[0, 0]
            + M[1, 1]
            + np.sqrt((M[0, 0] - M[1, 1]) ** 2 + 4 * M[0, 1] * M[1, 0])
        )
        l2 = 0.5 * (
            M[0, 0]
            + M[1, 1]
            - np.sqrt((M[0, 0] - M[1, 1]) ** 2 + 4 * M[0, 1] * M[1, 0])
        )
        # print(f"Autovalori: {l1}, {l2}")
        # Calcolo Autovettori
        v1 = np.array([1, (l1 - M[0, 0]) / M[0, 1]])
        v2 = np.array([1, (l2 - M[0, 0]) / M[0, 1]])
        # Calcolo Costanti
        c1 = (y0[1] * v2[0] - y0[0] * v2[1]) / (v1[1] * v2[0] - v1[0] * v2[1])
        c2 = (y0[0] - c1 * v1[0]) / v2[0]

        # (l1, l2), (v1, v2) = np.linalg.eig(M) #v2 NON VA BENE! Deve essere v2 = (1/v22, 1)!!!
        # (c1, c2) = np.linalg.solve(np.array([v1, v2]).T, y0)

        # Calcolo Soluzione
        tt = np.linspace(t0, T)
        SOL = lambda t: (
            c1 * np.exp(l1 * t) * v1[:, np.newaxis]
            + c2 * np.exp(l2 * t) * v2[:, np.newaxis]
        ).T
        sol = SOL(tt)
        plt.plot(tt, sol, label="Soluzione esatta")
    plt.legend()
    plt.xlabel("Tempo")
    plt.ylabel("Valori")
    plt.title("Soluzione del sistema di ODE")

    for i in range(len(y0)):
        plt.figure()
        plt.plot(t, y_Newton[:, i], ".-", label=f"y{i+1} Newton")
        plt.plot(t, y_EKI[:, i], "--", label=f"y{i+1} EKI")
        plt.legend()
        plt.xlabel("Tempo")
        plt.ylabel("Valori")
        plt.title(f"Soluzione del sistema di ODE per y{i+1}")
    
    #Grafico spazio delle fasi
    if case in [9, 11, 13, 14]: #Casi in dimensione 2
        plt.figure()
        plt.plot(y_Newton[:,0], y_Newton[:,1], '.-', label="Newton")
        plt.plot(y_EKI[:,0], y_EKI[:,1], '--', label="EKI")
        plt.legend()
        plt.xlabel("y1")
        plt.ylabel("y2")
        plt.title("Spazio delle fasi")

    # Verifica convergenza
    if calcolaOrdine and case == 9:
        Dt = [dt / 2**i for i in range(5)]
        NT = np.zeros(len(Dt))
        errore_Newton = np.zeros((len(Dt), 2))
        errore_EKI = np.zeros((len(Dt), 2))
        for i, dt in enumerate(Dt):
            print(f"i, dt = {i}, {dt}")
            Nt = int((T - t0) / dt)
            NT[i] = Nt
            T = t0 + Nt * dt
            t, y_Newton, y_EKI = ode(f, y0, t0, T, dt, A, b, c)
            errore_Newton[i] = np.abs(y_Newton[-1] - SOL(t[-1]))
            errore_EKI[i] = np.abs(y_EKI[-1] - SOL(t[-1]))
        # Calcolo ordine p
        for i in range(len(Dt) - 1):
            p_Newton = np.log(errore_Newton[i] / errore_Newton[i + 1]) / np.log(2)
            print(f"Ordine di convergenza per Newton: {p_Newton}")
            p_EKI = np.log(errore_EKI[i] / errore_EKI[i + 1]) / np.log(2)
            print(f"Ordine di convergenza per EKI: {p_EKI}")

        plt.figure()
        # plt.loglog(NT, errore, 'x--', NT, 1/NT, 'o-', label='Ordine')
        plt.loglog(Dt, errore_Newton, "o--", label="Errore Newton")
        plt.loglog(Dt, errore_EKI, "x--", label="Errore EKI")
        plt.legend()
        plt.xlabel("Dt")
        plt.ylabel("Errore")
        plt.title("Ordine di convergenza")
    plt.show()


def autovalori(case):
    global ma, mb, mc, md, mu
    if case == 9:
        J = np.array([[-1, 0.01], [0.01, -16]])
    elif case == 10:
        ma = 0.04
        mb = 1e2
        mc = 3e3
        Y = np.array([0, 0, 2])  # Punto di equilibrio.
        J = np.array(
            [
                [-ma, mb * Y[2], mb * Y[1]],
                [ma, -mb * Y[2] - 2*mc * Y[1], -mb * Y[1]],
                [0, 2*mc * Y[1], 0],
            ]
        )
    elif case == 11:
        mu = 25
        Y = np.array([0, 0])  # Punto di equilibrio.
        J = np.array([[0, 1], [-2 * mu * Y[0] * Y[1] - 1, mu * (1 - Y[0] ** 2)]])
    elif case == 12:
        q = (8.375e-6 + np.sqrt((8.375e-6) ** 2 + 8e-6)) / -2e-6
        (8.375e-6 - np.sqrt((8.375e-6) ** 2 + 8e-6)) / -2e-6
        Y = np.array([q, q / (1 + q), q])  # Punto di equilibrio.
        # Y = np.array([p, p / (1 + p), p]) #Punto di equilibrio.
        # Y = np.array([0, 0, 0]) #Punto di equilibrio.
        J = np.array(
            [
                [77.27 * (1 - 2 * 8.375e-6 * Y[0] - Y[1]), 77.27 * (1 - Y[0]), 0],
                [-1 / 77.27 * Y[1], -1 / 77.27 * (1 + Y[0]), 1 / 77.27],
                [0.161, 0, -0.161],
            ]
        )
    elif case == 13:
        J = np.array([[-41, 59],
                      [40, -60]])
    elif case == 14:
        ma, mb, mc, md = 100, 1, 1, 1
        Y = np.array([0, 0]) # Punto di equilibrio.
        #Y = np.array([md / mc, ma / mb]) # Punto di equilibrio.
        J = np.array([[ma-mb*Y[1], -mb*Y[0]],
                        [mc*Y[1], mc*Y[0] - md]])
    lam = np.linalg.eigvals(J)
    print(f"Autovalori: {lam}")
    return lam


if __name__ == "__main__":
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
    dati.comincia()
