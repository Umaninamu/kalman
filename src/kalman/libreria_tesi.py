"""
Lista funzioni disponibili:
- my_line_profiler(funzione_da_testare, *args)
- my_pstats_profiler(funzione_da_testare, *args)
- TableauRK(metodo)
- autovalori(case)
- controllo_iniziale(y, d, K, N, dt, A, b, c, f, t, contr=2)
- grafico_ode(y0, y_Newton, y_EKI, t, case, calcolaOrdine, dt, t0, T, f, A, b, c, Jf)
- ff(y, t, case)
"""

import numpy as np
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt
from scipy.linalg import cho_solve, cho_factor
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.sparse import block_diag
import einops
from scipy.stats import gaussian_kde
import EKI_coupled as eki
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import sympy as sp
import dati_iniziali_EKI_ODE as dati
import ode_implicite as odei
from line_profiler import LineProfiler
import cProfile, pstats


############################################################################################
############################################################################################
############################################################################################
def my_line_profiler(funzione_da_testare, *args):
    profiler = LineProfiler()
    profiler.add_function(funzione_da_testare)
    profiler.enable()
    output = funzione_da_testare(*args)
    profiler.disable()
    profiler.print_stats()
    return output


def my_pstats_profiler(funzione_da_testare, *args):
    profiler = cProfile.Profile()
    profiler.enable()
    output = funzione_da_testare(*args)
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative").print_stats(20)
    return output


############################################################################################
############################################################################################
############################################################################################
def TableauRK(metodo):
    # ode_implicite.py
    if metodo == "EulerImplicit" or metodo == 1:
        A = np.array([[1]])
        b = np.array([1])
        c = np.array([1])
    elif metodo == "RK4" or metodo == 2:
        A = np.array([[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]])
        b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
        c = np.array([0, 1 / 2, 1 / 2, 1])
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


############################################################################################
############################################################################################
############################################################################################
def autovalori(case):
    # ode_implicite.py
    ma, mb, mc, md, mu = 0, 0, 0, 0, 0
    if case == 9:
        Y = np.array([1, 1])  # NON SERVE
        J = lambda Y: np.array([[-1, 0.01], [0.01, -16]])
    elif case == 10:
        ma = 0.04
        mb = 1e2
        mc = 3e3
        Y = np.array([0, 0, 2])  # Punto di equilibrio.
        J = lambda Y: np.array(
            [
                [-ma, mb * Y[2], mb * Y[1]],
                [ma, -mb * Y[2] - 2 * mc * Y[1], -mb * Y[1]],
                [0, 2 * mc * Y[1], 0],
            ]
        )
    elif case == 11:
        mu = 25
        Y = np.array([0, 0])  # Punto di equilibrio.
        J = lambda Y: np.array(
            [[0, 1], [-2 * mu * Y[0] * Y[1] - 1, mu * (1 - Y[0] ** 2)]]
        )
    elif case == 12:
        q = (8.375e-6 + np.sqrt((8.375e-6) ** 2 + 8e-6)) / -2e-6
        (8.375e-6 - np.sqrt((8.375e-6) ** 2 + 8e-6)) / -2e-6
        Y = np.array([q, q / (1 + q), q])  # Punto di equilibrio.
        # Y = np.array([p, p / (1 + p), p]) #Punto di equilibrio.
        # Y = np.array([0, 0, 0]) #Punto di equilibrio.
        J = lambda Y: np.array(
            [
                [77.27 * (1 - 2 * 8.375e-6 * Y[0] - Y[1]), 77.27 * (1 - Y[0]), 0],
                [-1 / 77.27 * Y[1], -1 / 77.27 * (1 + Y[0]), 1 / 77.27],
                [0.161, 0, -0.161],
            ]
        )
    elif case == 13:
        Y = np.array([0, 0])  # NON SERVE
        J = lambda Y: np.array([[-41, 59], [40, -60]])
    elif case == 14:
        ma, mb, mc, md = 10, 1, 1, 1
        Y = np.array([0, 0])  # Punto di equilibrio.
        # Y = np.array([md / mc, ma / mb]) # Punto di equilibrio.
        J = lambda Y: np.array(
            [[ma - mb * Y[1], -mb * Y[0]], [mc * Y[1], mc * Y[0] - md]]
        )
    elif case == 15:
        mu = 100
        Y = np.array([1])  # Punto di equilibrio.
        J = lambda Y: np.array([-2 * mu * Y + 1])  # Derivata della funzione f
    lam = np.linalg.eigvals(J(Y))
    print(f"Autovalori: {lam}")
    return lam, J, (ma, mb, mc, md, mu)


############################################################################################
############################################################################################
############################################################################################
def controllo_iniziale(y, d, K, N, dt, A, b, c, f, t, contr=2):
    # dati_iniziali_EKI_ODE.py
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
        u0 = np.random.normal(loc=y, scale=dt, size=(N, d, K))
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
        """
        6) Controlli iniziali deterministici uniformi
        """
        u0 = np.random.uniform(low=y - 0.5, high=y + 0.5, size=(N, d, K))
    elif contr == 7:
        u0 = np.random.normal(loc=y, scale=c * dt, size=(N, K))

    return u0


############################################################################################
############################################################################################
############################################################################################
def grafico_ode(y0, y_Newton, y_EKI, t, case, calcolaOrdine, dt, t0, T, f, A, b, c, Jf):
    # ode_implicite.py
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
        # Calcolo Autovettori
        v1 = np.array([1, (l1 - M[0, 0]) / M[0, 1]])
        v2 = np.array([1, (l2 - M[0, 0]) / M[0, 1]])
        # Calcolo Costanti
        c1 = (y0[1] * v2[0] - y0[0] * v2[1]) / (v1[1] * v2[0] - v1[0] * v2[1])
        c2 = (y0[0] - c1 * v1[0]) / v2[0]

        # (l1, l2), (v1, v2) = np.linalg.eig(M) #v2 NON VA BENE! Deve essere v2 = (1/v22, 1)!!!
        # (c1, c2) = np.linalg.solve(np.array([v1, v2]).T, y0)

        # Calcolo Soluzione
        tt = np.linspace(t0, T, 500)
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
        if case == 9:
            plt.plot(tt, sol[:, i], label=f"y{i+1} Soluzione esatta")
        plt.legend()
        plt.xlabel("Tempo")
        plt.ylabel("Valori")
        plt.title(f"Soluzione del sistema di ODE per y{i+1}")

    # Grafico spazio delle fasi
    if case in [9, 11, 13, 14]:  # Casi in dimensione 2
        plt.figure()
        plt.plot(y_Newton[:, 0], y_Newton[:, 1], ".-", label="Newton")
        plt.plot(y_EKI[:, 0], y_EKI[:, 1], "--", label="EKI")
        if case == 9:
            plt.plot(sol[:, 0], sol[:, 1], label="Soluzione esatta")
        plt.legend()
        plt.xlabel("y1")
        plt.ylabel("y2")
        plt.title("Spazio delle fasi")

    # Verifica convergenza
    if calcolaOrdine and case == 9:
        Dt = [dt / 2**i for i in range(6)]
        NT = np.zeros(len(Dt))
        errore_Newton = np.zeros((len(Dt), 2))
        errore_EKI = np.zeros((len(Dt), 2))
        for i, dt in enumerate(Dt):
            print(f"i, dt = {i}, {dt}")
            Nt = int((T - t0) / dt)
            NT[i] = Nt
            T = t0 + Nt * dt
            t, y_Newton, y_EKI = odei.ode(f, y0, t0, T, dt, A, b, c, Jf)
            errore_Newton[i] = np.abs(y_Newton[-1] - SOL(t[-1]))
            errore_EKI[i] = np.abs(y_EKI[-1] - SOL(t[-1]))
        # Calcolo ordine p
        for i in range(len(Dt) - 1):
            p_Newton = np.log(errore_Newton[i] / errore_Newton[i + 1]) / np.log(2)
            print(f"Ordine di convergenza per Newton: {p_Newton}")
            p_EKI = np.log(errore_EKI[i] / errore_EKI[i + 1]) / np.log(2)
            print(f"Ordine di convergenza per EKI: {p_EKI}")

        plt.figure()
        plt.loglog(Dt, errore_Newton, "o--", label="Errore Newton")
        plt.loglog(Dt, errore_EKI, "x--", label="Errore EKI")
        plt.legend()
        plt.xlabel("Dt")
        plt.ylabel("Errore")
        plt.title("Ordine di convergenza")
    plt.show()


############################################################################################
############################################################################################
############################################################################################
def ff(y, t, case, coefficienti_m):
    ma, mb, mc, md, mu = coefficienti_m
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
        J = np.array([[-1, 0.01], [0.01, -16]])
        if len(y.shape) == 1:  # y=y1,y2. Per Newton
            Y = J @ y
        elif len(y.shape) == 2:  # y: N x d x K. Per EKI
            Y = (J @ y.T).T
        elif len(y.shape) == 3:
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
        elif len(y.shape) == 2:  # y: N x d x K. Per EKI
            Y = np.array(
                [
                    -ma * y[:, 0] + mb * y[:, 1] * y[:, 2],
                    ma * y[:, 0] - mb * y[:, 1] * y[:, 2] - mc * y[:, 1] ** 2,
                    mc * y[:, 1] ** 2,
                ]
            ).T
        elif len(y.shape) == 3:
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
        elif len(y.shape) == 2:  # y: N x d x K. Per EKI
            Y = np.array([y[:, 1], mu * (1 - y[:, 0] ** 2) * y[:, 1] - y[:, 0]]).T
        elif len(y.shape) == 3:
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
        elif len(y.shape) == 2:  # y: N x d x K. Per EKI
            Y = np.array(
                [
                    77.27 * (y[:, 1] + y[:, 0] * (1 - 8.375e-6 * y[:, 0] - y[:, 1])),
                    1 / 77.27 * (y[:, 2] - (1 + y[:, 0]) * y[:, 1]),
                    0.161 * (y[:, 0] - y[:, 2]),
                ]
            ).T
        elif len(y.shape) == 3:
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
    elif case == 13:  # Caso Affine y' = Ay + f(t), Separabile
        # T=20, y0=9.9,0
        if len(y.shape) == 1:
            Y = np.array(
                [
                    -41 * y[0]
                    + 59 * y[1]
                    + -2 * t**3 * (t**2 - 50 * t - 2) * np.exp(-(t**2)),
                    40 * y[0]
                    - 60 * y[1]
                    + 2 * t**3 * (t**2 - 50 * t - 2) * np.exp(-(t**2)),
                ]
            )
        elif len(y.shape) == 2:  # y: N x d x K. Per EKI
            Y = np.array(
                [
                    -41 * y[:, 0]
                    + 59 * y[:, 1]
                    + -2 * t**3 * (t**2 - 50 * t - 2) * np.exp(-(t**2)),
                    40 * y[:, 0]
                    - 60 * y[:, 1]
                    + 2 * t**3 * (t**2 - 50 * t - 2) * np.exp(-(t**2)),
                ]
            ).T
        elif len(y.shape) == 3:
            Y = np.array(
                [
                    -41 * y[:, :, 0]
                    + 59 * y[:, :, 1]
                    + -2 * t**3 * (t**2 - 50 * t - 2) * np.exp(-(t**2)),
                    40 * y[:, :, 0]
                    - 60 * y[:, :, 1]
                    + 2 * t**3 * (t**2 - 50 * t - 2) * np.exp(-(t**2)),
                ]
            ).transpose(1, 2, 0)
    elif case == 14:  # Preda-Predatore
        # T=15, y0=1,0.1
        if len(y.shape) == 1:
            Y = np.array([ma * y[0] - mb * y[0] * y[1], mc * y[0] * y[1] - md * y[1]])
        elif len(y.shape) == 2:  # y: N x d x K. Per EKI
            Y = np.array(
                [
                    ma * y[:, 0] - mb * y[:, 0] * y[:, 1],
                    mc * y[:, 0] * y[:, 1] - md * y[:, 1],
                ]
            ).T
        elif len(y.shape) == 3:
            Y = np.array(
                [
                    ma * y[:, :, 0] - mb * y[:, :, 0] * y[:, :, 1],
                    mc * y[:, :, 0] * y[:, :, 1] - md * y[:, :, 1],
                ]
            ).transpose(1, 2, 0)
    elif case == 15:  # Caso scalare non lineare stiff
        if len(y.shape) == 1:  # y scalare. Per Newton
            Y = np.array([mu * y[0] * (1 - y[0])])  # y scalare. Per Newton
        elif len(y.shape) == 2:  # y: N x d x K. Per EKI
            Y = np.array([mu * y[:, 0] * (1 - y[:, 0])]).T
        elif len(y.shape) == 3:
            Y = np.array([mu * y[:, :, 0] * (1 - y[:, :, 0])]).transpose(1, 2, 0)
    return Y


############################################################################################
############################################################################################
############################################################################################
