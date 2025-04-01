import libreria_tesi as lib
import numpy as np


def EKI_Coupled(u0, U, y, G, eta, Ntmax, IGamma, s_stadi=0, K_dim=0):
    # g = lambda u: np.array(list(map(g, u))) # Applico g a ogni controllo
    coefficienti_coupled_direct_approach(s_stadi)
    N, d = np.shape(u0)
    u = np.zeros((Ntmax + 1, N, d))
    u[0] = u0
    TETA = np.zeros(Ntmax)  # Vettore dei Misfit
    res = np.zeros(Ntmax)  # Vettore dei residui
    for n in range(Ntmax):
        # print(n, '/', Ntmax)
        # Gu = g(u[n]) # Applico g a u
        Gu = G(u[n])
        if __name__ == "__main__":
            r = u[n] - U  # Residuo
            rr = np.mean(lib.norm(r, axis=1) ** 2)
            print("    residuo =", rr)
            res[n] = rr
        # teta_vett = (G(r) - eta)
        teta_vett = Gu - y
        teta = np.mean(lib.norm(teta_vett, axis=1) ** 2)
        TETA[n] = teta
        # print('    misfit =', teta, '\n    norm(rumore) =', norm(eta) ** 2)
        # Creo le matrici di covarianza
        u_centrato = u[n] - np.mean(u[n], axis=0)
        Gu_centrato = Gu - np.mean(Gu, axis=0)
        CuG = (u_centrato.T @ Gu_centrato) / (N)
        CGG = (Gu_centrato.T @ Gu_centrato) / (N)

        # Aggiorno u
        # u[n+1] = u[n] + np.tile(CuG @ np.linalg.inv(CGG + IGamma), (N,1,1)) @ (np.tile(y, (N,1,1)) - np.tile(G, (N,1,1)) @ u[n].T).T
        L, l = lib.cho_factor(CGG + IGamma)
        K_gain = CuG @ lib.cho_solve((L, l), np.eye(len(L)))
        u[n + 1] = u[n] + (K_gain @ (y - Gu).T).T
        # u[n + 1] = u[n] + (CuG @ np.linalg.inv(CGG + IGamma) @ (y - Gu).T).T
        if teta < lib.norm(eta) ** 2:
            print("EKI-Converge in ", n, "iterazioni")
            return u[: n + 2], TETA[: n + 1], res[: n + 1]
    # print(f"EKI-Non converge in {Ntmax} iterazioni. Misfit = {teta}, norm(eta)2 = {norm(eta) ** 2}")
    print(
        f"EKI-Non converge in {Ntmax} iterazioni. Misfit - norm(eta)2 = {teta - lib.norm(eta) ** 2 :.6f}. Misfit / norm(eta)2 = {teta / lib.norm(eta) ** 2 :.6f}"
    )
    return u, TETA, res


def coefficienti_coupled_direct_approach(s, Nlam=20):
    lam = np.random.uniform(0, 1, (Nlam, s))
    lam /= np.sum(lam, axis=1)[:, np.newaxis]
    return lam


"""Esempi di G"""


def G(u, d, K, dx, dy, Gg, calc):  # G lineare
    N = len(u)
    if Gg == 0:
        # Discretizzazione di (-d^2/dx^2 + 1)^-1
        A = 1 / (dx * dy) * (
            2 * np.eye(K) - np.diag(np.ones(K - 1), 1) - np.diag(np.ones(K - 1), -1)
        ) + np.eye(K)
        g = np.linalg.inv(A)
        if calc:
            gC0 = np.linalg.inv(A - np.eye(d))
            return gC0
        if len(np.shape(u)) == 1:  # u è un vettore d. Serve per calcolare y e teta
            return g @ u
        else:  # u è una matrice N x d
            g = lib.block_diag([g] * N)
            p = g @ u.flatten()
            return p.reshape((N, K))
    elif Gg == 1:
        # Discretizzazione di (d^2/dx^2)^-1
        A = (
            1
            / (dx * dy)
            * (
                -2 * np.diag(np.ones(K))
                + np.diag(np.ones(K - 1), 1)
                + np.diag(np.ones(K - 1), -1)
            )
        )
        g = np.linalg.inv(A)
        if calc:
            gC0 = np.linalg.inv(A - np.eye(d))
            return gC0
        if len(np.shape(u)) == 1:  # u è un vettore d. Serve per calcolare y e teta
            return g @ u
        else:  # u è una matrice N x d
            g = lib.block_diag([g] * N)
            p = g @ u.flatten()
            return p.reshape((N, K))
    elif Gg == 2:
        # Discretizzazione di (d/dx + 1)^-1
        A = 1 / (2 * dx) * (np.eye(K) - np.diag(np.ones(K - 1), -1)) + np.eye(K)
        g = np.linalg.inv(A)
        if calc:
            gC0 = np.linalg.inv(A - np.eye(d))
            return gC0
        if len(np.shape(u)) == 1:  # u è un vettore d. Serve per calcolare y e teta
            return g @ u
        else:  # u è una matrice N x d
            g = lib.block_diag([g] * N)
            p = g @ u.flatten()
            return p.reshape((N, K))
    elif Gg == 3:  # G generale
        # Discretizzazione di -(exp(u) p')' = 1
        if calc:  # Calcolo gC0
            A = (
                np.exp(u)
                / dx**2
                * (
                    -np.diag(u[:-1], -1)
                    + np.diag(u + np.append(u[1:], 2 * u[-1] - u[-2]))
                    + np.diag(-u[1:], 1)
                )
            )
            gC0 = np.linalg.inv(A - np.eye(d))
            return gC0
        # se calc == False
        if len(np.shape(u)) == 1:  # u è un vettore d. Serve per calcolare y e teta
            A = (
                np.exp(u)
                / dx**2
                * (
                    -np.diag(u[:-1], -1)
                    + np.diag(u + np.append(u[1:], 2 * u[-1] - u[-2]))
                    + np.diag(-u[1:], 1)
                )
            )
            b = np.ones(K)
            p = np.linalg.solve(A, b)
            return p
        else:  # u è una matrice N x d
            # ud = np.hstack((u[:, :-1], np.zeros((N, 1)))); ud = ud.flatten() # u[i-1] sottodiagonale
            # uu = np.hstack((np.zeros((N, 1)), u[:, 1:])); uu = uu.flatten() # u[i+1] sopradiagonale
            uc = u.flatten()  # u[i] diagonale
            ud = u.flatten()
            ud[d - 1 :: d] = 0  # u[i-1] sottodiagonale
            uu = u.flatten()
            uu[0::d] = 0  # u[i+1] sopradiagonale
            diagonals = [
                -ud[:-1] * np.exp(ud[:-1]) / dx**2,  # subdiagonal
                (uc + np.hstack((u[:, 1:], 2 * u[:, -1:] - u[:, -2:-1])).flatten())
                * np.exp(uc)
                / dx**2,  # main diagonal
                -uu[1:] * np.exp(uu[1:]) / dx**2,  # superdiagonal
            ]
            A = lib.diags(diagonals, offsets=[-1, 0, 1], shape=(N * K, N * K))
            b = np.ones(N * K)
            p = lib.spsolve(A, b)
            return p.reshape((N, K))
    elif Gg == 4:
        # Discretizzazione di -p'' + up' = u
        if calc:
            A = (
                1
                / dx**2
                * (
                    2 * np.eye(K)
                    - np.diag(np.ones(K - 1), 1)
                    - np.diag(np.ones(K - 1), -1)
                )
            )
            A += 1 / dx * (-np.diag(u) + np.diag((u[:-1]), 1))
            gC0 = np.linalg.inv(A - np.eye(d))
            return gC0
        if len(np.shape(u)) == 1:  # u è un vettore d. Serve per calcolare y=g(u)+eta
            A = (
                1
                / dx**2
                * (
                    2 * np.eye(K)
                    - np.diag(np.ones(K - 1), 1)
                    - np.diag(np.ones(K - 1), -1)
                )
            )
            A += 1 / dx * (-np.diag(u) + np.diag((u[:-1]), 1))
            b = u
            p = np.linalg.solve(A, b)
            return p
        else:
            uc = u.flatten()  # u[i] diagonale
            ud = u.flatten()
            ud[d - 1 :: d] = 0  # u[i-1] sottodiagonale
            uu = u.flatten()
            uu[0::d] = 0  # u[i+1] sopradiagonale
            uno = np.ones(N * K)
            uno[0::d] = 0
            diagonals = [
                1 / dx**2 * (-uno[1:]),  # subdiagonal
                1 / dx**2 * (2) + 1 / dx * (-uc),  # main diagonal
                1 / dx**2 * (-uno[1:]) + 1 / dx * (uu[:-1]),  # superdiagonal
            ]
            A = lib.diags(diagonals, offsets=[-1, 0, 1], shape=(N * K, N * K))
            b = uc
            p = lib.spsolve(A, b)
            return p.reshape((N, K))
    elif Gg == 5:
        # Discretizzazione di d2/dx2(pu) + d/Dx(p log(|u|)) + p = 1
        if calc:
            A = (
                1
                / dx**2
                * (
                    np.diag(u[:-1], -1)
                    - 2 * np.diag(u + np.append(u[1:], 2 * u[-1] - u[-2]))
                    + np.diag(2 * u[1:] - u[:-1], 1)
                )
            )
            A += (
                1
                / dx
                * (
                    -np.diag(np.log(np.abs(u + 1e-8)))
                    + np.diag(np.log(np.abs(u[:-1] + 1e-8)), 1)
                )
            )
            A += np.diag(1 / np.abs(u + 1e-8) + 1)
            gC0 = np.linalg.inv(A - np.eye(d))
            return gC0
        # se calc == False
        if len(np.shape(u)) == 1:  # u è un vettore d. Serve per calcolare y=g(u)+eta
            A = (
                1
                / dx**2
                * (
                    np.diag(u[1:], -1)
                    - 2 * np.diag(u + np.append(u[1:], 2 * u[-1] - u[-2]))
                    + np.diag(2 * u[1:] - u[:-1], 1)
                )
            )
            A += (
                1
                / dx
                * (
                    -np.diag(np.log(np.abs(u + 1e-8)))
                    + np.diag(np.log(np.abs(u[:-1] + 1e-8)), 1)
                )
            )
            A += np.diag(1 / np.abs(u + 1e-8) + 1)
            b = np.ones(K)
            p = np.linalg.solve(A, b)
            return p
        else:  # u è una matrice N x d
            uc = u.flatten()  # u[i] diagonale
            ud = u.flatten()
            ud[d - 1 :: d] = 0  # u[i-1] sottodiagonale
            uu = u.flatten()
            uu[0::d] = 0  # u[i+1] sopradiagonale
            diagonals = [
                1 / dx**2 * (ud[1:]),  # subdiagonal
                1
                / dx**2
                * (
                    -2
                    * (
                        uc
                        + np.hstack((u[:, 1:], 2 * u[:, -1:] - u[:, -2:-1])).flatten()
                    )
                )
                + 1 / dx * (-np.log(np.abs(uc + 1e-8)))
                + (1 / np.abs(uc + 1e-8) + 1),  # main diagonal
                1 / dx**2 * (2 * uu[1:] - uc[:-1])
                + 1 / dx * (np.log(np.abs(uc[:-1] + 1e-8))),  # superdiagonal
            ]
            A = lib.diags(diagonals, offsets=[-1, 0, 1], shape=(N * K, N * K))
            b = np.ones(N * K)
            p = lib.spsolve(A, b)
            return p.reshape((N, K))
    elif Gg == 6:
        # Discretizzazione di -(exp(u_1) p')' = 1, DIM == 2
        if calc:  # Calcolo gC0
            u1 = u[0]
            u2 = u[1]
            A = np.exp(u1) / dx**2 * (-np.eye(K, k=-1) + 2 * np.eye(K) - np.eye(K, k=1))
            gC0 = np.linalg.inv(A - np.eye(K))
            return gC0
        # se calc == False
        if len(np.shape(u)) == 1:  # u è un vettore d. Serve per calcolare y e teta
            u1 = u[0]
            u2 = u[1]
            A = np.exp(u1) / dx**2 * (-np.eye(K, k=-1) + 2 * np.eye(K) - np.eye(K, k=1))
            b = np.ones(K)
            # Condizioni al bordo
            A[0, 0] = 1
            A[-1, -1] = 1
            b[0] = 0
            b[-1] = u2
            p = np.linalg.solve(A, b)
            return p
        else:  # u è una matrice N x d
            u1 = u[:, 0]
            u2 = u[:, 1]
            uc1 = lib.einops.repeat(u1, "n -> (n rep)", rep=K)  # u[i] diagonale
            ud1 = lib.einops.repeat(u1, "n -> (n rep)", rep=K)
            ud1[K - 1 :: K] = 0  # u[i-1] sottodiagonale
            uu1 = lib.einops.repeat(u1, "n -> (n rep)", rep=K)
            uu1[K - 1 :: K] = 0  # u[i+1] sopradiagonale
            diagonals = [
                (-np.exp(ud1) / dx**2),  # subdiagonal
                (2 * np.exp(uc1) / dx**2),  # main diagonal
                (-np.exp(uu1) / dx**2),  # superdiagonal
            ]
            A = lib.diags(diagonals, offsets=[-1, 0, 1], shape=(N * K, N * K))
            b = np.ones(N * K)
            # Condizioni al bordo
            A.data[1, 0::K] = 1
            A.data[1, K - 1 :: K] = 1
            b[0::K] = 0
            b[K - 1 :: K] = u2
            p = lib.spsolve(A, b)
            return p.reshape((N, K))


"""Fine esempi di G"""


# Grafico
def grafico(x, y, u0, uM, U, GU, GuM, teta, res, SALVA=False, savepath="\Desktop\\"):
    lib.plt.figure(1)
    lib.plt.title("Dati")
    lib.plt.plot(x, y, "x--y", label="osservazioni")
    lib.plt.plot(x, GU, "-g", label=f"Sol Esatta")
    lib.plt.plot(x, GuM, "-r", label=f"usando controllo ricostruito")
    lib.plt.xlabel("x")
    lib.plt.legend()
    if SALVA:
        lib.plt.savefig(savepath + "Dati Ricostruiti.png")

    lib.plt.figure(2)
    lib.plt.title("Ricostruzione Controllo")
    lib.plt.plot(x, uM, "x-r", label=f"controllo ricostruito")
    lib.plt.plot(x, np.mean(u0, axis=0), "y", label=f"controllo iniziale")
    lib.plt.plot(x, U, "-g", label=f"controllo esatto")
    lib.plt.xlabel("x")
    lib.plt.legend()
    if SALVA:
        lib.plt.savefig(savepath + "Ricostruzione Controllo.png")

    lib.plt.figure(3)
    lib.plt.title("Misfit")
    lib.plt.semilogy(teta, "x--", label="Misfit")
    lib.plt.semilogy(np.ones(len(teta)) * lib.norm(eta) ** 2, label="norm(eta)2")
    lib.plt.grid(True, axis="y", which="major", linestyle="--", linewidth=0.5)
    lib.plt.xlabel("Iterazioni")
    lib.plt.legend()
    if SALVA:
        lib.plt.savefig(savepath + "Misfit.png")

    lib.plt.figure(4)
    lib.plt.title("Residuo")
    lib.plt.semilogy(res, "x--", label="Residuo")
    lib.plt.grid(True, axis="y", which="both", linestyle="--", linewidth=0.5)
    lib.plt.xlabel("Iterazioni")
    lib.plt.legend()
    if SALVA:
        lib.plt.savefig(savepath + "Residuo.png")

    lib.plt.figure(5)
    lib.plt.title("Controlli iniziali")
    for i in range(N):
        lib.plt.plot(x, u0[i], "-", label=f"controllo iniziale {i}")
    lib.plt.plot(x, U, "o", label=f"controllo esatta")
    if SALVA:
        lib.plt.savefig(savepath + "Controlli iniziali.png")

    lib.plt.show()


def grafico2D(
    x, y, u0, uM, u, U, GU, GuM, teta, res, SALVA=False, savepath="\Desktop\\"
):
    lib.plt.figure(1)
    lib.plt.title("Dati")
    lib.plt.plot(x, y, "x--y", label="osservazioni")
    lib.plt.plot(x, GU, "-g", label=f"Sol Esatta")
    lib.plt.plot(x, GuM, "-r", label=f"usando controllo ricostruito")
    lib.plt.xlabel("x")
    lib.plt.legend()
    if SALVA:
        lib.plt.savefig(savepath + "Dati Ricostruiti.png")

    lib.plt.figure(2)
    lib.plt.title("Ricostruzione Controllo")
    lib.plt.plot(u[:, 0], u[:, 1], ".r", label=f"ensemble controllo finale")
    lib.plt.plot(uM[0], uM[1], "oy", label=f"media controllo finale")
    lib.plt.plot(U[0], U[1], "xg", label=f"controllo esatto")
    lib.plt.plot(u0[:, 0], u0[:, 1], "b.", label=f"controllo iniziale")
    lib.plt.xlabel("u1")
    lib.plt.ylabel("u2")
    lib.plt.legend()
    if SALVA:
        lib.plt.savefig(savepath + "Ricostruzione Controllo.png")

    lib.plt.figure(3)
    lib.plt.title("Misfit")
    lib.plt.semilogy(teta, "x--", label="Misfit")
    lib.plt.semilogy(np.ones(len(teta)) * lib.norm(eta) ** 2, label="norm(eta)2")
    lib.plt.grid(True, axis="y", which="major", linestyle="--", linewidth=0.5)
    lib.plt.xlabel("Iterazioni")
    lib.plt.legend()
    if SALVA:
        lib.plt.savefig(savepath + "Misfit.png")

    lib.plt.figure(4)
    lib.plt.title("Residuo")
    lib.plt.semilogy(res, "x--", label="Residuo")
    lib.plt.grid(True, axis="y", which="both", linestyle="--", linewidth=0.5)
    lib.plt.xlabel("Iterazioni")
    lib.plt.legend()
    if SALVA:
        lib.plt.savefig(savepath + "Residuo.png")

    fig = lib.plt.figure(5)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Controlli iniziali")
    # Calcolo della densità
    xy = np.vstack([u0[:, 0], u0[:, 1]])
    xi, yi = np.meshgrid(
        np.linspace(u0[:, 0].min(), u0[:, 0].max(), 100),
        np.linspace(u0[:, 1].min(), u0[:, 1].max(), 100),
    )
    zi = lib.gaussian_kde(xy)(np.vstack([xi.flatten(), yi.flatten()]))
    zi = zi.reshape(xi.shape)
    ax.plot_surface(xi, yi, zi, cmap="viridis")
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    ax.set_zlabel("Densità")
    if SALVA:
        lib.plt.savefig(savepath + "Controlli iniziali.png")

    lib.plt.show()


if __name__ == "__main__":
    print("Dati iniziali")
    Ntmax = 100
    gamma = 1e-2
    Gg = 6  # Modello G da usare (0-1-2 lineari, 3-4-5 non lineari, 6 2D)
    if Gg in [0, 1, 2, 3, 4, 5]:
        d = K = 256
    elif Gg in [6]:
        d, K = 2, 256
    Dx = [0, np.pi]  # Dominio spaziale
    dx = (Dx[1] - Dx[0]) / K  # Passo spaziale
    # dy = (Dx[1] - Dx[0]) / d #???

    N = 100  # Realizzazione del controllo, numero di ensemble
    # Uu = lambda x: np.ones(len(x)) # Controllo esatto. U_croce
    # Uu = lambda x: np.sin(8 * x) # Controllo esatto. U_croce
    # Uu = lambda x: np.abs(x) # Controllo esatto. U_croce
    # Uu = lambda x: x**3
    Uu = lambda x: np.exp(-80 * (x - np.pi / 2) ** 2)
    savepath = "Desktop\\"
    # SALVA = True
    SALVA = False

    x = np.linspace(Dx[0], Dx[1], K)
    y = np.linspace(Dx[0], Dx[1], d)
    if Gg in [0, 1, 2, 3, 4, 5]:
        U = Uu(y)
    elif Gg in [6]:
        U = np.array([-2.65, 104.5])
    g = lambda u, calc=False: G(u, d, K, dx, dx, Gg, calc)  # calc==True calcola cG0
    gC0 = g(U, True)

    # Controlli iniziali
    t = np.linspace(0, 1, d)
    if Gg in [0, 1, 2, 3, 4, 5]:
        # browniano prof
        # try: u0 = np.mean(U)*np.ones((N, d)) + np.random.randn(N, d) @ np.linalg.cholesky(10*gC0)
        # except: u0 = np.mean(U)*np.ones((N, d)) + np.random.randn(N, d) @ np.linalg.cholesky(10*(gC0@gC0.T))
        # casuale prof
        u0 = np.array([U + 0.25 + np.random.randn() for _ in range(N)])
        # casuale
        # u0 = np.array([U + 10*np.random.normal(0, gamma, d) for _ in range(N)])
    elif Gg == 6:
        u0 = np.zeros((N, d))
        u0[:, 0] = np.random.randn(N)
        u0[:, 1] = np.random.uniform(90, 110, N)

    IGamma = gamma**2 * np.eye(K)  # Matrice di Covarianza del rumore eta
    eta = np.random.normal(0, gamma, K)  # Rumore se IGamma è diagonale

    GU = g(U)  # G(Controllo esatto)
    y = GU + eta  # Dati osservati
    u, teta, res = EKI_Coupled(u0, U, y, g, eta, Ntmax, IGamma)
    um = np.mean(u[-1], axis=0)  # Media di u finale
    print("Fine calcolo")
    GuM = g(um)  # G(Controllo ricostruito)
    if Gg in [0, 1, 2, 3, 4, 5]:
        grafico(x, y, u0, um, U, GU, GuM, teta, res, SALVA, savepath)
    elif Gg in [6]:
        grafico2D(x, y, u0, um, u[-1], U, GU, GuM, teta, res, SALVA, savepath)
    print("Fine grafico")
