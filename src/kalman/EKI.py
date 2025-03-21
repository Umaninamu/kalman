import numpy as np
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from scipy.linalg import solve_banded
from scipy.linalg import cho_solve, cho_factor
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.sparse import block_diag
import einops as ep
from scipy.stats import gaussian_kde


def EKI(u0, U, y, G, eta, Ntmax, IGamma, d_stadi=0, K_dim=0):
    # g = lambda u: np.array(list(map(g, u))) # Applico g a ogni controllo
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
            rr = np.mean(norm(r, axis=1) ** 2)
            print("    residuo =", rr)
            res[n] = rr
        # teta_vett = (G(r) - eta)
        teta_vett = Gu - y
        teta = np.mean(norm(teta_vett, axis=1) ** 2)
        TETA[n] = teta
        # print('    misfit =', teta, '\n    norm(rumore) =', norm(eta) ** 2)
        # Creo le matrici di covarianza
        # co = np.cov(np.vstack((u[n].T, Gu.T)))
        # co = np.cov(u[n].T, Gu.T)  # Simmetrica
        # CuG = co[:d, d:]
        # CGG = co[d:, d:]
        u_centrato = u[n] - np.mean(u[n], axis=0)
        Gu_centrato = Gu - np.mean(Gu, axis=0)
        CuG = (u_centrato.T @ Gu_centrato) / (N)
        CGG = (Gu_centrato.T @ Gu_centrato) / (N)
        # if __name__ == "__main__": #Per EKI classico
        #     CuG = (u_centrato.T @ Gu_centrato) / (N)
        #     CGG = (Gu_centrato.T @ Gu_centrato) / (N)
        # else: #Per ODE implicite. K_dim, d_stadi != 0.
        #     # d_stadi = Numero di stadi del metodo RK
        #     # K_dim = Dimensione del sistema
        #     u_centrato_d_K = ep.rearrange(u_centrato, "n (d k) -> n d k", d=d_stadi, k=K_dim)
        #     Gu_centrato_d_K = ep.rearrange(Gu_centrato, "n (d k) -> n d k", d=d_stadi, k=K_dim)
        #     CuG, CGG = np.zeros((d_stadi, d_stadi, K_dim)), np.zeros((d_stadi, d_stadi, K_dim))
        #     # for i in range(K_dim):
        #     #     CuG[:,:,i] = (u_centrato_d_K[:,:,i].T @ Gu_centrato_d_K[:,:,i]) / (N)
        #     #     CGG[:,:,i] = (Gu_centrato_d_K[:,:,i].T @ Gu_centrato_d_K[:,:,i]) / (N)
        #     CuG = np.einsum("ndk,nek->dek", u_centrato_d_K, Gu_centrato_d_K) / (N)
        #     CGG = np.einsum("ndk,nek->dek", Gu_centrato_d_K, Gu_centrato_d_K) / (N)
        #     #CuG = block_diag([CuG[:, :, i] for i in range(K_dim)]).toarray()
        #     #CGG = block_diag([CGG[:, :, i] for i in range(K_dim)]).toarray()
        #     for i in range(K_dim):
        #         u_next = np.zeros((N, d))
        #         u_n = ep.rearrange(u[n], "n (d k) -> n (k d)", d=d_stadi, k=K_dim)[:, i*d_stadi: (i+1)*d_stadi]
        #         L, l = cho_factor(CGG[:, :, i] + IGamma[:d_stadi, :d_stadi])
        #         K_gain = CuG[:, :, i] @ cho_solve((L,l), np.eye(len(L)))
        #         u_next[:, i*d_stadi: (i+1)*d_stadi] = u_n + (K_gain @ ((y - Gu)[:, i*d_stadi: (i+1)*d_stadi]).T).T
        #     u[n + 1] = ep.rearrange(u_next, "n (k d) -> n (d k)", d=d_stadi, k=K_dim)

        # CuG, CGG = np.zeros((d, d)), np.zeros((d, d))
        # Cug = (u_centrato.T @ Gu_centrato) / (N)
        # Cgg = (Gu_centrato.T @ Gu_centrato) / (N)
        # for i in range(K_dim):
        #     CuG[i*d_stadi:(i+1)*d_stadi, i*d_stadi:(i+1)*d_stadi] = Cug[i*d_stadi:(i+1)*d_stadi, i*d_stadi:(i+1)*d_stadi]
        #     CGG[i*d_stadi:(i+1)*d_stadi, i*d_stadi:(i+1)*d_stadi] = Cgg[i*d_stadi:(i+1)*d_stadi, i*d_stadi:(i+1)*d_stadi]

        # Aggiorno u
        # u[n+1] = u[n] + np.tile(CuG @ np.linalg.inv(CGG + IGamma), (N,1,1)) @ (np.tile(y, (N,1,1)) - np.tile(G, (N,1,1)) @ u[n].T).T
        L, l = cho_factor(CGG + IGamma)
        K_gain = CuG @ cho_solve((L, l), np.eye(len(L)))
        u[n + 1] = u[n] + (K_gain @ (y - Gu).T).T
        # u[n + 1] = u[n] + (CuG @ np.linalg.inv(CGG + IGamma) @ (y - Gu).T).T
        if teta < norm(eta) ** 2:
            print("EKI-Converge in ", n, "iterazioni")
            return u[: n + 2], TETA[: n + 1], res[: n + 1]
    # print(f"EKI-Non converge in {Ntmax} iterazioni. Misfit = {teta}, norm(eta)2 = {norm(eta) ** 2}")
    print(
        f"EKI-Non converge in {Ntmax} iterazioni. Misfit - norm(eta)2 = {teta - norm(eta) ** 2 :.6f}. Misfit / norm(eta)2 = {teta / norm(eta) ** 2 :.6f}"
    )
    return u, TETA, res


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
            g = block_diag([g] * N)
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
            g = block_diag([g] * N)
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
            g = block_diag([g] * N)
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
            A = diags(diagonals, offsets=[-1, 0, 1], shape=(N * K, N * K))
            b = np.ones(N * K)
            p = spsolve(A, b)
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
            A = diags(diagonals, offsets=[-1, 0, 1], shape=(N * K, N * K))
            b = uc
            p = spsolve(A, b)
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
            A = diags(diagonals, offsets=[-1, 0, 1], shape=(N * K, N * K))
            b = np.ones(N * K)
            p = spsolve(A, b)
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
            uc1 = ep.repeat(u1, "n -> (n rep)", rep=K)  # u[i] diagonale
            ud1 = ep.repeat(u1, "n -> (n rep)", rep=K)
            ud1[K - 1 :: K] = 0  # u[i-1] sottodiagonale
            uu1 = ep.repeat(u1, "n -> (n rep)", rep=K)
            uu1[K - 1 :: K] = 0  # u[i+1] sopradiagonale
            diagonals = [
                (-np.exp(ud1) / dx**2),  # subdiagonal
                (2 * np.exp(uc1) / dx**2),  # main diagonal
                (-np.exp(uu1) / dx**2),  # superdiagonal
            ]
            A = diags(diagonals, offsets=[-1, 0, 1], shape=(N * K, N * K))
            b = np.ones(N * K)
            # Condizioni al bordo
            A.data[1, 0::K] = 1
            A.data[1, K - 1 :: K] = 1
            b[0::K] = 0
            b[K - 1 :: K] = u2
            p = spsolve(A, b)
            return p.reshape((N, K))


"""Fine esempi di G"""


# Grafico
def grafico(x, y, u0, uM, U, GU, GuM, teta, res, SALVA=False, savepath="\Desktop\\"):
    plt.figure(1)
    plt.title("Dati")
    plt.plot(x, y, "x--y", label="osservazioni")
    plt.plot(x, GU, "-g", label=f"Sol Esatta")
    plt.plot(x, GuM, "-r", label=f"usando controllo ricostruito")
    plt.xlabel("x")
    plt.legend()
    if SALVA:
        plt.savefig(savepath + "Dati Ricostruiti.png")

    plt.figure(2)
    plt.title("Ricostruzione Controllo")
    plt.plot(x, uM, "x-r", label=f"controllo ricostruito")
    plt.plot(x, np.mean(u0, axis=0), "y", label=f"controllo iniziale")
    plt.plot(x, U, "-g", label=f"controllo esatto")
    plt.xlabel("x")
    plt.legend()
    if SALVA:
        plt.savefig(savepath + "Ricostruzione Controllo.png")

    plt.figure(3)
    plt.title("Misfit")
    plt.semilogy(teta, "x--", label="Misfit")
    plt.semilogy(np.ones(len(teta)) * norm(eta) ** 2, label="norm(eta)2")
    plt.grid(True, axis="y", which="major", linestyle="--", linewidth=0.5)
    plt.xlabel("Iterazioni")
    plt.legend()
    if SALVA:
        plt.savefig(savepath + "Misfit.png")

    plt.figure(4)
    plt.title("Residuo")
    plt.semilogy(res, "x--", label="Residuo")
    plt.grid(True, axis="y", which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("Iterazioni")
    plt.legend()
    if SALVA:
        plt.savefig(savepath + "Residuo.png")

    plt.figure(5)
    plt.title("Controlli iniziali")
    for i in range(N):
        plt.plot(x, u0[i], "-", label=f"controllo iniziale {i}")
    plt.plot(x, U, "o", label=f"controllo esatta")
    if SALVA:
        plt.savefig(savepath + "Controlli iniziali.png")

    plt.show()


def grafico2D(
    x, y, u0, uM, u, U, GU, GuM, teta, res, SALVA=False, savepath="\Desktop\\"
):
    plt.figure(1)
    plt.title("Dati")
    plt.plot(x, y, "x--y", label="osservazioni")
    plt.plot(x, GU, "-g", label=f"Sol Esatta")
    plt.plot(x, GuM, "-r", label=f"usando controllo ricostruito")
    plt.xlabel("x")
    plt.legend()
    if SALVA:
        plt.savefig(savepath + "Dati Ricostruiti.png")

    plt.figure(2)
    plt.title("Ricostruzione Controllo")
    plt.plot(u[:, 0], u[:, 1], ".r", label=f"ensemble controllo finale")
    plt.plot(uM[0], uM[1], "oy", label=f"media controllo finale")
    plt.plot(U[0], U[1], "xg", label=f"controllo esatto")
    plt.plot(u0[:, 0], u0[:, 1], "b.", label=f"controllo iniziale")
    plt.xlabel("u1")
    plt.ylabel("u2")
    plt.legend()
    if SALVA:
        plt.savefig(savepath + "Ricostruzione Controllo.png")

    plt.figure(3)
    plt.title("Misfit")
    plt.semilogy(teta, "x--", label="Misfit")
    plt.semilogy(np.ones(len(teta)) * norm(eta) ** 2, label="norm(eta)2")
    plt.grid(True, axis="y", which="major", linestyle="--", linewidth=0.5)
    plt.xlabel("Iterazioni")
    plt.legend()
    if SALVA:
        plt.savefig(savepath + "Misfit.png")

    plt.figure(4)
    plt.title("Residuo")
    plt.semilogy(res, "x--", label="Residuo")
    plt.grid(True, axis="y", which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("Iterazioni")
    plt.legend()
    if SALVA:
        plt.savefig(savepath + "Residuo.png")

    fig = plt.figure(5)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Controlli iniziali")
    # Calcolo della densità
    xy = np.vstack([u0[:, 0], u0[:, 1]])
    xi, yi = np.meshgrid(
        np.linspace(u0[:, 0].min(), u0[:, 0].max(), 100),
        np.linspace(u0[:, 1].min(), u0[:, 1].max(), 100),
    )
    zi = gaussian_kde(xy)(np.vstack([xi.flatten(), yi.flatten()]))
    zi = zi.reshape(xi.shape)
    ax.plot_surface(xi, yi, zi, cmap="viridis")
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    ax.set_zlabel("Densità")
    if SALVA:
        plt.savefig(savepath + "Controlli iniziali.png")

    plt.show()


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
    u, teta, res = EKI(u0, U, y, g, eta, Ntmax, IGamma)
    um = np.mean(u[-1], axis=0)  # Media di u finale
    print("Fine calcolo")
    GuM = g(um)  # G(Controllo ricostruito)
    if Gg in [0, 1, 2, 3, 4, 5]:
        grafico(x, y, u0, um, U, GU, GuM, teta, res, SALVA, savepath)
    elif Gg in [6]:
        grafico2D(x, y, u0, um, u[-1], U, GU, GuM, teta, res, SALVA, savepath)
    print("Fine grafico")
