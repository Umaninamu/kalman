import libreria_tesi as lib
import numpy as np
from numba import njit, float64, int64


@njit(
    (
        float64[:, :, :],
        float64[:],
        float64[:, :],
        float64[:],
        float64[:, :],
        int64,
        int64,
        float64[:],
        float64[:],
    )
)
def EKI_n(u, y, Gu, eta, IGamma, N, n, TETA, res):
    teta_vett = Gu - y
    teta = np.sum(teta_vett**2) / N
    TETA[n] = teta

    if teta < np.sum(eta**2):  # Converge
        return u[: n + 1], TETA[: n + 1], res[: n + 1], 0
    elif np.abs(TETA[n] - TETA[n - 1]) < 1e-14 and teta < np.sum(eta**2) * 5:  # Residuo
        return u[: n + 1], TETA[: n + 1], res[: n + 1], 1
    elif teta < np.sum(eta**2) * 1.3 and np.abs(TETA[n] - TETA[n - 1]) < 1:  # Plus
        return u[: n + 1], TETA[: n + 1], res[: n + 1], 2

    u_centrato = u[n] - np.sum(u[n], axis=0) / N
    Gu_centrato = Gu - np.sum(Gu, axis=0) / N
    CuG = (u_centrato.T @ Gu_centrato) / N
    CGG = (Gu_centrato.T @ Gu_centrato) / N

    K_gain = CuG @ np.linalg.inv(CGG + IGamma)
    u[n + 1] = u[n] + (K_gain @ (y - Gu).T).T

    return u, TETA, res, 10


def EKI(u0, U, y, G, eta, Ntmax, IGamma, s_stadi=0, Nlam=15):
    # g = lambda u: np.array(list(map(g, u))) # Applico g a ogni controllo
    N, d = np.shape(u0)
    u = np.zeros((Ntmax + 1, N, d))
    u[0] = u0
    TETA = np.zeros(Ntmax)  # Vettore dei Misfit
    res = np.zeros(Ntmax)  # Vettore dei residui
    for n in range(Ntmax):
        Gu = G(u[n])
        u, TETA, res, converge = EKI_n(u, y, Gu, eta, IGamma, N, n, TETA, res)
        # u, TETA, res, converge = lib.my_pstats_profiler(EKI_n,u, y, Gu, eta, IGamma, N, n, TETA, res)
        if converge == 0:
            # print("\tEKI-Converge in ", n, "iterazioni")
            return u, TETA, res
        elif converge == 1:
            # print("\tEKI-Residuo in ", n, "iterazioni")
            return u, TETA, res
        elif converge == 2:
            # print("\tEKI-Plus in ", n, "iterazioni")
            return u, TETA, res
        elif converge == 3:
            # print("\tEKI-ResPlus in ", n, "iterazioni")
            return u, TETA, res
    if converge == 10:
        print(
            "\tEKI-Non converge in ",
            n,
            "iterazioni. Diff = ",
            TETA[n] - np.sum(eta**2),
            "Rapp = ",
            TETA[n] / np.sum(eta**2),
        )

    return u, TETA, res


# Grafico
def grafico(
    x, y, u0, uM, U, GU, GuM, teta, eta, N, res, SALVA=False, savepath="\Desktop\\"
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
    lib.plt.semilogy(np.ones(len(teta)) * np.sum(eta**2), label="norm(eta)2")
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
    x, y, u0, uM, u, U, GU, GuM, teta, eta, res, SALVA=False, savepath="\Desktop\\"
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
    lib.plt.semilogy(np.ones(len(teta)) * np.sum(eta**2), label="norm(eta)2")
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
    lib.dati.comincia()
