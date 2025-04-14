import libreria_tesi as lib
import numpy as np
from numba import njit, float64, int64


# @njit(
#     (
#         float64[:, :, :],
#         float64[:],
#         float64[:, :],
#         float64[:],
#         float64[:, :],
#         int64,
#         int64,
#         float64[:],
#         float64[:],
#         float64[:],
#         float64[:],
#     ),
#     cache=True,
# )
def EKI_n(u, y, Gu, eta, IGamma, N, n, TETA, res, Sigma, G_Sigma, alpha, beta):
    teta_vett = Gu - y
    teta = np.sum(teta_vett**2) / N
    TETA[n] = teta
    err = np.sum(eta**2)
    Tdiff = np.abs(TETA[n] - TETA[n - 1])
    if teta < err:  # Converge
        return u[: n + 1], TETA[: n + 1], res[: n + 1], 0
    elif teta < err * 1.3:  # and Tdiff < 1:  # Plus
        return u[: n + 1], TETA[: n + 1], res[: n + 1], 2
    elif Tdiff < 1e-10:  # and teta < err * 5:  # Residuo
        return u[: n + 1], TETA[: n + 1], res[: n + 1], 1

    # Metodo Covarianza Diagonale + Convergenza Rapida
    u_centrato = u[n] - np.sum(u[n], axis=0) / N
    Gu_centrato = Gu - np.sum(Gu, axis=0) / N
    # Cuu = np.sum(u_centrato * u_centrato, axis=0) / N
    CuG = np.sum(u_centrato * Gu_centrato, axis=0) / N
    CGG = np.sum(Gu_centrato * Gu_centrato, axis=0) / N
    K_gain = (CuG + (1 - alpha) * G_Sigma) / (CGG + IGamma[0, 0])
    R = beta * (CuG + (1 - alpha) * Sigma) * u_centrato
    u[n + 1] = u[n] + (K_gain * (y - Gu)) + R

    return u, TETA, res, 10


def EKI(u0, y, G, eta, Ntmax, IGamma, alpha, beta):
    N, d = np.shape(u0)
    u = np.zeros((Ntmax + 1, N, d))
    u[0] = u0
    TETA = np.zeros(Ntmax)  # Vettore dei Misfit
    res = np.zeros(Ntmax)  # Vettore dei residui
    Sigma = np.copy(np.diag(IGamma))  # Dovrebbe essere matrice dxd simmetrica
    G_Sigma = np.copy(np.diag(G(IGamma)))
    for n in range(Ntmax):
        Gu = G(u[n])
        u, TETA, res, converge = EKI_n(
            u, y, Gu, eta, IGamma, N, n, TETA, res, Sigma, G_Sigma, alpha, beta
        )
        if converge == 0:
            # print("\tEKI-Converge in ", n, "iterazioni")
            return u, TETA, res
        elif converge == 1:
            # print("\tEKI-Residuo in ", n, "iterazioni")
            return u, TETA, res
        elif converge == 2:
            # print("\tEKI-Plus in ", n, "iterazioni")
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


if __name__ == "__main__":
    np.random.seed(42)
    lib.dati.comincia()
