import numpy as np
from scipy.optimize import fsolve


def dirk_step(f, y_n, t_n, dt, A, b, c):
    """
    Esegue un passo del metodo DIRK per un sistema di equazioni differenziali.

    Parametri:
    - f: funzione che rappresenta il sistema (f(t, y))
    - y_n: valore attuale della soluzione
    - t_n: tempo attuale
    - dt: passo temporale
    - A: matrice dei coefficienti di Butcher (DIRK)
    - b: vettore dei pesi di Butcher
    - c: vettore dei nodi di Butcher

    Ritorna:
    - y_n1: valore della soluzione al passo successivo
    """
    s = len(c)  # Numero di stadi
    d = len(y_n)  # Dimensione del sistema
    Y = np.zeros((s, d))  # Valori intermedi

    # Funzione per Newton-Raphson
    def F(Y_j, i):
        return (
            Y_j
            - y_n
            - dt * sum(A[i, k] * f(t_n + c[k] * dt, Y[k]) for k in range(i + 1))
        )

    for i in range(s):
        # Risolvi il sistema non lineare per Y[i] usando Newton-Raphson
        Y[i] = fsolve(lambda Y_j: F(Y_j, i), y_n)

    # Calcola y_n+1
    y_n1 = y_n + dt * sum(b[i] * f(t_n + c[i] * dt, Y[i]) for i in range(s))
    return y_n1


def runge_kutta_dirk(f, y0, t0, t_end, dt, A, b, c):
    """
    Risolve un sistema di equazioni differenziali usando il metodo DIRK.

    Parametri:
    - f: funzione che rappresenta il sistema (f(t, y))
    - y0: condizione iniziale
    - t0: tempo iniziale
    - t_end: tempo finale
    - dt: passo temporale
    - A, b, c: parametri della matrice di Butcher

    Ritorna:
    - t: array dei tempi
    - y: array delle soluzioni
    """
    t = np.arange(t0, t_end + dt, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for n in range(1, len(t)):
        y[n] = dirk_step(f, y[n - 1], t[n - 1], dt, A, b, c)

    return t, y


# Esempio di utilizzo
if __name__ == "__main__":
    # Sistema di equazioni differenziali: dy/dt = f(t, y)
    def f(t, y):
        J = np.array([[-1, 0.01], [0.01, -16]])
        return J @ y

    # Parametri del metodo DIRK (ad esempio, metodo implicito di ordine 2)
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

    # Condizioni iniziali
    y0 = np.array([1.0, 1.0])
    t0 = 0.0
    t_end = 10.0
    dt = 1 / 16

    # Risolvi il sistema
    t, y = runge_kutta_dirk(f, y0, t0, t_end, dt, A, b, c)

    # Stampa i risultati
    for i in range(len(t)):
        print(f"t = {t[i]:.2f}, y = {y[i]}")

    import matplotlib.pyplot as plt

    # Grafico delle componenti insieme
    plt.figure(figsize=(10, 6))
    plt.plot(t, y[:, 0], label="y1 (prima componente)")
    plt.plot(t, y[:, 1], label="y2 (seconda componente)")
    plt.xlabel("Tempo t")
    plt.ylabel("Valore di y")
    plt.title("Componenti di y nel tempo")
    plt.legend()
    plt.grid()

    # Grafico delle componenti separate
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, y[:, 0], label="y1 (prima componente)", color="blue")
    plt.xlabel("Tempo t")
    plt.ylabel("y1")
    plt.title("Prima componente di y nel tempo")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(t, y[:, 1], label="y2 (seconda componente)", color="orange")
    plt.xlabel("Tempo t")
    plt.ylabel("y2")
    plt.title("Seconda componente di y nel tempo")
    plt.grid()
    plt.tight_layout()

    # Grafico nello spazio delle fasi
    plt.figure(figsize=(8, 6))
    plt.plot(y[:, 0], y[:, 1], label="Traiettoria nello spazio delle fasi")
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.title("Spazio delle fasi")
    plt.grid()
    plt.legend()
    plt.show()
