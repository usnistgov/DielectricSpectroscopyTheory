import sys
import numpy as np
import matplotlib.pyplot as plt
from pulay import Pulay


def run(M):
    # Solve for several (dimensionless) electric fields and frequencies:
    N = 101  # number of times per period
    E = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    W = np.logspace(-2, 2, 81)
    P = np.zeros((len(W), len(E), N))
    M = 0
    for iW, Wi in enumerate(W):
        Pi = None
        Eprev = None
        for iE, Ei in enumerate(E):
            Esteps = [Ei]
            if Eprev is not None:
                # Step E in smaller intervals for stability if E is large:
                dEmax = 0.5
                n_steps = int(np.ceil((Ei - Eprev) / dEmax))
                Esteps = np.linspace(Eprev, Ei, n_steps+1)[1:]
            for Efine in Esteps:
                Pi = solve_dynamics(Efine, Wi, M, N, Pinitial=Pi)
            P[iW, iE] = Pi
            Eprev = Ei


    # Extract harmonics and plot susceptibilities:
    #   Perturbation is sin(WT)
    #   Need to extract coefficient of sin(nWT) for Re and cos(nWT) for Im.
    #   Numpy rfft extracts coefficients, say c_n of exp(inWT).
    #   Resulting fit corresponds to:
    #        c_n exp(inWT) + c_n* exp(-inWT)
    #      = (c_n + c_n*) cos(nWT) + i(c_n - c_n*) sin(nWT)
    #      = 2 Re(c_n) cos(nWT) - 2 Im(c_n) sin(nWT)
    #      = Im(2i c_n) cos(nWT) + Re(2i c_n) sin(nWT)
    #   Therefore: Re(2i rfft()) -> ReChi, Im(2i rfft()) -> ImChi.
    Ptilde = (np.fft.rfft(P, axis=-1, norm='forward') * 2j).conj()
    for harmonic in (1, 3, 5):
        chi = Ptilde[..., harmonic] / E[None, :] ** harmonic
        fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True)
        plt.subplots_adjust(hspace=0)
        header = ["W"]
        data = [W]
        for ax, comp, comp_name in zip(axes, (np.real, np.imag), ("Re", "Im")):
            plt.sca(ax)
            for Ei, chi_i in zip(E, comp(chi.T)):
                plt.plot(W, chi_i, label=f'$\mathcal{{E}} = {Ei}$')
                header.append(f"{comp_name}(E={Ei})")
                data.append(chi_i)
            plt.legend()
            plt.ylabel(f"{comp_name}$\chi_{{{harmonic}}}/\chi_0$")
            plt.xlabel("$\mathcal{W}$")
            plt.xscale("log")
            plt.xlim(W.min(), W.max())
            plt.axhline(0, color='k', ls='dotted', lw=1)
            plt.axvline(1, color='k', ls='dotted', lw=1)
        prefix = f"chi{harmonic}" + (f"_M{M:g}" if M else "")
        plt.savefig(f"{prefix}.pdf", bbox_inches="tight")
        np.savetxt(f"{prefix}.dat", np.array(data).T, header=" ".join(header))
    
    # Plot field patterns for a subset of frequencies:
    WT = np.arange(N) * (2 * np.pi / N)
    for Wi, Pw in zip(W[::20], P[::20]):
        plt.figure()
        header = ["WT"]
        data = [WT]
        for Ei, Pwe in zip(E, Pw):
            plt.plot(WT, Pwe, label=f'$\mathcal{{E}} = {Ei}$')
            header.append(f"P(E={Ei})")
            data.append(Pwe)
        plt.legend()
        for x_line in np.arange(1, 4)*0.5*np.pi:
            plt.axvline(x_line, color='k', ls='dotted', lw=1)
        plt.xlim(0, 2*np.pi)
        plt.xticks(
            np.arange(5)*0.5*np.pi,
            ["$0$", "$\pi/2$", "$\pi$", "$3\pi/2$", "$2\pi$"]
        )
        plt.xlabel("$\mathcal{WT}$")
        plt.ylabel("$\mathcal{P}$")
        plt.axhline(0, color='k', ls='dotted', lw=1)
        plt.title(f"$\mathcal{{W}} = {Wi}$")
        prefix = f"pol_W{Wi}" + (f"_M{M:g}" if M else "")
        plt.savefig(f"{prefix}.pdf", bbox_inches="tight")
        np.savetxt(f"{prefix}.dat", np.array(data).T, header=" ".join(header))
    plt.show()


def solve_dynamics(E, W, M=0., N=100, Pinitial=None):
    """
    Solve dynamics for non-dimensional field amplitude E and frequency W.
    Optionally include an inertia term parameterized by dimensionless M, which
    sets the resonant frequency at 1/sqrt(M) relative to the Debye frequency.
    Discretize one period of the oscillation with N time points,
    and solve for the steady state using periodic boundary conditions.
    Return the non-dimensional polarization over one period.
    """
    WT = np.arange(N) * (2 * np.pi / N)
    Eprofile = E * np.sin(WT)  # applied electric field profile over one period
    Wsq = W * W
    if Pinitial is None:
        Pinitial = langevin((3j * E * np.exp(-1j * WT) / (1  - 1j * W - M * Wsq)).real)
    
    # Construct differential operator for dP/dT
    w = np.concatenate((np.arange((N+1)//2), np.arange((N+1)//2 - N, 0))) * W
    if N % 2 == 0:
        w[N//2] = 0.0  # zero out Nyquist component
    Dtilde = 1j * w
    if M:
        Dtilde -= M * w * w  # Add M d^2/dT^2 to d/dT, if needed

    print(f"{E = :.2E} {W=:.2E} Residual:", end=" ", flush=True)
    dynamics = Dynamics(Eprofile, Dtilde, Pinitial)
    converged = dynamics.optimize()
    # assert converged
    print("done.", flush=True)
    return dynamics.P


class Dynamics(Pulay):
    def __init__(self, Eprofile, Dtilde, Pinitial) -> None:
        super().__init__(
            n_iterations=200,
            residual_threshold=1e-6,
            n_history=15,
            mix_fraction=0.5
        )
        self.Eprofile = Eprofile
        self.Dtilde = Dtilde
        self.P = Pinitial
        self.K = 1.0 / (1.0 + Dtilde)  # preconditioner
        self.M = abs(1.0 + Dtilde) ** 2  # DIIS metric
    
    def cycle(self) -> None:
        Eeff = self.Eprofile - np.fft.ifft(self.Dtilde * self.Ptilde).real
        self.P = langevin(3 * Eeff)
    
    @property
    def P(self) -> np.ndarray:
        return np.fft.ifft(self.Ptilde).real

    @P.setter
    def P(self, Pnew: np.ndarray) -> None:
        self.Ptilde = np.fft.fft(Pnew)

    @property
    def variable(self) -> np.ndarray:
        return self.Ptilde

    @variable.setter
    def variable(self, v: np.ndarray) -> None:
        self.Ptilde = v

    def precondition(self, v: np.ndarray) -> np.ndarray:
        return self.K * v

    def metric(self, v: np.ndarray) -> np.ndarray:
        return self.M * v


def langevin(x, prime=False):
    """Compute langevin function, and optinally its derivative if prime=True."""
    x_cut = 0.1

    # Direct evaluation for large enough x
    mask = (abs(x) >= x_cut)
    result = np.reciprocal(np.tanh(x), where=mask) - np.reciprocal(x, where=mask)
    if prime:
        result_prime = (
            np.reciprocal(x, where=mask) ** 2
            - np.reciprocal(np.sinh(x), where=mask) ** 2
        )

    # Continued fraction expansion for small x (=x/(3 + x^2/(5 + x^2/...)))
    sel = np.where(abs(x) < x_cut)
    if len(sel[0]):
        x_sel = x[sel]
        x_sel_sq = x_sel * x_sel
        r = np.zeros_like(x_sel)
        if prime:
            r_prime = np.zeros_like(x_sel)
        for c in (9.0, 7.0, 5.0, 3.0):
            den = 1.0 / (c + x_sel * r)
            r = x_sel * den
            if prime:
                r_prime = den * den * (c - x_sel_sq * r_prime)
        result[sel] = r
        if prime:
            result_prime[sel] = r_prime
    
    return (result, result_prime) if prime else result


if __name__ == "__main__":
    M = float(sys.argv[1]) if (len(sys.argv) > 1) else 0.0
    run(M)
