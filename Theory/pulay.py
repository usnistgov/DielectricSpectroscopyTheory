from __future__ import annotations
import numpy as np
import numpy as np
from collections import deque
from abc import ABC, abstractmethod


class Pulay(ABC):
    """Abstract base class implementing the Pulay mixing algorithm.
    (Simplfied from qimpy.utils.Pulay)."""

    def __init__(
        self,
        *,
        n_iterations: int,
        residual_threshold: float,
        n_history: int,
        mix_fraction: float,
    ) -> None:
        """Initialize Pulay algorithm parameters."""
        super().__init__()
        self.n_iterations = n_iterations
        self.residual_threshold = residual_threshold
        self.n_history = n_history
        self.mix_fraction = mix_fraction
        self._variables = deque(maxlen=n_history)
        self._residuals = deque(maxlen=n_history)
        self._overlaps = np.zeros((0, 0), dtype=float)

    @abstractmethod
    def cycle(self) -> None:
        """Single cycle of the Pulay-mixed self-consistent iteration.
        In each subsequent cycle, Pulay will try to zero the difference
        between get_variable() before and after the cycle.
        """

    @property  # type: ignore
    @abstractmethod
    def variable(self) -> np.ndarray:
        """Current variable in the state of the system."""

    @variable.setter  # type: ignore
    @abstractmethod
    def variable(self, v: np.ndarray) -> None:
        ...

    @property
    def residual(self) -> np.ndarray:
        """Get the current residual from state of system (read-only).
        Override this only if this Pulay mixing is not for a self-consistent
        iteration i.e. the residual is not the change of `variable`.
        """
        return self.variable - self._variables[-1]

    @abstractmethod
    def precondition(self, v: np.ndarray) -> np.ndarray:
        """Apply preconditioner to variable/residual."""

    @abstractmethod
    def metric(self, v: np.ndarray) -> np.ndarray:
        """Apply metric to variable/residual."""

    def optimize(self) -> bool:
        """Minimize residual using a Pulay-mixing / DIIS algorithm."""

        # Cleanup previous history (if any):
        self._variables.clear()
        self._residuals.clear()
        self._overlaps = np.zeros((0, 0), dtype=float)

        for i_iter in range(self.n_iterations):
            # Cache variable:
            self._variables.append(self.variable)

            # Perform cycle:
            self.cycle()

            # Cache residual:
            residual = self.residual
            Mresidual = self.metric(residual)
            res_norm = np.sqrt(np.vdot(residual, Mresidual).real)
            self._residuals.append(residual)

            # Check and report convergence:
            print(f"{res_norm:.0E}", end=" ", flush=True)
            if res_norm < self.residual_threshold:
                return True

            # Pulay mixing / DIIS step:
            # --- update the overlap matrix
            new_overlaps = np.array([np.vdot(r, Mresidual).real for r in self._residuals])
            N = len(new_overlaps)
            self._overlaps = np.vstack((
                np.hstack((
                    self._overlaps[-(N - 1) :, -(N - 1) :],
                    new_overlaps[:-1, None],
                )),
                new_overlaps[None, :],
            ))
            # --- invert overlap matrix to minimize residual
            overlapC = np.ones((N + 1, N + 1))  # extra row/col for norm constraint
            overlapC[:-1, :-1] = self._overlaps
            overlapC[-1, -1] = 0.0
            alpha = np.linalg.inv(overlapC)[N, :-1]  # optimum coefficients
            # --- update variable
            v = 0.0 * residual
            for i_hist, alpha_i in enumerate(alpha):
                v += alpha_i * (
                    self._variables[i_hist]
                    + self.mix_fraction * self.precondition(self._residuals[i_hist])
                )
            self.variable = v  # type: ignore
            
        return False  # convergence failed 
