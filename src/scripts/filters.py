"""
One Euro Filter implementation for smoothing noisy signals.
Based on "The One Euro Filter" (Casiez et al., 2012).
"""
import math
import time
import numpy as np


class OneEuroFilter:
    """Low-pass filter with adaptive cutoff frequency for reducing jitter."""

    def __init__(self, freq=120.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        Initialize One Euro Filter.

        Args:
            freq: Expected update frequency (Hz)
            min_cutoff: Minimum cutoff frequency
            beta: Cutoff slope (speed coefficient)
            d_cutoff: Cutoff frequency for derivative
        """
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, freq):
        """Calculate smoothing factor from cutoff frequency."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    @staticmethod
    def _exp_smooth(a, x, x_prev):
        """Exponential smoothing."""
        return a * x + (1.0 - a) * x_prev

    def reset(self):
        """Reset filter state."""
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t=None):
        """
        Filter a new value.

        Args:
            x: New value to filter
            t: Timestamp (uses current time if None)

        Returns:
            Filtered value
        """
        now = time.time() if t is None else t

        # Initialize on first call
        if self.t_prev is None:
            self.t_prev = now
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        # Estimate frequency from timestamps
        dt = max(1e-6, now - self.t_prev)
        freq = 1.0 / dt

        # Filter the derivative of the signal
        dx = (x - self.x_prev) * freq
        a_d = self._alpha(self.d_cutoff, freq)
        dx_hat = self._exp_smooth(a_d, dx, self.dx_prev)

        # Filter the signal with adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, freq)
        x_hat = self._exp_smooth(a, x, self.x_prev)

        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = now

        return x_hat


class OneEuro2D:
    """One Euro Filter for 2D points (x, y coordinates)."""

    def __init__(self, freq=120.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """Initialize with separate filters for x and y coordinates."""
        self.fx = OneEuroFilter(freq, min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(freq, min_cutoff, beta, d_cutoff)

    def reset(self):
        """Reset both x and y filters."""
        self.fx.reset()
        self.fy.reset()

    def __call__(self, pt, t=None):
        """
        Filter a 2D point.

        Args:
            pt: (x, y) tuple or array
            t: Timestamp (uses current time if None)

        Returns:
            Filtered (x, y) as numpy array
        """
        x, y = pt
        return np.array([self.fx(x, t), self.fy(y, t)], dtype=float)