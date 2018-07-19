# Module utils provides necessary utils for API module.

import numpy as np
from scipy.stats import norm


class CollProbEstimator:

    """ Estimate probability of collision between two objects. """

    def ChenBai_approach(rV1, rV2,
                         cs_r1=100, cs_r2=0.1,
                         sigma_1N=50, sigma_1T=50, sigma_1W=50,
                         sigma_2N=300, sigma_2T=300, sigma_2W=300):
        """ Returns probability of collision between two objects.

        Lei Chen, Xian-Zong Bai, Yan-Gang Liang, Ke-Bo Li
        closest approach between objects from "Orbital Data Applications for Space Objects".

        Args:
            rV1, rV2 (np.array([x, y, z, Vx, Vy, Vz])): objects coordinates (meters) and velocities (m/s).
            cs_r1, cs_r2 (float): objects cross-section radii (meters).
            sigma_N, sigma_T, sigma_W (float): objects positional error standard deviations
                in normal, tangential, cross-track direction (meters).

        Note:
            The default args imply the first object is protected satellite
            and other is debris.

        Returns:
            float: probability.

        """

        r1_vec = rV1[:3]
        r2_vec = rV2[:3]
        v1_vec = rV1[3:]
        v2_vec = rV2[3:]

        r1 = np.linalg.norm(r1_vec)
        r2 = np.linalg.norm(r2_vec)
        v1 = np.linalg.norm(v1_vec)
        v2 = np.linalg.norm(v2_vec)
        dr0_vec = r2_vec - r1_vec
        dr0 = np.linalg.norm(dr0_vec)

        rA = cs_r1 + cs_r2  #: Combined collision cross-section radii
        miss_distance = dr0
        nu = v2 / v1
        #: The angle w between two velocities
        if np.array_equal(v1_vec, v2_vec):
            psi = 0
        else:
            psi = np.arccos(np.dot(v1_vec, v2_vec) / (v1 * v2))
        epsilon = 10e-7
        temp = v1 * v2 * np.sin(psi)**2
        if temp == 0:
            temp += epsilon
        t1_min = (nu * np.dot(dr0_vec, v1_vec) - np.cos(psi) *
                  np.dot(dr0_vec, v2_vec)) / temp
        t2_min = (np.cos(psi) * np.dot(dr0_vec, v1_vec) -
                  np.dot(dr0_vec, v2_vec) / nu) / temp
        #: Crossing time difference of the common perpendicular line (sec)
        dt = abs(t2_min - t1_min)
        #: The relative position vector at closest approach between orbits
        dr_min_vec = dr0_vec + v2_vec * t2_min - v1_vec * t1_min
        #: The closest approach distance between two straight line trajectories
        dr_min = np.linalg.norm(dr_min_vec)
        # Probability.
        temp = 1 + nu**2 - 2 * nu * np.cos(psi)
        if temp == 0:
            temp += epsilon
        mu_x = dr_min
        mu_y = v2 * np.sin(psi) * dt / temp**0.5
        sigma_x_square = sigma_1N**2 + sigma_2N**2
        sigma_y_square = ((sigma_1T * nu * np.sin(psi))**2
                          + ((1 - nu * np.cos(psi)) * sigma_1W)**2
                          + (sigma_2T * np.sin(psi))**2
                          + ((nu - np.cos(psi)) * sigma_2W)**2
                          ) / temp
        if sigma_x_square == 0:
            sigma_x_square += epsilon
        if sigma_y_square == 0:
            sigma_y_square += epsilon
        probability = np.exp(
            -0.5 * (mu_x**2 / sigma_x_square + mu_y**2 / sigma_y_square)
        ) * (1 - np.exp(-rA**2 / (2 * (sigma_x_square * sigma_y_square)**0.5)))
        return probability

    def norm_approach(rV1, rV2, sigma=50):
        """ Returns probability of collision between two objects.

        Args:
            rV1, rV2 (np.array([x, y, z, Vx, Vy, Vz])): objects coordinates (meteres) and velocities (m/s).
            d1, d2 (float, float): objects size (meters).
            sigma (float): standard deviation.

        Returns:
            float: probability.

        Raises:
            ValueError: If any probability has incorrect value.
            TypeError: If any probability has incorrect type.

        """
        probability = 1.
        for c1, c2 in zip(rV1[:3], rV2[:3]):
            av = (c1 + c2) / 2.
            integtal = norm.cdf(av, loc=min(c1, c2), scale=sigma)
            probability *= (1 - integtal) / integtal

        return probability
