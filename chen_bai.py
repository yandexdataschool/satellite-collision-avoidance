import numpy as np


def CheckValue(result, true, name, deviation=0.01):
    l = true * (1 - deviation)
    r = true * (1 + deviation)
    if (result < l) | (result > r):
        er = name + " = " + str(result) + " != " + str(true)
        raise ValueError(er)
    return


def TestChenBai(rA, miss_distance, dr_min, dt, nu, probability):
    # true values
    true_rA = 100.13  # meters
    true_miss_distance = 2.423292  # km
    true_dt = 0.161021
    true_dr_min = 2.182973  # km
    true_nu = 1.102111
    true_probability = 4.749411 * 10e-5

    CheckValue(rA, true_rA, "rA")
    CheckValue(miss_distance, true_miss_distance, "miss_distance")
    CheckValue(dt, true_dt, "dt")
    CheckValue(dr_min, true_dr_min, "dr_min")
    CheckValue(nu, true_nu, "nu")
    CheckValue(probability, true_probability, "probability")

    return


def ChenBai_coll_prob_estimation(rV1=np.ones(6), rV2=np.ones(6),
                                 sigma_1N=1, sigma_1T=1, sigma_1W=1,
                                 sigma_2N=1, sigma_2T=1, sigma_2W=1,
                                 cs_r1=100, cs_r2=0.1,
                                 test=False):
    # TODO - check units - m, km, time
    if test:
        # collision cross-section radii of ISS and the debris
        rV1 = np.array([
            3126.0188, 5227.1461, -2891.3029, -3.2980, 4.7587, 5.0543
        ])  # x,y,z - km; Vx,Vy,Vz - km/s
        rV2 = np.array([
            3124.3685, 5226.0042, -2889.9446, -7.7726, 1.9308, -2.7580
        ])  # x,y,z - km; Vx,Vy,Vz - km/s
        cs_r1 = 100  # meters
        cs_r2 = 0.13  # meters

        # sigma/km
        sigma_1N = 0.5548968
        sigma_1T = 6.185655
        sigma_1W = 1.9433925
        sigma_2N = 0.8717616
        sigma_2T = 12.306207
        sigma_2W = 0.9210618

    # r and V
    r1_vec = rV1[:3]
    r2_vec = rV2[:3]
    v1_vec = rV1[3:]
    v2_vec = rV2[3:]
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)
    v1 = np.linalg.norm(v1_vec)
    v2 = np.linalg.norm(v2_vec)
    dr0_vec = np.abs(r1_vec - r2_vec)
    dr0 = np.linalg.norm(dr0_vec)

    # combined collision cross-section radii
    rA = cs_r1 + cs_r2
    # miss distance
    miss_distance = dr0
    # nu
    nu = v2 / v1
    # dt
    temp = np.dot(v1_vec, v2_vec) / (v1 * v2)
    psi = np.arccos(temp)
    t1_min = nu * np.dot(dr0_vec, v1_vec) - np.cos(psi) * \
        np.dot(dr0_vec, v2_vec)
    t1_min /= v1 * v2 * np.sin(psi)**2
    t2_min = np.cos(psi) * np.dot(dr0_vec, v1_vec) - \
        np.dot(dr0_vec, v2_vec) / nu
    t2_min /= v1 * v2 * np.sin(psi)**2
    dt = abs(t2_min - t1_min)
    """
	!!! force true value
    """
    # dt = 0.161021
    # dr_min
    dr_min_vec = dr0_vec + v2_vec * t2_min - v1_vec * t1_min
    dr_min = np.linalg.norm(v2_vec)
    """
	!!! force true value
    """
    # dr_min = 2.182973
    # prob
    temp = 1 + nu**2 - 2 * nu * np.cos(psi)
    mu_x = dr_min
    mu_y = v2 * np.sin(psi * dt) / temp**0.5
    sigma_x = (sigma_1N**2 + sigma_2N**2)**0.5
    sigma_y = ((sigma_1T * nu * np.sin(psi))**2
               + ((1 - nu * np.cos(psi)) * sigma_1W)**2
               + (sigma_2T * np.sin(psi))**2
               + ((nu - np.cos(psi)) * sigma_2W)**2
               ) / temp

    probability = np.exp(
        -0.5 * ((mu_x / sigma_x)**2 + (mu_y / sigma_y)**2)
        * (1 - np.exp(-rA**2 / (2 * sigma_x * sigma_y)))
    )
    if test:
        TestChenBai(rA, miss_distance, dr_min, dt, nu, probability)
        print("test passed")
        return
    return probability

if __name__ == "__main__":
    ChenBai_coll_prob_estimation(test=True)
