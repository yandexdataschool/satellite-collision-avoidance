import numpy as np


def CheckValue(result, true, name, deviation=0.01):
    l = true * (1 - deviation)
    r = true * (1 + deviation)
    if (result < l) | (result > r):
        er = name + " = " + str(result) + " != " + str(true)
        raise ValueError(er)
    print(name + " is correct")
    return


def TestChenBai(rA, miss_distance, dr_min, dt, nu, probability):
    # true values
    true_rA = 100.13  # meters
    true_miss_distance = 2423.292  # meters
    true_dt = 0.161021  # sec
    true_dr_min = 2182.973  # m
    true_nu = 1.102111
    true_probability = 4.749411e-5

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
    if test:
        # collision cross-section radii of ISS and the debris
        r1 = np.array([3126018.8, 5227146.1, -2891302.9])  # meters
        r2 = np.array([3124368.5, 5226004.2, -2889944.6])  # meters
        V1 = np.array([-3298.0, 4758.7, 5054.3])  # m/s
        V2 = np.array([-7772.6, 1930.8, -2758.0])  # m/s
        rV1 = np.hstack([r1, V1])
        rV2 = np.hstack([r2, V2])
        # sizes
        cs_r1 = 100  # meters
        cs_r2 = 0.13  # meters

        # sigma/ m
        sigma_1N = 554.8968
        sigma_1T = 6185.655
        sigma_1W = 1943.3925
        sigma_2N = 871.7616
        sigma_2T = 12306.207
        sigma_2W = 921.0618

    # r and V
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

    # combined collision cross-section radii
    rA = cs_r1 + cs_r2
    # miss distance
    miss_distance = dr0
    # nu
    nu = v2 / v1
    # dt
    psi = np.arccos(np.dot(v1_vec, v2_vec) / (v1 * v2))
    temp = v1 * v2 * np.sin(psi)**2
    t1_min = (nu * np.dot(dr0_vec, v1_vec) - np.cos(psi) *
              np.dot(dr0_vec, v2_vec)) / temp
    t2_min = (np.cos(psi) * np.dot(dr0_vec, v1_vec) -
              np.dot(dr0_vec, v2_vec) / nu) / temp
    dt = abs(t2_min - t1_min)
    # dr_min
    dr_min_vec = dr0_vec + v2_vec * t2_min - v1_vec * t1_min
    dr_min = np.linalg.norm(dr_min_vec)
    # prob
    temp = 1 + nu**2 - 2 * nu * np.cos(psi)
    mu_x = dr_min
    mu_y = v2 * np.sin(psi) * dt / temp**0.5
    sigma_x_square = sigma_1N**2 + sigma_2N**2
    sigma_y_square = ((sigma_1T * nu * np.sin(psi))**2
                      + ((1 - nu * np.cos(psi)) * sigma_1W)**2
                      + (sigma_2T * np.sin(psi))**2
                      + ((nu - np.cos(psi)) * sigma_2W)**2
                      ) / temp

    probability = np.exp(
        -0.5 * (mu_x**2 / sigma_x_square + mu_y**2 / sigma_y_square)
    ) * (1 - np.exp(-rA**2 / (2 * (sigma_x_square * sigma_y_square)**0.5)))
    if test:
        TestChenBai(rA, miss_distance, dr_min, dt, nu, probability)
        print("test passed")
        return
    return probability
import unittest
if __name__ == "__main__":
    ChenBai_coll_prob_estimation(test=True)
