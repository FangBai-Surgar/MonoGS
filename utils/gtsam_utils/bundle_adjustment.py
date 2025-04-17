
"""

Install GTSAM: 

    pip install gtsam

"""


import matplotlib.pyplot as plt
import numpy as np
import gtsam


# from gtsam import symbol_shorthand
# from gtsam import (Cal3_S2, DoglegOptimizer, GenericProjectionFactorCal3_S2,
#                    Marginals, NonlinearFactorGraph, PinholeCameraCal3_S2,
#                    PriorFactorPoint3, PriorFactorPose3, Values)

# from gtsam.Values import Values

from gtsam.examples import SFMdata
from gtsam.utils import plot




def bundle_adjustment(kpt_measurements, poses_w2c, points, K, compute_marginals = False, plot_figure = False):
    """
        Parameters:
            kpt_measurements   ({{}}):  [i][j] = np.ndarray 2D keypoint position of j-th landmark in i-th pose
            poses      ({np.ndarray}):  [i] = i-th pose   : W2C (from world to camera)
            points     ({np.ndarray}):  [j] = j-th point
            K            (np.ndarray):  shape(3,3). calibration intrinsic matrix

        Returns:
            opt_poses  ({np.ndarray}):  [i] = i-th pose   : W2C (from world to camera)
            opt_points ({np.ndarray}):  [j] = j-th point

        Remarks:
            graph                   :  gtsam.NonlinearFactorGraph()
            result                  :  gtsam.Values()
    """

    L = gtsam.symbol_shorthand.L
    X = gtsam.symbol_shorthand.X

    fx, fy, s, u0, v0 = K[0,0], K[1,1], K[0,1], K[0,2], K[1,2]
    intrinsic_K = gtsam.Cal3_S2(fx, fy, s, u0, v0)

    # PoseDirection in GTSAM          : From Camera to World
    # Code Pieces:
    #         // gtsam/gtsam/geometry/CalibratedCamera.cpp
    #         const Point3 q = pose().transformTo(point)   # from World to Camera
    #         // gtsam/gtsam/geometry/Pose3.cpp
    #         const Matrix3 Rt = R_.transpose();
    #         const Point3 q(Rt*(point - t_));
    poses_c2w = {}
    for i, w2c in poses_w2c.items():
        poses_c2w[i] = gtsam.Pose3( np.linalg.inv( w2c ) )


    # Create a factor graph
    graph = gtsam.NonlinearFactorGraph()

    # Add a prior on pose x1. This indirectly specifies where the origin is.
    # 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
    for i, pose in poses_c2w.items():
        factor = gtsam.PriorFactorPose3(X(i), pose, pose_noise)
        graph.push_back(factor)
        break

    # Define the camera observation noise model
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v

    # measurements from each camera pose, adding them to the factor graph
    for i, kpt_measurements_per_image in kpt_measurements.items():
        for j, measurement in kpt_measurements_per_image.items():
            factor = gtsam.GenericProjectionFactorCal3_S2(measurement, measurement_noise, X(i), L(j), intrinsic_K)
            graph.push_back(factor)

    # Because the structure-from-motion problem has a scale ambiguity, the problem is still under-constrained
    # Here we add a prior on the position of the first landmark. This fixes the scale by indicating the distance
    # between the first camera and the first landmark. All other landmark positions are interpreted using this scale.
    point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    for j, point in points.items():
        factor = gtsam.PriorFactorPoint3(L(j), gtsam.Point3(point), point_noise)
        graph.push_back(factor)
        break
    # graph.print("Factor Graph:\n")

    # Create the data structure to hold the initial estimate to the solution
    initial_estimate = gtsam.Values()
    for i, pose in poses_c2w.items():
        initial_estimate.insert(X(i), pose)
    for j, point in points.items():
        initial_estimate.insert(L(j), point)
    # initial_estimate.print("Initial Estimates:\n")

    # Optimize the graph and print results
    params = gtsam.DoglegParams()
    params.setVerbosity("TERMINATION")
    optimizer = gtsam.DoglegOptimizer(graph, initial_estimate, params)
    print("GTSAM Optimizing:")
    result = optimizer.optimize()
    # result.print("Final results:\n")
    print("\tinitial error = {}".format(graph.error(initial_estimate)))
    print("\tfinal error   = {}".format(graph.error(result)))

    # mariginal covariance
    marginals = gtsam.Marginals(graph, result) if compute_marginals else None
    if plot_figure:
        plot.plot_3d_points(1, result, marginals=marginals)
        plot.plot_trajectory(1, result, marginals=marginals, scale=8)
        plot.set_axes_equal(1)
        plt.show()

    # extract arrays
    opt_poses_w2c, opt_points = {}, {}
    for i, pose in poses_c2w.items():
        c2w = result.atPose3 ( X(i) ).matrix()      # [ [R, t],  [0,0,0,1] ]. #numpy.ndarray, shape(4,4)
        opt_poses_w2c[i] =  np.linalg.inv( c2w )

    for j, point in points.items():
        opt_points[j] =  result.atPoint3 ( L(j) )   #nnumpy.ndarray  shape(3)

    return opt_poses_w2c, opt_points




if __name__ == "__main__":

    """
    Camera observations of landmarks (i.e. pixel coordinates) will be stored as Point2 (x, y).

    Each variable in the system (poses and landmarks) must be identified with a unique key.
    We can either use simple integer keys (1, 2, 3, ...) or symbols (X1, X2, L1).
    Here we will use Symbols

    In GTSAM, measurement functions are represented as 'factors'. Several common factors
    have been provided with the library for solving robotics/SLAM/Bundle Adjustment problems.
    Here we will use Projection factors to model the camera's landmark observations.
    Also, we will initialize the robot at some location using a Prior factor.

    When the factors are created, we will add them to a Factor Graph. As the factors we are using
    are nonlinear factors, we will need a Nonlinear Factor Graph.

    Finally, once all of the factors have been added to our factor graph, we will want to
    solve/optimize to graph to find the best (Maximum A Posteriori) set of variable values.
    GTSAM includes several nonlinear optimizers to perform this step. Here we will use a
    trust-region method known as Powell's Dogleg

    The nonlinear solvers within GTSAM are iterative solvers, meaning they linearize the
    nonlinear functions around an initial linearization point, then solve the linear system
    to update the linearization point. This happens repeatedly until the solver converges
    to a consistent set of variable values. This requires us to specify an initial guess
    for each variable, held in a Values container.
    """

    # Define the camera calibration parameters

    fx, fy, s, u0, v0 = 50.0, 50.0, 0.0, 50.0, 50.0
    intrK = gtsam.Cal3_S2(fx, fy, s, u0, v0)
    K = np.array( [ [fx, 0, u0],
                    [0, fy, v0],
                    [0, 0, 1  ] ] )


    # Create the set of ground-truth landmarks
    points = SFMdata.createPoints()

    # Create the set of ground-truth poses
    poses = SFMdata.createPoses(intrK)

    # Simulated measurements from each camera pose, adding them to the factor graph
    kpt_measurements = {}
    for i, pose in enumerate(poses):
        kpt_measurements[i] = {}
        camera = gtsam.PinholeCameraCal3_S2(pose, intrK)
        for j, point in enumerate(points):
            measurement = camera.project(point)
            kpt_measurements[i][j] = measurement
            print(f" measurement ({i}, {j}) = {measurement}, {type(measurement)}")

    # convert c2w to w2c
    # print(poses)
    W2C = []
    for pose in poses:
        T = np.linalg.inv(pose.matrix())
        W2C.append(T)
    # print(W2C)
    
    # perform BA with given calibration K
    opt_poses, opt_points = bundle_adjustment(kpt_measurements,
                                              dict( enumerate(W2C) ),
                                              dict( enumerate(points) ),
                                              K,
                                              True, True)





