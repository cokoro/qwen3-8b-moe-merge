import json
import numpy as np
import sys
import scipy.optimize

def main():
    with open('./data/input_0.json') as f:
        input_0 = json.load(f)
    G = input_0['gravitational_constant']
    sky_sigma = input_0['sky_sigma']
    vlos_sigma = input_0['vlos_sigma']
    close_threshold = input_0['close_threshold']
    case_order = input_0['case_order']
    case_count = len(case_order)

    input_1 = np.load('./data/input_1.npy')  # shape (4,4)
    input_2 = np.load('./data/input_2.npy')  # shape (4,32)
    input_3 = np.load('./data/input_3.npy')  # shape (4,32,4,4)
    input_4 = np.load('./data/input_4.npy')  # shape (4,121)
    input_5 = np.load('./data/input_5.npy')  # shape (4,72,4,6)

    # Step 1: For each case, propagate prior particles to fit times
    # Step 2: At every fit time, compare predicted detections with detected rows only
    # Step 3: Convert particle likelihoods into normalized posterior weights
    # Step 4: Propagate posterior particles to all prediction times
    # Step 5: Compute label-free observable vector for each prediction time and particle
    # Step 6: Save posterior weighted quantiles for these observables
    # Step 7: Compute risk summary

    # Initialize posterior particles and weights
    posterior_particles = np.zeros((case_count, input_5.shape[1], input_5.shape[2], input_5.shape[3]))
    posterior_weights = np.zeros((case_count, input_5.shape[1], input_5.shape[2]))

    for case_id in range(case_count):
        case_masses = input_1[case_id]
        fit_times = input_2[case_id]
        input_3_case = input_3[case_id]
        input_4_case = input_4[case_id]
        input_5_case = input_5[case_id]

        # Propagate prior particles to fit times
        prior_particles = input_5_case.copy()
        for fit_time in fit_times:
            # Propagate particles to fit_time
            # Use Newtonian point-mass dynamics with G = input_0['gravitational_constant']
            # Update particles' positions and velocities
            # ...

        # Compute likelihood for each particle at each fit time
        # ...
        # Normalize posterior weights
        # ...

        # Propagate posterior particles to all prediction times
        # ...

        # Compute label-free observable vector for each prediction time and particle
        # ...

        # Save posterior weighted quantiles for these observables
        # ...

        # Compute risk summary
        # ...
        # Save risk_summary.csv

    # Save prediction_quantiles.npy
    # ...

    # Save risk_summary.csv
    # ...

if __name__ == '__main__':
    main()