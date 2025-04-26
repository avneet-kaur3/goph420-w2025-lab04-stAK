from goph420_lab04.regression import multi_regress
import numpy as np
import matplotlib.pyplot as plt

def main():

    """from the file provided in the lab assignment resources, M_data_raw1.tif shows the magnitudes to be concentrated in the interval [-0.5, 0.5]. based on that, the 
    magnitudes for which the number of events N that can be calculated are chosen to be -0.5, -0.25, 0.0, 0.25 and 0.5."""

    #importing data from the text file.
    data = np.loadtxt("M_data1.txt")
    t_data = data[:, 0]
    M_data = data[:, 1]

    #choosing time intervals of 12 hours over 120 hours. this will return 10 data points of the coefficients a and b.
    time_intervals = [(i, i + 12) for i in range(0, 120, 12)]
    magnitude_points = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    #definining an empty list for storing M, N and the values from the linear regression function for each time interval. the time intervals are 12 hours.
    trend_data = []
    for start, end in time_intervals:
        time_indices = np.argwhere((t_data >= start) and (t_data < end)).flatten()
        M_window = M_data[time_indices]
        N_window = [np.count_nonzero(M_window >= threshold) for threshold in magnitude_points]
        log_N_window = np.log(np.array(N_window) + 1e-5) #the factor of 1e-5 ensures stability with zero errors.

        plt.figure(figsize=(10, 8))
        plt.plot(magnitude_points, N_window)
        plt.title("N versus M for time interval: {start} to {end} hours")
        plt.xlabel("Magnitude M")
        plt.ylabel("Number of Events N")
        plt.grid(True)
        plt.savefig(f"dataplot_{start}_{end}.png", dpi=300)
        plt.show()

        #performing the multiple linear regression function.
        y = magnitude_points
        Z = log_N_window
        a, e, rsq = multi_regress(y, Z)

        #appending the data to our original empty list to store the different values of coefficients.
        trend_data.append((start, end, a[0], a[1], e, rsq))

        #plotting the linear data from the text file and the calculated best fit line.
        plt.figure(figsize=(10, 8))
        plt.scatter(magnitude_points, log_N_window, label="Observed Data", color="blue")
        plt.plot(magnitude_points, a[0]+(magnitude_points*a[1]), label="Best Fit Line", color="red")
        plt.title("log N versus M for time interval: {start} to {end} hours")
        plt.xlabel("Magnitude M")
        plt.ylabel("log(N)")
        plt.grid(True)
        plt.legend()
        saving_directory = "C:/Users/HP/Desktop/University Courses/Winter 2025/GOPH 420/goph420-w2025-lab04-stAK/goph420-w2025-lab04-stAK/figures/"
        plt.savefig(f"{saving_directory}bestfitplot_{start}_{end}.png", dpi=300)
        plt.show()

    trend_data = np.array(trend_data)
    print("Time Interval | a (Intercept) | b (Gradient) | e Residual | R^2")
    print(trend_data)


if __name__ == "__main__":
    main()