import numpy as np
import torch
import gpytorch
import plotly.graph_objs as go
from scipy.optimize import minimize
import virtual_lab as vl
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from sklearn.decomposition import PCA
import conditions_data as data
from scipy.optimize import NonlinearConstraint

# Copy this to import all the functions:
# ExactGPModel, standardize_data, unstandardize_y, train_gp_model, expected_improvement, summed_feeding, optimize_acquisition_function, plot_data, plot_acquisition_function

# GP Model Definition
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, dims):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dims))
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Data Standardization
def standardize_data(X, y=None):
    """Standardizes input X and output y (if provided)."""
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_standardized = (X - X_mean) / X_std

    if y is not None:
        y = np.array(y)
        y_mean, y_std = y.mean(), y.std()
        y_standardized = (y - y_mean) / y_std
        return X_standardized, y_standardized, X_mean, X_std, y_mean, y_std
    return X_standardized, X_mean, X_std

def unstandardize_y(y_standardized, y_mean, y_std):
    """Unstandardizes predictions to original scale."""
    return y_standardized * y_std + y_mean

# Train GP Model
def train_gp_model(train_x, train_y, learning_rate=0.2, training_iter=500, dims=2):
    """Trains a GP model using standardized data."""
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood, dims)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        model.train()
        likelihood.train()
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    return model, likelihood

# EI Acquisition Function
def expected_improvement(model, X, best_f):
    """Calculates Expected Improvement (EI)."""
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model(X)
        mean = posterior.mean
        std = posterior.variance.sqrt()
        improvement = mean - best_f
        Z = improvement / std
        ei = improvement * torch.distributions.Normal(0, 1).cdf(Z) + std * torch.distributions.Normal(0, 1).log_prob(Z).exp()
        ei[std == 0.0] = 0.0  # Avoid division by zero
    return ei

# Define the feeding constraint
def summed_feeding(x, X_mean, X_std):
    return sum(x[2:5] * X_std[2:5] + X_mean[2:5])

# Optimization Routine of Acquisition function
def optimize_acquisition_function(af_function, bounds, X_mean, X_std, dtype, method='differential_evolution', n_restarts=10, random_seed=None):
    """
    Finds the point that maximizes the acquisition function (EI).
    
    Parameters:
        ei_function (callable): Acquisition function to maximize.
        bounds (list of tuples): Bounds for the search space [(low1, high1), (low2, high2), ...].
        method (str): Optimization method ('random', or 'differential_evolution').
        n_restarts (int): Number of restarts for random/global methods.
        random_seed (int, optional): Seed for reproducibility in random methods.
    
    Returns:
        ndarray: Optimal input parameters.
    """
    np.random.seed(random_seed)

    if method == 'random':
        # Random search for global optimization
        best_x = None
        best_ei = -np.inf
        for _ in range(n_restarts):
            candidate = np.array([np.random.uniform(low, high) for low, high in bounds])
            ei_value = af_function(torch.tensor(candidate, dtype=dtype).unsqueeze(0)).item()
            if ei_value > best_ei:
                best_x = candidate
                best_ei = ei_value
        print(f"Random search EI value: {best_ei}")
        return best_x
    
    elif method == 'differential_evolution':

        # Create a nonlinear constraint object
        feeding_nonlinear_constraint = NonlinearConstraint(
            lambda x: summed_feeding(x, X_mean, X_std), 0, 50  # Lower bound 0, upper bound 50
        )
        
        # Global optimization with differential evolution
        result = differential_evolution(
            lambda x: -af_function(torch.tensor(x, dtype=dtype).unsqueeze(0)).item(),
            bounds=bounds,
            seed=random_seed,
            popsize=50,
            constraints=feeding_nonlinear_constraint,
        )

        # Refinement of solution with COBYLA
        result_local = minimize(
            lambda x: -af_function(torch.tensor(x, dtype=dtype).unsqueeze(0)).item(),
            x0=result.x,
            bounds=bounds,
            method='COBYLA',
            constraints=feeding_nonlinear_constraint
        )
        # print(f"Differential Evolution + COBYLA EI value: {-result.fun}")
        return result_local.x, result_local.fun

    else:
        raise ValueError("Invalid method. Choose 'random', or 'differential_evolution'.")


def plot_data(Xtrain, ytrain, model, likelihood, X_mean, X_std, y_mean, y_std,
                     iteration, new_point=None, do_dim_reduction=False, projection='2d', fun_max=None):
    """
    Plots the GP mean surface and training points using matplotlib.
    
    Parameters:
        do_dim_reduction (bool): If True, performs dimensionality reduction (PCA or t-SNE) and plots in reduced space.
    """

    if do_dim_reduction:

        # Unstandardize training points
        Xtrain_unstandardized = Xtrain.numpy() * X_std + X_mean
        ytrain_unstandardized = unstandardize_y(ytrain.numpy(), y_mean, y_std)

        # Perform dimensionality reduction (PCA)
        dim_reducer = PCA(n_components=2)
        Xtrain_reduced = dim_reducer.fit_transform(Xtrain_unstandardized)
        
        
        if projection == '3d':
            # Create plot
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Add training points
            scatter = ax.scatter(
                Xtrain_reduced[:, 0], Xtrain_reduced[:, 1], ytrain_unstandardized,
                c=(ytrain_unstandardized), cmap='viridis', label='Training Points', zorder=1
            )

            # Highlight the maximum value
            max_value_idx = np.argmax(ytrain_unstandardized)
            ax.scatter(Xtrain_reduced[max_value_idx, 0], Xtrain_reduced[max_value_idx, 1],ytrain_unstandardized[max_value_idx],
                        color='blue', s=70, label='Maximum Value', zorder=10)
            
            if fun_max is not None:
                y_max = vl.conduct_experiment(fun_max)
                reduced_fun_max = dim_reducer.transform(fun_max)
                ax.scatter(
                    reduced_fun_max[:,0], reduced_fun_max[:,1], y_max,
                    color='pink', s=70, label='Known Maximum', edgecolor='black' 
                )

            # Optionally add new points
            if new_point is not None:
                new_X, new_y = new_point
                new_X = dim_reducer.transform(new_X)
                for i in range(len(new_X)):
                    ax.scatter(
                        [new_X[i][0]], [new_X[i][1]], [new_y[i]],
                        color='red', label='New Points' if i == 0 else None, s=60, zorder=10
                    )

                    # Add text label with point number next to it
                    ax.text(
                        new_X[i][0], new_X[i][1], new_y[i],  # Use x, y, and z coordinates for the label
                        str(i+1),  # Ensure the text is passed as a string
                        color='black', fontsize=12, ha='left', va='bottom'
                    )

            # Labels and legend
            ax.set_title(f'Bayesian Optimization Iteration: {iteration}', fontsize=16)
            ax.set_xlabel('PC 1', fontsize=12)
            ax.set_ylabel('PC 2', fontsize=12)
            ax.set_zlabel('P (g/L)', fontsize=12)
            ax.legend(loc='upper left', fontsize=10)

            # Add color bar
            fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, label='Objective Function')

            plt.show()
        
        elif projection == '2d':
            # Create plot
            fig, ax = plt.subplots(figsize=(6, 4))

            # Add training points
            scatter = ax.scatter(
                Xtrain_reduced[:, 0], Xtrain_reduced[:, 1],
                c=ytrain_unstandardized, cmap='viridis', label='Training Points', s=50, zorder=1
            )

            # Highlight the maximum value
            max_value_idx = np.argmax(ytrain_unstandardized)
            ax.scatter(
                Xtrain_reduced[max_value_idx, 0], Xtrain_reduced[max_value_idx, 1],
                color='blue', s=70, label='Maximum Value Found', edgecolor='black', zorder=11
            )

            if fun_max is not None:
                reduced_fun_max = dim_reducer.transform(fun_max)
                ax.scatter(
                    reduced_fun_max[:,0], reduced_fun_max[:,1], 
                    color='pink', s=70, label='Known Maximum', edgecolor='black' 
                )

            # Optionally add new points
            if new_point is not None:
                new_X, new_y = new_point
                new_X = dim_reducer.transform(new_X)
                for i in range(len(new_X)):
                    ax.scatter(
                        new_X[i, 0], new_X[i, 1],
                        color='red', label='New Points' if i == 0 else None, s=60, zorder=10
                    )

                    # Add text label with point number next to it
                    ax.text(
                        new_X[i, 0], new_X[i, 1],
                        str(i + 1),  # Ensure the text is passed as a string
                        color='black', fontsize=12, ha='left', va='bottom'
                    )

            # Labels and legend
            ax.set_title(f'Bayesian Optimization Iteration: {iteration}', fontsize=16)
            ax.set_xlabel('PC 1', fontsize=12)
            ax.set_ylabel('PC 2', fontsize=12)
            ax.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
            #ax.legend(loc='upper left', fontsize=10)

            # Add color bar
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
            cbar.set_label('Objective Function', fontsize=12)

            plt.show()

        else:
            raise ValueError("Invalid projection. Choose '2d' or '3d'.")

    else:
        # Define test grid
        n_values_pH, n_values_T = 50, 50
        pH_values = np.linspace(4, 9, n_values_pH)
        T_values = np.linspace(30, 40, n_values_T)
        T, pH = np.meshgrid(T_values, pH_values)
        test_x = np.stack((T, pH), axis=-1).reshape(-1, 2)

        # Standardize test points
        test_x_standardized = (test_x - X_mean) / X_std
        test_x_standardized = torch.tensor(test_x_standardized, dtype=torch.float32)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(test_x_standardized))

        # Unstandardize predictions
        y_mean_pred = unstandardize_y(observed_pred.mean.numpy(), y_mean, y_std)

        # Reshape predictions for plotting
        y_mean_pred_grid = y_mean_pred.reshape(n_values_pH, n_values_T)

        # Unstandardize training points
        Xtrain_unstandardized = Xtrain.numpy() * X_std + X_mean
        ytrain_unstandardized = unstandardize_y(ytrain.numpy(), y_mean, y_std)
    
        # Standard 3D GP mean surface plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the GP mean surface
        surf = ax.plot_surface(T, pH, y_mean_pred_grid, cmap='viridis', alpha=0.4, edgecolor='none')

        # Add training points
        ax.scatter(
            Xtrain_unstandardized[:, 0], Xtrain_unstandardized[:, 1],
            ytrain_unstandardized, color='black', label='Training Points'
        )

        # Optionally add new points
        if new_point is not None:
            new_X, new_y = new_point
            for i in range(len(new_X)):
                ax.scatter(
                    [new_X[i][0]], [new_X[i][1]], [new_y[i]],
                    color='red', label='New Points' if i == 0 else None, s=60
                )

                # Add text label with point number next to it
                ax.text(
                    new_X[i][0], new_X[i][1], new_y[i],  # Use x, y, and z coordinates for the label
                    str(i+1),  # Ensure the text is passed as a string
                    color='black', fontsize=12, ha='left', va='bottom'
                )

        # Labels and legend
        ax.set_title(f'Bayesian Optimization Iteration: {iteration}', fontsize=16)
        ax.set_xlabel('T (°C)', fontsize=12)
        ax.set_ylabel('pH', fontsize=12)
        ax.set_zlabel('P (g/L)', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)

        # Add color bar for surface
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='GP Posterior Mean')

        plt.show()


def plot_acquisition_function(ei_function, bounds, X_mean, X_std, iteration, resolution=40):
        """
        Plots the acquisition function surface in 3D.
        
        Parameters:
            ei_function (callable): Acquisition function (Expected Improvement).
            bounds (list of tuples): Bounds for the input space [(low1, high1), (low2, high2)].
            X_mean (ndarray): Mean of the original data for standardization.
            X_std (ndarray): Standard deviation of the original data for standardization.
            iteration (int): Current iteration number.
            resolution (int): Number of points along each dimension for the grid.
        """
        # Create a grid over the input space
        T = np.linspace(bounds[0][0], bounds[0][1], resolution)
        pH = np.linspace(bounds[1][0], bounds[1][1], resolution)
        T_grid, pH_grid = np.meshgrid(T, pH)
        grid_points = np.stack((T_grid, pH_grid), axis=-1).reshape(-1, 2)

        # Standardize the grid points
        grid_points_standardized = (grid_points - X_mean) / X_std
        grid_points_tensor = torch.tensor(grid_points_standardized, dtype=torch.float32)

        # Compute acquisition function values
        with torch.no_grad():
            ei_values = np.array([ei_function(x.unsqueeze(0)).item() for x in grid_points_tensor])

        # Reshape EI values to match the grid
        ei_values_grid = ei_values.reshape(resolution, resolution)

        # Plot the acquisition function
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Surface plot
        surf = ax.plot_surface(
            T_grid, pH_grid, ei_values_grid, cmap='viridis', alpha=0.8, edgecolor='none'
        )

        # Labels and title
        ax.set_title(f'Acquisition Function (EI) - Iteration {iteration}', fontsize=16)
        ax.set_xlabel('T (°C)', fontsize=12)
        ax.set_ylabel('pH', fontsize=12)
        ax.set_zlabel('EI Value', fontsize=12)

        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='EI Value')

        plt.show()
