import numpy as np
import torch
from torch import nn
import torch.distributions as dist


# Define a function to replace semicolons with commas
def correct_linestring_format(linestring_str):
    start_index = linestring_str.find('(') + 1
    end_index = linestring_str.rfind(')')
    coordinates = linestring_str[start_index:end_index]
    corrected_coordinates = coordinates.replace(';', ', ')
    return f'LINESTRING({corrected_coordinates})'


# Min-Max normalization
def min_max_normalization(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


# Z-score normalization
def z_score_normalization(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    normalized_data = (data - mean_val) / std_val
    return normalized_data


def is_positive_definite_cholesky_pytorch(cov_tensor):
    """
    Check if a covariance matrix (tensor) is positive definite using Cholesky decomposition
    :param cov_tensor: Covariance matrix tensor
    :return: True if positive definite, False otherwise
    """
    try:
        torch.linalg.cholesky(cov_tensor)
        return True
    except RuntimeError:
        return False


def high_dimensional_gmm_probability(data, means, covs, weights):
    """
    Calculate probabilities for high-dimensional Gaussian Mixture Model
    :param data: Input data, shape (batch_size, dim)
    :param means: Means of each Gaussian component, shape (num_components, batch_size, dim)
    :param covs: Covariance matrices of each Gaussian component, shape (num_components, batch_size, dim, dim)
    :param weights: Weights of each Gaussian component, shape (num_components, batch_size), with sum of weights equal to 1
    :return: Probabilities of each data point under the high-dimensional Gaussian mixture distribution, shape (batch_size,)
    """
    batch_size, dim = data.size()
    num_components = means.size(0)

    # Initialize probability list
    probabilities = []

    # Iterate over each data point
    for i in range(batch_size):
        # Initialize list for probabilities of current data point under each Gaussian component
        component_probs = []
        # Iterate over each Gaussian component
        for k in range(num_components):
            mvn = dist.MultivariateNormal(means[:, i, :][k], covs[:, i, :, :][k])

            # Calculate probability of current data point under this Gaussian component
            prob = torch.exp(mvn.log_prob(data[i]))
            component_probs.append(prob)

        # Convert probability list to tensor
        component_probs = torch.stack(component_probs)
        # Weighted sum of probabilities from each Gaussian component based on weights

        weighted_probs = weights[:, i].unsqueeze(1) * component_probs
        # Calculate final probability
        final_prob = torch.sum(weighted_probs)

        # Set decimal places
        formatted_prob = round(final_prob.item(), 5)
        probabilities.append(formatted_prob)

    return probabilities


def high_dimensional_gmm_log_likelihood(data, means, covs, weights):
    """
    Calculate log likelihood for high-dimensional Gaussian Mixture Model
    :param data: Input data, shape (batch_size, dim)
    :param means: Means of each Gaussian component, shape (num_components, batch_size, dim)
    :param covs: Covariance matrices of each Gaussian component, shape (num_components, batch_size, dim, dim)
    :param weights: Weights of each Gaussian component, shape (num_components, batch_size), with sum of weights equal to 1
    :return: Log likelihood value, a scalar
    """
    batch_size, dim = data.size()
    num_components = means.size(0)

    # Initialize log likelihood list
    log_likelihoods = []

    status = "true"
    # Iterate over each data point
    for i in range(batch_size):
        # Initialize list for log probabilities of current data point under each Gaussian component
        component_log_probs = []
        # Iterate over each Gaussian component
        for k in range(num_components):
            # Create multivariate Gaussian distribution object

            if not is_positive_definite_cholesky_pytorch(covs[k, i, :, :]):
                status = "false"
                print("false")
                break
            else:
                pass

            epsilon = 1e-6
            mvn = dist.MultivariateNormal(means[k, i, :], covs[k, i, :, :])
            # Calculate log probability of current data point under this Gaussian component
            log_prob = mvn.log_prob(data[i])
            component_log_probs.append(log_prob)

        # Convert log probability list to tensor
        component_log_probs = torch.stack(component_log_probs)
        # Weighted sum of log probabilities from each Gaussian component based on weights
        weighted_log_probs = torch.log(weights[:, i].unsqueeze(1)) + component_log_probs

        # Use logsumexp to avoid numerical underflow and calculate final log likelihood
        log_likelihood = torch.logsumexp(weighted_log_probs, dim=0)
        log_likelihoods.append(log_likelihood)

    # Stack log likelihoods of all data points into a tensor
    log_likelihoods = torch.stack(log_likelihoods)
    # Sum log likelihoods of all data points
    total_log_likelihood = log_likelihoods.sum()
    # Take the negative to get negative log likelihood loss
    negative_log_likelihood = -total_log_likelihood

    return negative_log_likelihood, status


def high_dimensional_gmm_log_likelihood_new(data, means, covs, weights):
    batch_size, dim = data.size()
    num_components = means.size(0)
    log_likelihoods = []
    status = "true"  # Default to normal

    # Add numerical stability保障 for covariance matrices (critical!)
    epsilon = 1e-4
    eye = torch.eye(dim, device=data.device)  # Identity matrix

    for i in range(batch_size):
        # Check if all component covariances for current data point are positive definite
        component_log_probs = []
        valid = True  # Mark if all components of current data point are valid
        for k in range(num_components):
            # Force covariance matrix to be positive definite (add small perturbation)
            cov = covs[k, i, :, :] + eye * epsilon
            # Re-check positive definiteness (avoid non-positive definite after perturbation)
            if not is_positive_definite_cholesky_pytorch(cov):
                status = "false"
                valid = False
                break  # Terminate component loop for current data point
            # Calculate log probability for current component
            mvn = dist.MultivariateNormal(means[k, i, :], cov)
            log_prob = mvn.log_prob(data[i])
            component_log_probs.append(log_prob)

        print("component_log_probs: ", component_log_probs)
        # If current data point has invalid components, skip it (or handle exception)
        if not valid:
            continue  # Or record error to avoid using invalid results

        # Stack component log probabilities and check dimension validity
        component_log_probs = torch.stack(component_log_probs)  # Shape (num_components,)

        print("weights[:, i]: ", weights[:, i])
        # Check weight validity (avoid log(weights) exceptions)
        if (weights[:, i] < 0).any() or (weights[:, i].sum() < 1 - 1e-3):
            status = "false"
            continue

        # Calculate weighted log probabilities
        weighted_log_probs = torch.log(weights[:, i]) + component_log_probs  # Shape (num_components,)
        # Calculate log likelihood for current data point
        log_likelihood = torch.logsumexp(weighted_log_probs, dim=0)
        log_likelihoods.append(log_likelihood)
        print("log_likelihood: ", log_likelihood)
    # Handle extreme case where all data points are invalid
    if not log_likelihoods:
        return torch.tensor(float('inf'),
                            device=data.device), status  # Return infinite loss to trigger parameter correction

    # Calculate total negative log likelihood
    total_log_likelihood = torch.stack(log_likelihoods).sum()
    negative_log_likelihood = -total_log_likelihood
    return negative_log_likelihood, status


def construct_covariance_matrix(theta_L_k, d):
    """
    Construct lower triangular matrix L_k from fully connected layer output and calculate covariance matrix Sigma_k
    :param theta_L_k: Output of fully connected layer, shape (batch_size, d*(d + 1) // 2)
    :param d: Dimension of time series
    :return: Covariance matrix Sigma_k, shape (batch_size, d, d)
    """
    batch_size = theta_L_k.size(0)
    # Initialize lower triangular matrix L_k
    L_k = torch.zeros((batch_size, d, d), dtype=theta_L_k.dtype, device=theta_L_k.device)

    # Define softplus activation function
    softplus = nn.Softplus()

    # Define small offset to prevent diagonal elements from being close to 0
    epsilon = 1e-4  # Can be adjusted according to actual scenarios (e.g., 1e-5 or 1e-4)

    # Fill lower triangular matrix L_k
    index = 0
    for i in range(d):
        # Apply softplus activation to main diagonal elements
        L_k[:, i, i] = softplus(theta_L_k[:, index])
        index += 1
        for j in range(i):
            # Directly assign non-main diagonal elements
            L_k[:, i, j] = theta_L_k[:, index]
            index += 1

    # Calculate covariance matrix Sigma_k
    Sigma_k = torch.bmm(L_k, L_k.transpose(1, 2))

    return Sigma_k