import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from scipy.stats import multivariate_normal
from Func import construct_covariance_matrix, \
    high_dimensional_gmm_log_likelihood, high_dimensional_gmm_probability
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import networkx as nx
import torch.nn.utils as utils
from scipy.stats import norm
import matplotlib.font_manager as fm
from filterpy.kalman import KalmanFilter

# Set English and number font to Times New Roman
fm.fontManager.addfont(fm.findfont('Times New Roman'))
plt.rcParams['font.sans-serif'] = ['Times New Roman']
# Solve the problem of negative sign display
plt.rcParams['axes.unicode_minus'] = False
# Set global font size to 12
plt.rcParams.update({'font.size': 12})


# Define attention mechanism module
class Attention(nn.Module):
    def __init__(self, num_components):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(num_components, 1))

    def forward(self, weights):
        # Calculate attention scores
        attention_scores = F.softmax(torch.matmul(weights.t(), self.attention_weights), dim=0)
        # Adjust weights
        adjusted_weights = weights * attention_scores.t()
        return adjusted_weights


# Calculate Gaussian Gaussian Gaussian
def compute_p_x_gmm_parameters(mu, sigma, weights):
    """
    Calculate Gaussian mixture parameters for P(X)

    Parameters:
    - mu: Means of GMM, shape [num_components, num_features]
    - sigma: Covariance matrices of GMM, shape [num_components, num_features, num_features]
    - weights: Weights of GMM, shape [num_components]

    Returns:
    - x_mu: Means of the Gaussian mixture model for P(X), shape [num_components]
    - x_sigma: Standard deviations of the Gaussian mixture model for P(X), shape [num_components]
    - x_weights: Weights of the Gaussian mixture model for P(X), shape [num_components]
    """
    # Extract means of X (mean of X in each component)
    x_mu = mu[:, 0]  # Shape: [num_components]

    # Extract variance of X (first element on the diagonal of the covariance matrix)
    x_var = sigma[:, 0, 0]  # Shape: [num_components]

    # Calculate standard deviation of X
    x_sigma = torch.sqrt(x_var)  # Shape: [num_components]

    # Weights of X are the same as the original GMM
    x_weights = weights  # Shape: [num_components]

    return x_mu, x_sigma, x_weights


# Calculate probability density of P(X) at given x values
def compute_p_x_density(x_values, x_mu, x_sigma, x_weights):
    """
    Calculate probability density of P(X) at given x values

    Parameters:
    - x_values: x values at which to calculate probability density, shape [n_points]
    - x_mu: Means of the Gaussian mixture model for P(X), shape [num_components]
    - x_sigma: Standard deviations of the Gaussian mixture model for P(X), shape [num_components]
    - x_weights: Weights of the Gaussian mixture model for P(X), shape [num_components]

    Returns:
    - density: Probability density of P(X) at x_values, shape [n_points]
    """
    n_points = len(x_values)
    density = np.zeros(n_points)

    # Weighted sum over each component
    for k in range(len(x_weights)):
        # Calculate probability density of the k-th Gaussian component at x_values
        component_density = x_weights[k].item() * norm.pdf(
            x_values,
            loc=x_mu[k].item(),
            scale=x_sigma[k].item()
        )
        density += component_density

    return density


# Visualize the distribution of P(X)
def visualize_p_x_distribution(x_values, density, x_mu, x_sigma, x_weights):
    """Visualize the distribution of P(X)"""
    # Visualization settings
    width_cm = 16
    height_cm = width_cm * 0.618  # Golden ratio height, adjustable as needed
    width_inch = width_cm / 2.54
    height_inch = height_cm / 2.54
    # print(width_inch, height_inch)
    # Plot the figure
    fig, ax = plt.subplots(figsize=(width_inch, height_inch))

    # Plot the distribution curve of P(X)
    ax.plot(x_values, density, '#E84445', linestyle='-', linewidth=2, label='P(X)')

    # Plot each Gaussian component
    for k in range(len(x_weights)):
        component_density = x_weights[k].item() * norm.pdf(
            x_values,
            loc=x_mu[k].item(),
            scale=x_sigma[k].item()
        )
        ax.plot(x_values, component_density, '#23BAC5', linestyle='--', linewidth=1,
                label='Component' if k == 0 else None)
    """
    # Mark the mean position of each component
    for k in range(len(x_weights)):
        ax.axvline(x=x_mu[k].item(), color='r', linestyle=':', alpha=0.5,
                    label=f'Component mean' if k == 0 else None)
    """

    # Set x-axis tick positions and labels
    y_ticks = np.arange(0, 0.07, 0.01)  # From 0 to 10, interval 2
    ax.set_yticks(y_ticks)

    legend = ax.legend(loc='upper right', frameon=True, handletextpad=0.1, fontsize=10)
    legend.get_frame().set_facecolor('white')
    # Set legend border color, here 'red' as an example, can be modified as needed
    legend.get_frame().set_edgecolor('white')
    # plt.title('Marginal Distribution P(X) and Its Components', fontsize=16)
    ax.set_xlabel('Flow')
    ax.set_ylabel('Probability Density')

    # Set axis line widths
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)

    ax.tick_params(width=1, length=2)

    plt.show()


def sliding_window(X, window_size):
    """
    This function is used to create sliding windows
    :param X: Input data
    :param window_size: Window size
    :return: Data within the window and the first data after the window
    """
    X_windows = []
    y_windows = []
    for i in range(len(X) - window_size):
        X_windows.append(X[i:i + window_size])
        y_windows.append(X[i + window_size])
    return np.array(X_windows), np.array(y_windows)


def check_intersections(row):
    for value in row:
        if pd.notna(value) and value not in allowed_intersections:
            return False
    return True


# Define multi-head GAT layer
class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, alpha=0.2):
        super(MultiHeadGATLayer, self).__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([GATLayer(in_features, out_features, alpha) for _ in range(num_heads)])

    def forward(self, input, adj):
        head_outs = [head(input, adj) for head in self.heads]
        return torch.cat(head_outs, dim=1)


# Define GAT layer
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, h)

        return h_prime


# Define GAT model
class GAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads, num_layers=2):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(MultiHeadGATLayer(in_features, hidden_features, num_heads))
        for _ in range(num_layers - 2):
            self.layers.append(MultiHeadGATLayer(hidden_features * num_heads, hidden_features, num_heads))
        self.layers.append(MultiHeadGATLayer(hidden_features * num_heads, out_features, num_heads))

    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = F.elu(layer(x, adj))
        x = self.layers[-1](x, adj)
        return x


class Gaussian(nn.Module):

    def __init__(self, hidden_size, mu_size, d):
        '''
        Gaussian Likelihood Supports Continuous Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(Gaussian, self).__init__()
        sigma_size = d * (d + 1) // 2
        self.d = d
        # self.mu_layer = nn.Linear(hidden_size, mu_size)

        self.mu_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, mu_size)
        )

        # Learnable scaling factor and bias
        self.scale = nn.Parameter(torch.ones(mu_size))
        self.bias = nn.Parameter(torch.zeros(mu_size))

        self.sigma_layer = nn.Linear(hidden_size, sigma_size)  # d*(d+1)/2

        """
        # Add layers to mu_layer
        self.mu_layer = nn.Sequential(
            nn.Linear(hidden_size, 128),  # First linear layer, output dimension adjustable as needed
            nn.ReLU(),  # Activation function
            nn.Linear(128, 64),  # Second linear layer, output dimension adjustable as needed
            nn.ReLU(),  # Activation function
            nn.Linear(64, mu_size)  # Final output layer
        )

        # Add layers to sigma_layer
        self.sigma_layer = nn.Sequential(
            nn.Linear(hidden_size, 128),  # First linear layer, output dimension adjustable as needed
            nn.ReLU(),  # Activation function
            nn.Linear(128, 64),  # Second linear layer, output dimension adjustable as needed
            nn.ReLU(),  # Activation function
            nn.Linear(64, sigma_size)  # Final output layer
        )
        """
        # Modify initialization range of mu_layer
        # nn.init.uniform_(self.mu_layer.weight, -1, 1)  # Adjustable range as needed
        # nn.init.uniform_(self.mu_layer.bias, -1, 1)  # Adjustable range as needed

    def forward(self, h):
        _, hidden_size = h.size()

        Mu_K = self.mu_layer(h)
        Mu_K = Mu_K * self.scale + self.bias

        theta_L_k = self.sigma_layer(h)
        Sigma_k = construct_covariance_matrix(theta_L_k, self.d)

        return Mu_K, Sigma_k


class DeepAR(nn.Module):
    def __init__(self, input_size, d_model, seq_length, nhead, output_dim, embedding_size, hidden_size, num_layers, num_layers_LSTM,
                 num_components, lr=1e-3):
        """
        Initialize the DeepAR model.

        Parameters:
        input_size (int): Dimension of input features.
        embedding_size (int): Dimension for embedding target values.
        hidden_size (int): Dimension of LSTM hidden layer.
        num_layers (int): Number of LSTM layers.
        lr (float): Learning rate, default 1e-3.
        """
        super(DeepAR, self).__init__()

        # Number of Gaussian components
        self.num_components = num_components

        # Embedding layer to convert input feature dimension to d_model
        self.embedding = nn.Linear(input_size, d_model)

        # Positional encoding layer, using learnable positional encoding here to add positional information to input sequence
        # Shape of positional encoding is (seq_length, d_model), each time step corresponds to a d_model dimension vector
        self.positional_encoding = nn.Parameter(torch.randn(seq_length, d_model))

        # Define Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        # Define Transformer encoder, stacked with multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Multiple fully connected layers to generate multiple outputs
        # self.output_layers = nn.ModuleList([nn.Linear(d_model, output_dim) for _ in range(num_components)])
        self.output_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, 256),  # Increase number of neurons
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU()
        ) for _ in range(num_components)])

        # Create independent LSTM encoders for each parallel data
        self.encoders = nn.ModuleList([
            nn.LSTM(output_dim + m, hidden_size, num_layers_LSTM, bias=True, batch_first=True)
            for _ in range(num_components)
        ])

        # Process weights
        self.weight_linear = nn.Sequential(
            nn.Linear(num_components, 128),  # Increase number of neurons
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_components))

        self.weight_linear_mu = nn.Sequential(
            nn.Linear(num_components, 128),  # Increase number of neurons
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1))

        # Define attention mechanism
        self.attention = Attention(num_components)

        # Define softmax layer
        self.softmax = nn.Softmax(dim=0)

        # Add Gaussian likelihood layers
        self.likelihood_layer = nn.ModuleList([Gaussian(hidden_size, input_size, input_size)
                                               for _ in range(num_components)])

        self.pred = nn.Sequential(
            nn.Linear(4, 128),  # Increase number of neurons
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1))

    def forward_part1(self, X):
        # Get shape information of historical input features
        num_ts, seq_len, input_dim = X.size()  # (batch_size, seq_length, input_size)

        # Convert input dimension to d_model via embedding layer
        X = self.embedding(X)  # (batch_size, seq_length, d_model)

        X = X + self.positional_encoding

        # Input dimension-adjusted data to Transformer encoder for processing
        output = self.transformer_encoder(X)  # (batch_size, seq_length, d_model)
        return output

    def forward_part2(self, output):
        # Get multiple outputs via multiple fully connected layers
        outputs = [layer(output) for layer in
                   self.output_layers]  # outputs[0].size() = (batch_size, seq_length, output_dim)
        outputs = torch.stack(outputs, dim=0)  # (num_components, batch_size, seq_length, output_dim)

        # All Gaussian components
        all_mus = []
        all_sigmas = []

        for i in range(self.num_components):
            input = outputs[i, :, :, :]  # (batch_size, seq_length, output_dim)

            # Means, variances (single component)
            mus = []
            sigmas = []

            # For each component, initialize LSTM hidden state and cell state
            h, c = None, None

            # Iterate over historical and future time steps
            for s in range(5):
                inp = input[:, s, :]  # s-th time step  (batch_size, 1, output_dim)

                # Integrate position information
                inp = torch.cat((inp, one_hot_loca_batch), dim=1)
                inp = inp.unsqueeze(1)

                if h is None and c is None:
                    # First call to LSTM, no initial hidden state and cell state provided
                    out, (h, c) = self.encoders[i](inp)  # h shape is (num_layers, num_ts, hidden_size)
                else:
                    # Subsequent calls to LSTM, provide hidden state and cell state from previous time step
                    out, (h, c) = self.encoders[i](inp, (h, c))

                # Get hidden state of the last LSTM layer
                hs = h[-1, :, :]
                """
                # Apply attention mechanism
                attn_weights = self.attention(hs.unsqueeze(0), output)
                context = torch.bmm(attn_weights, output)
                hs = torch.cat([hs, context.squeeze(1)], dim=1)
                """
                # Apply ReLU activation function
                hs = F.leaky_relu(hs)

                # Calculate mean vector and covariance matrix of probability distribution via likelihood layer
                mu, sigma = self.likelihood_layer[i](hs)

                mus = mu
                sigmas = sigma

            # All Gaussian components
            all_mus.append(mus.unsqueeze(0))
            all_sigmas.append(sigmas.unsqueeze(0))

        all_mus = torch.cat(all_mus, dim=0)  # torch.Size([10, 398, 6])
        print("all_mus: ", all_mus.size())
        all_sigmas = torch.cat(all_sigmas, dim=0)

        # Mus_w = all_mus[:, :, 0].t()
        # print("Mus_w: ", Mus_w.size())   # torch.Size([398, 10])
        # pred = self.weight_linear_mu(Mus_w)

        # Average over seq_length and output_dim dimensions
        # outputs  num_components, batch_size, seq_length, output_dim
        aggregated = self.weight_linear(outputs.sum(dim=(2, 3)).t())  # Shape: (num_components, batch_size)

        # Transpose to match dimension order expected by softmax
        temperature = 40
        transposed = aggregated / temperature  # Shape: (batch_size, num_components)
        print("transposed.size(): ", transposed.size())  # torch.Size([398, 10])

        # Apply softmax activation function (along dim=1, i.e., num_components dimension)
        weights = F.softmax(transposed, dim=1)
        # print(weights, weights.size())
        weights = weights.t()
        print("weights: ", weights.size())  # torch.Size([10, 398])

        # Adjust weights using attention mechanism
        # attention = Attention(self.num_components)
        # weights = attention(weights)
        # Adjust weights using attention mechanism
        # adjusted_weights = self.attention(weights)
        # print("weights: ", weights.size())  # torch.Size([10, 398])
        return all_mus, all_sigmas, weights, outputs

    def forward(self, X):
        output = self.forward_part1(X)
        return self.forward_part2(output)

    def forward_part3(self, X):
        Predict = self.pred(X)
        return Predict


# Road network data
gdf = pd.read_csv(r"./data/gdf.csv")
allowed_intersections = list(gdf["Down_Node"])

# Intersection information
Nodes = pd.read_csv(r"./data/gdf_processed.csv", encoding="gbk")

# Traffic flow data
Flow = pd.read_csv(r"./data/flow_interpolated.csv", encoding="gbk")

# Select column to normalize
column_to_normalize = 'Volume'

# Create MinMaxScaler object
# scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler = MinMaxScaler(feature_range=(0, 2))

# Normalize specified column
Flow[column_to_normalize] = scaler.fit_transform(Flow[[column_to_normalize]])

# Get scaling factor and offset for normalization
# min_val = scaler.data_min_[0]
# max_val = scaler.data_max_[0]
# print(min_val, max_val)

# Get scaling factor and offset from normalization
scale = scaler.scale_[0]  # Scaling factor
min_val = scaler.data_min_[0]  # Original data minimum value

# Store all sliding window results
all_X_windows = []
all_y_windows = []
for index, unit in Nodes.iterrows():
    # Extract traffic flow data  Node flow sequence shifted forward + Node flow sequence
    print(unit["target"], unit["source"], unit["neighbor1"], unit["neighbor2"], unit["neighbor3"])

    # Length of time series
    Num = len(Flow[Flow["Down_Node"] == unit["target"]]["Volume"][1:])

    # Initialize f list, first add traffic flow data of target intersection
    if Flow[Flow["Down_Node"] == unit["target"]]["Volume"][1:].shape[0] == 0:
        f = [np.array([0] * Num)]
    else:
        f = [Flow[Flow["Down_Node"] == unit["target"]]["Volume"][1:]]

    # Iterate over each node in unit
    for node in [unit["target"], unit["source"], unit["neighbor1"], unit["neighbor2"], unit["neighbor3"]]:
        if pd.notna(node):  # If node value is not empty, add traffic flow data of this node
            if Flow[Flow["Down_Node"] == node]["Volume"][:-1].shape[0] == 0:  # If node traffic flow data is empty
                f.append(np.array([0] * Num))
            else:  # If node traffic flow data is not empty
                f.append(Flow[Flow["Down_Node"] == node]["Volume"][:-1])
        else:
            # If node value is empty, add an array of all zeros
            f.append(np.array([0] * Num))
    F1 = np.c_[f].T
    F1 = F1[:143]

    """
    # Apply Kalman filter to each column of F1
    F1_denoised = np.zeros_like(F1)
    for i in range(F1.shape[1]):
        # Initialize Kalman filter
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.x = np.array([F1[0, i]])  # Initial state
        kf.F = np.array([[1.]])  # State transition matrix
        kf.H = np.array([[1.]])  # Observation matrix
        kf.P = np.array([[1.]])  # Initial covariance matrix
        kf.R = np.array([[0.1]])  # Observation noise covariance
        kf.Q = np.array([[0.01]])  # Process noise covariance

        for j in range(F1.shape[0]):
            kf.predict()
            kf.update(np.array([F1[j, i]]))
            F1_denoised[j, i] = kf.x[0]
    """
    # Set window size
    window_size = 12
    X_windows, y_windows = sliding_window(F1, window_size)

    # Add current sliding window results to lists
    all_X_windows.append(X_windows)
    all_y_windows.append(y_windows)
    print("Shape of data within window:", X_windows.shape)
    print("Shape of first data after window:", y_windows.shape)
    print("---------------------------------------------")


# Combine all sliding window results into arrays
all_X_windows = torch.tensor(np.array(all_X_windows), dtype=torch.float32)  # torch.Size([398, 67, 5, 6])
all_y_windows = torch.tensor(np.array(all_y_windows), dtype=torch.float32)  # torch.Size([398, 67, 6])
print("Shape of all data within windows:", all_X_windows.size())
print("Shape of all first data after windows:", all_y_windows.size())

XX = all_X_windows.permute(1, 0, 2, 3)  # Shape of all data within windows: torch.Size([67, 398, 5, 6])  batch_size, num_unit, seq_len, num_fea
y = all_y_windows.permute(1, 0, 2)  # Shape of all first data after windows: torch.Size([67, 398, 6])  batch_size, num_unit, num_fea
print("Shape of all data within windows:", XX.size())
print("Shape of all first data after windows:", y.size())

X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size=0.5, random_state=42)
# X_train = XX[:65, :, :, :]
# X_test = XX[65:, :, :, :]

# y_train = y[:65, :, :]
# y_test = y[65:, :, :]
print("X_train:", X_train.size())  # torch.Size([398, 137, 12, 6])   num_unit, batch_size, num_data, feature_num
print("y_train:", y_train.size())  # torch.Size([398, 137, 6])  num_unit, batch_size, feature_num

print("X_test:", X_test.size())  # torch.Size([398, 137, 12, 6])   num_unit, batch_size, num_data, feature_num
print("y_test:", y_test.size())  # torch.Size([398, 137, 6])  num_unit, batch_size, feature_num

# Position indices
# Define values of n and m
n = 1  # Amount of data per intersection
m = len(Nodes)  # Number of intersections
# print(n, m)

# Create a tensor containing n zeros, n ones, ..., n m's
loca = torch.cat([torch.full((n,), i) for i in range(m)])
one_hot_loca_batch = torch.nn.functional.one_hot(loca.long(), num_classes=m)
# print(one_hot_loca_batch.size())

# Parameters
input_size = 6  # Input dimension
num_epoches = 40  # Training epochs
num_components = 10

# Result storage
TrLoss = []
TeLoss = []
P = []
T = []

res1 = []
# Instantiate
model = DeepAR(input_size=input_size, seq_length=12, d_model=128, nhead=4, output_dim=32, embedding_size=64,
               hidden_size=64, num_layers=2, num_layers_LSTM=2, num_components=num_components, lr=1e-3)
optimizer = Adam(model.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()  # Or nn.MSELoss(reduction='sum'), etc.
mae_loss = nn.L1Loss()


def custom_matrix_power1(matrices):
    # Check if input matrices are square and have the same shape
    rows, cols = matrices[0].shape
    if rows != cols:
        raise ValueError("Input matrices must be square for computation.")
    for matrix in matrices[1:]:
        m_rows, m_cols = matrix.shape
        if m_rows != m_cols or m_rows != rows:
            raise ValueError("All input matrices must be square and of the same shape.")

    # Initialize result matrix as the first matrix
    result = matrices[0]
    # Iterate over remaining matrices for custom multiplication
    for matrix in matrices[1:]:
        new_result = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                # Custom multiplication rule: take minimum of corresponding elements and sum
                new_result[i, j] = np.sum(np.minimum(matrix[i, :], result[:, j]))
        result = new_result
    return result


def custom_matrix_power(matrices):
    # Check if input matrices are square and have the same shape
    rows, cols = matrices[0].shape
    if rows != cols:
        raise ValueError("Input matrices must be square for computation.")
    for matrix in matrices[1:]:
        m_rows, m_cols = matrix.shape
        if m_rows != m_cols or m_rows != rows:
            raise ValueError("All input matrices must be square and of the same shape.")

    # Initialize result as zero matrix
    result = np.zeros((rows, cols))

    # Initialize current product as identity matrix (for "product" of the first matrix)
    current_product = np.eye(rows)

    # Iterate over each matrix, compute cumulative custom product and add to result
    for matrix in matrices:
        # Update current product: custom multiplication of current product and current matrix
        new_product = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                # Custom multiplication rule: take minimum of corresponding elements and sum
                new_product[i, j] = np.sum(np.minimum(matrix[i, :], current_product[:, j]))
        current_product = new_product

        # Add current product to result
        result += current_product

    return result


def create_graph(TP):
    # Create directed graph
    G = nx.DiGraph()
    # Add edges and edge weights
    for index, row in TP.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['PTE'])
    # Compute adjacency matrix using NetworkX's adjacency_matrix function and convert to dense matrix
    adj_matrix = nx.adjacency_matrix(G, weight='weight').todense()
    # Set diagonal elements of adjacency matrix to 1
    for i in range(adj_matrix.shape[0]):
        adj_matrix[i, i] = 1
    return adj_matrix, G


# Read CSV file containing latitude and longitude
TP = pd.read_csv(r"./data/gdf_processed_with_location.csv")

# Initialize GAT model
in_features = 1
hidden_features = 8
out_features = 64
num_heads = 12  # Number of attention heads
model_gat = GAT(in_features, hidden_features, out_features, num_heads)

Res_pred = []
Res_true = []
KK = 0
for epoch in range(num_epoches):
    true_value = []
    pred_value = []
    for i in range(X_train.size()[0]):
        print("t =", i)

        # Iterate over dimension 0 (batch_size dimension), extract data for each batch
        X_train_batch = X_train[i, :, :, :]
        y_train_batch = y_train[i, :, :]

        # Part 1 inference
        output_part1 = model.forward_part1(X_train_batch)

        if len(res1) >= 5:
            mats11 = res1[-5:]

            PTE = pd.DataFrame(mats11)
            PTE = PTE.clip(lower=0)  # Set all negative numbers in all columns to 0
            print(PTE)

            power = 4
            mats = []
            for i in range(power):
                row_list = PTE.iloc[-5 + i].tolist()
                TP["PTE"] = row_list
                mat, G = create_graph(TP)
                mats.append(mat)

            matrix_power_result = custom_matrix_power(mats)
            # print(matrix_power_result)
            # print(matrix_power_result.shape)  # (124, 124)

            emb_result = []
            for target_node in list(TP['target']):
                random_node = target_node  # 5021

                # Create a new directed graph
                new_G = nx.DiGraph()
                # Add target node
                new_G.add_node(random_node)

                # Find all nodes with one-way edge weight > 0 to random_node
                nodes = list(G.nodes())
                target_index = nodes.index(random_node)
                valid_nodes = []

                for i in range(len(nodes)):
                    if matrix_power_result[i, target_index] > 0:
                        valid_nodes.append(nodes[i])

                if len(valid_nodes) > 1:
                    # Add nodes with one-way edge weight > 0 to random_node and the edges
                    for node in valid_nodes:
                        new_G.add_node(node)

                    # Add edges from other nodes to random_node
                    for other_node in valid_nodes:
                        if other_node != random_node:
                            source_index = nodes.index(other_node)
                            target_index = nodes.index(random_node)
                            new_G.add_edge(other_node, random_node,
                                           weight=matrix_power_result[source_index, target_index])

                    # Add directed edges between other nodes
                    # Add original edges and their weights between these nodes, excluding self-loops
                    for u, v, data in G.edges(data=True):
                        if u in valid_nodes and v in valid_nodes and u != v and u != random_node and v != random_node:
                            new_G.add_edge(u, v, weight=data['weight'])

                    # Output adjacency matrix of new_G
                    adj_matrix_new_G = nx.adjacency_matrix(new_G, weight='weight').todense()
                    # ("Adjacency matrix of new_G:")
                    # print(adj_matrix_new_G)

                    # Assume node feature dimension is 1, here simply use all-ones vector as node feature
                    num_nodes = new_G.number_of_nodes()
                    x = torch.ones((num_nodes, 1), dtype=torch.float)

                    # Convert adjacency matrix to torch tensor
                    adj = torch.tensor(adj_matrix_new_G, dtype=torch.float)

                    # Forward pass to get node embeddings
                    node_embeddings = model_gat(x, adj)

                    # Calculate edge embeddings, here simply use sum of embeddings of two ends of the edge as edge embedding
                    edge_embeddings = []
                    for u, v in new_G.edges():
                        u_index = list(new_G.nodes()).index(u)
                        v_index = list(new_G.nodes()).index(v)
                        edge_embedding = node_embeddings[u_index] + node_embeddings[v_index]
                        edge_embeddings.append(edge_embedding)

                    edge_embeddings = torch.stack(edge_embeddings)

                else:
                    edge_embeddings = torch.zeros((num_heads, out_features))

                # Adjust shape of edge embeddings to get [K, out_dim] output
                final_edge_embeddings = edge_embeddings.view(num_heads, -1, out_features).mean(dim=1)

                # print("Shape of edge embeddings:", final_edge_embeddings.shape)  # Shape of edge embeddings: torch.Size([5, 64])
                emb_result.append(final_edge_embeddings)
                # print("Edge embeddings:")
                # print(final_edge_embeddings)
                # print("----------------------------------")

            # Stack all tensors
            final_edge_embeddings = torch.stack(emb_result, dim=0)  # Shape: [398, 5, 64]
            print("final_edge_embeddings.size: ", final_edge_embeddings.size())

            # Two dimensions???
            # final_edge_embeddings = final_edge_embeddings.unsqueeze(0).expand(398, -1, -1)  # torch.Size([111, 5, 16])
            # final_edge_embeddings = final_edge_embeddings.unsqueeze(0).expand(10, -1, -1, -1)  # torch.Size([10, 111, 5, 16])
            # print("Shape of edge embeddings:", final_edge_embeddings.shape)

            # Concatenate  # outputs = torch.stack(outputs, dim=0)  # (num_components, batch_size, seq_length, output_dim)  torch.Size([10, 398, 5, 32])
            print("output_part1.size: ", output_part1.size())
            Merge = torch.cat([output_part1, final_edge_embeddings], dim=2)  # Merge along dimension 2
            # print("Shape after concatenation:", Merge.shape)

            # Define linear projection layer: compress last dimension from 192 to 128
            projection = nn.Linear(192, 128)

            # Adjust dimensions to fit linear layer input: linear layer expects input [N, 192] (N is any batch dimension)
            # Here merge first two dimensions into one batch dimension, shape becomes [398*5, 192]
            batch_size, seq_len, feat_dim = Merge.shape  # 398, 5, 192
            Merge_reshaped = Merge.view(-1, feat_dim)  # Equivalent to Merge.view(398*5, 192)

            # Apply projection: compress 192 dimensions to 128, output shape [398*5, 128]
            Merge_projected = projection(Merge_reshaped)

            # Restore original dimension structure: split merged batch dimension back to [398, 5, 128]
            Merge_output = Merge_projected.view(batch_size, seq_len, 128)
            # print("Shape of tensor after projection:", Merge_output.shape)  # Output: torch.Size([398, 5, 128])

            # Multi-task DeepAR
            mu, sigma, weights, temp_outputs = model.forward_part2(Merge_output)
            # print("mu: ", mu.size())
            # print("sigma: ", sigma.size())
            # print("weights: ", weights.size())

        else:
            pass

        # Part 2 inference
        mu, sigma, weights, temp_outputs = model.forward_part2(output_part1)
        prob = high_dimensional_gmm_probability(y_train_batch, mu, sigma, weights)

        res = []
        for j in range(y_train_batch.size()[0]):
            # y_t+1, x_t, y_t, z_t
            # Calculate P(x, y, z, t)
            joint_prob = 0
            for k in range(num_components):
                joint_prob += weights[k, j] * multivariate_normal.pdf(y_train_batch[j, :].tolist(),
                                                                      mean=mu[k, j, :].detach().numpy(),
                                                                      cov=sigma[k, j, :, :].detach().numpy())

            # Calculate P(y, z, t)
            marginal_prob_yzt = 0
            for k in range(num_components):
                mean_yzt = mu[k, j, 1:].detach().numpy()
                cov_yzt = sigma[k, j, 1:, 1:].detach().numpy()
                marginal_prob_yzt += weights[k, j] * multivariate_normal.pdf(y_train_batch[j, 1:].squeeze(0).tolist(),
                                                                             mean=mean_yzt, cov=cov_yzt)

            # Calculate P(x|y, z, t)
            cond_prob_x_given_yzt = joint_prob / marginal_prob_yzt

            # Calculate P(x, y, z)
            joint_prob_x_yz = 0
            for k in range(num_components):
                mean_x_yz = mu[k, j, [0, 1, 3, 4, 5]].detach().numpy()
                cov_x_yz = sigma[k, j, [0, 1, 3, 4, 5], [0, 1, 3, 4, 5]].detach().numpy()
                joint_prob_x_yz += weights[k, j] * multivariate_normal.pdf(
                    y_train_batch[j, [0, 1, 3, 4, 5]].squeeze(0).tolist(), mean=mean_x_yz, cov=cov_x_yz)

            # Calculate P(y, z)
            marginal_prob_yz = 0
            for k in range(num_components):
                mean_yz = mu[k, j, [1, 3, 4, 5]].detach().numpy()
                cov_yz = sigma[k, j, [1, 3, 4, 5], [1, 3, 4, 5]].detach().numpy()
                marginal_prob_yz += weights[k, j] * multivariate_normal.pdf(
                    y_train_batch[j, [1, 3, 4, 5]].squeeze(0).tolist(), mean=mean_yz, cov=cov_yz)

            # Calculate P(x|y, z)
            cond_prob_x_given_yz = joint_prob_x_yz / marginal_prob_yz

            try:
                res.append(prob[j] * np.log(cond_prob_x_given_yzt.item() / cond_prob_x_given_yz.item()))
            except:
                res.append(0)
        res1.append(res)
        # print(len(res))

        # Calculate training loss
        Trloss, TrSta = high_dimensional_gmm_log_likelihood(y_train_batch, mu, sigma, weights)  # Time

        print("y_train_batch: ", y_train_batch.size())  # torch.Size([398, 6])
        print("mu: ", mu.size())  # torch.Size([10, 398, 6])
        print("sigma: ", sigma.size())  # torch.Size([10, 398, 6, 6])