# MATH7013 Project
# Deep Direct RL for Financial Signal Representation and Training - paper by Deng et al.
# Program for training the FDRNN
# Implemented by Joel Thomas
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import torch
import timeit

pd.options.mode.chained_assignment = None
# Set font size when plotting
plt.rcParams.update({'font.size': 14})


# Class for fuzzy layer
class FuzzyLayer(torch.nn.Module):
    """
    Custom PyTorch neural network module that implements a single fuzzy layer used as the second layer in the FDRNN.
    """
    def __init__(self, input_size, fuzzy_deg, fuzzy_params, batch_size, seq_length):
        """
        Initialises the fuzzy layer with important parameters.
        """
        super(FuzzyLayer, self).__init__()
        self.input_size = input_size                        # Size of each input f_t
        self.fuzzy_deg = fuzzy_deg                          # Number of fuzzy degrees to assign to each dimension of f_t
        self.fuzzy_params = fuzzy_params                    # 2D Matrix of means and widths of each node in the fuzzy layer 
        self.batch_size = batch_size                        # Number of total inputs in a single batch fed into the FDRNN
        self.seq_length = seq_length                        # Number of consecutive timesteps (f_ts) considered in one input
        self.output_size = self.input_size*self.fuzzy_deg   # Size of each output after passing f_t through fuzzy layer
    
    def forward(self, x):
        """
        Perform a forward pass of input x (batch containing multiple sequences of f_ts) through the fuzzy layer and return
        the fuzzied output for each f_t in each sequence in the given batch.
        """
        # Stores fuzzied x
        fuzzied_x = []
        for seq in x:
            # Stores fuzzied (sequence of) f_ts
            fuzzied_seq = []
            for f_t in seq:
                # Stores fuzzied f_t
                fuzzied_f_t = []
                for k in range(self.fuzzy_deg):
                    curr_idx = k*self.input_size
                    next_idx = (k + 1)*self.input_size
                    # Use Gaussian membership function in generating fuzzied representation of each provided f_t
                    fuzzied_f_t.append(np.exp(-1*np.divide(np.square(f_t - self.fuzzy_params[curr_idx:next_idx, 0]),
                                                           self.fuzzy_params[curr_idx:next_idx, 1])))
                
                fuzzied_seq.append(fuzzied_f_t)
            
            fuzzied_x.append(fuzzied_seq)
        
        fuzzied_x = np.reshape(np.array(fuzzied_x, dtype=np.float32), (self.batch_size, self.seq_length, self.output_size))
        # Store on device memory to help train model on GPU
        x = torch.cuda.FloatTensor(fuzzied_x)
        return x
    
    def extra_repr(self):
        """
        Use when printing information about an initialised fuzzy layer.
        """
        return f"in_features={self.input_size}, out_features={self.output_size}"


# Class for FDRNN model
class FDRNN(torch.nn.Module):
    def __init__(self, input_size, fuzzy_deg, fuzzy_params, batch_size, seq_length, fc_hidden_size, num_fc, output_size,
                 rec_hidden_size, num_rec):
        """
        Initialises the FDRNN with important parameters.
        """
        super(FDRNN, self).__init__()
        self.input_size = input_size            # Size of each input f_t
        self.batch_size = batch_size            # Number of total inputs in a single batch fed into the FDRNN
        self.seq_length = seq_length            # Number of consecutive timesteps (f_ts) considered in one input
        self.fc_hidden_size = fc_hidden_size    # Number of hidden cells in each fully-connected (dense/linear) layer
        self.num_fc = num_fc                    # Number of fully-connected layers
        self.output_size = output_size          # Size of each output F_t after passing through fuzzy and all deep layers
        self.rec_hidden_size = rec_hidden_size  # Number of features in a hidden state for the RNN
        self.num_rec = num_rec                  # Number of (stacked) recurrent layers
        
        # Fuzzy layer
        self.fuzzy = FuzzyLayer(self.input_size, fuzzy_deg, fuzzy_params, self.batch_size, self.seq_length)
        
        # Fully-connected layers (deep transformations)
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(self.fuzzy.output_size, self.fc_hidden_size)])
        self.fcs.extend([torch.nn.Linear(self.fc_hidden_size, self.fc_hidden_size) for i in range(self.num_fc - 2)])
        self.fcs.append(torch.nn.Linear(self.fc_hidden_size, self.output_size))
        
        # RNN layer (to calculate delta_t for each f_t for each sequence in given batch)
        self.rnn = torch.nn.RNN(self.output_size, self.rec_hidden_size, self.num_rec, nonlinearity='tanh', bias=True,
                                batch_first=True)
    
    def forward(self, x):
        """
        Perform a forward pass of input x (batch containing multiple sequences of f_ts) through the fuzzy, fully-connected and
        RNN layer and returns delta_t for each f_t in each sequence in the given batch.
        """
        x = self.fuzzy.forward(x)
        for fc in self.fcs:
            x = fc(x)
        # Next line is equivalent to delta_t, h = self.rnn(x)
        x, h = self.rnn(x)
        # Flatten from (batch_size, seq_length, rec_hidden_size)-shape tensor to 1D tensor
        # x = torch.flatten(x)
        return x


#  Batch generator for generating batches and profit function to be used during training and testing the FDRNN
def make_batch_generator(df, batch_size, seq_length):
    """
    Creates a batch generator that yields a single batch and corresponding sequence of z_ts whenever a call to the
    resulting generator object is made. Generated batches are used as raw input to the FDRNN during training and testing.
    The corresponding sequence of z_ts is used to help calculate the resulting loss after a forward pass has been made.
    """
    for i in range(0, len(df) - batch_size, batch_size):
        batch = []
        z_ts = []
        for j in range(batch_size):
            seq = df.iloc[i+j:i+j+seq_length].values
            batch.append(seq)
            z_t = df["z_t"].iloc[i+j+seq_length-1]
            z_ts.append(z_t)
        
        batch = np.reshape(np.array(batch), (batch_size, seq_length, input_size))
        z_ts = torch.reshape(torch.tensor(z_ts), (batch_size, 1)).to(device)
        yield (batch, z_ts)


# Function to calculate U_t
def calc_U_t(delta_ts, z_ts, c, pnl_over_time):
    """
    Calculate (a slice) of U_t based on the forward pass output of a single batch fed into the FDRNN. The loss function for
    the FDRNN is given by the negative of this function and hence, this function can be used to calculate the loss on the
    output of a single batch fed into the FDRNN. Each loss is used to update the parameters belonging to the fully-connected
    layers and RNN layer via regular backpropagation and backpropagation through time (BPTT) respectively. The fuzzy layer is
    excluded since it does not contain any parameters (recall used k-means to finalise the means and widths of the fuzzy nodes).
    """
    # Short way - use this for calculating gradients and updating 
    U_t = torch.sum(torch.mul(delta_ts[:, -2], z_ts) - c*torch.abs(delta_ts[:, -1] - delta_ts[:, -2]))
    # Long way - use this for plotting accumulated rewards R_ts (i.e. true U_T) over time
    if pnl_over_time != []:
        for i in range(delta_ts.shape[0]):
            pnl_over_time.append(pnl_over_time[-1] + delta_ts[i, -2].item()*z_ts[i].item() -
                                 c*abs(delta_ts[i, -1].item() - delta_ts[i, -2].item()))
    # Return U_t since loss = -1*U_t and use loss.backward() for updating all gradients
    return U_t


if __name__ == "__main__":
    # Check if GPU is available for training the FDRNN, quit program otherwise
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"NVIDIA CUDA GPU available: {torch.cuda.get_device_name(device)}\n")
    else:
        print("Could not detect any capable NVIDIA CUDA GPU!")
        print("Exiting program...")
        exit()


    asset = sys.argv[1]
    year = sys.argv[2]
    # Load in a preprocessed dataset
    df = pd.read_csv(f"../../Data Preprocessing/{asset}_{year}.csv", index_col="date")
    df.drop("p_t", axis=1, inplace=True)
    # Convert date column to datetime type
    df.index = pd.to_datetime(df.index)

    # Set important parameters
    # Number of FDRNNs to train in an Ensemble
    num_models = 10
    # Transaction cost to use for training and testing
    c = 15
    # Number of training samples
    num_train = len(df)//3
    # Training set
    train_df = df.iloc[:num_train]
    # Testing set
    test_df = df.iloc[num_train:]
    # Number of training iterations
    num_epochs = 100
    # Dimension of each feature vector f_t
    input_size = len(df.columns)
    # Number of clusters in k-means clustering (fuzzy degrees)
    k = 3


    # Set important hyperparameters
    # Number of training samples in each batch
    batch_size = 32
    # Number of time steps to feed into FDRNN at a time
    seq_length = 3
    # Input and output size to each hidden layer in the ANN component
    fc_hidden_size = 128
    # Number of fully-connected hidden layers in the ANN component
    num_fc = 4
    # Size of each output F_t after passing through fuzzy and deep representations
    output_size = 20
    # Number of features in a hidden state for the RNN component (shape of delta_t)
    rec_hidden_size = 1
    # Number of (stacked) recurrent layers
    num_rec = 1
    # Learning rate to be used during training
    eta = 0.00001
    # Weight decay (L2 regularisation penalty) to be used during training
    lambda_ = 0.000001

    # Use k-means clustering to store each fuzzy node's mean and width to be used later (i.e. fuzzy layer initialisation)
    kmeans = KMeans(n_clusters=k).fit(train_df)

    # Get cluster labels for samples and make a new column
    train_df["label"] = kmeans.labels_

    # Stores means and widths required for fuzzy layer initialisation
    fuzzy_params = []
    # Calculate the mean and variance of each dimension (feature) in each cluster using all training samples
    for label in range(k):
        for column in train_df.columns[:-1]:
            feat_given_k = train_df[train_df["label"] == label][column]
            fuzzy_params.append([feat_given_k.mean(), feat_given_k.var()])

    train_df.drop("label", axis=1, inplace=True)
    # Will always be of shape (input_size*k, 2)
    fuzzy_params = np.array(fuzzy_params, dtype=np.float32)

    # Training several FDRNN models - select "best" model as the one that has the highest final profit over the training period
    # after having trained each model for num_epochs epochs
    print("Beginning FDRNN training...")
    start_time = timeit.default_timer()

    # Initialise FDRNN model on GPU
    fdrnns = []
    fdrnns_epoch_losses = []
    fdrnns_pnl_over_time = []
    for model in range(num_models):
        print(f"\nTraining model {model + 1}")
        fdrnn = FDRNN(input_size, k, fuzzy_params, batch_size, seq_length, fc_hidden_size, num_fc, output_size,
                    rec_hidden_size, num_rec).cuda()

        # Initialise SGD optimiser with specified learning rate and L2 penalty
        optimiser = torch.optim.SGD(fdrnn.parameters(), lr=eta, weight_decay=lambda_)
        
        # Begin training the FDRNN model
        epoch_losses = []
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch: {epoch}")
            batch_generator = make_batch_generator(train_df, batch_size, seq_length)
            epoch_loss = 0
            if epoch == num_epochs:
                pnl_over_time = [0]
            for batch, z_ts in batch_generator:
                optimiser.zero_grad()
                delta_ts = fdrnn(batch)
                # Minimising loss here is equivalent to maximising total accumulated rewards
                if epoch == num_epochs:
                    loss = -1*calc_U_t(delta_ts, z_ts, c, pnl_over_time)
                else:
                    loss = -1*calc_U_t(delta_ts, z_ts, c, [])
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()

            epoch_losses.append(epoch_loss)
            
        fdrnns.append(fdrnn)
        fdrnns_epoch_losses.append(epoch_losses)
        fdrnns_pnl_over_time.append(pnl_over_time)

    end_time = timeit.default_timer()
    print(f"\nTotal training time: {end_time - start_time}s\n")

    # Plot training loss function
    plt.figure(figsize=(20, 10))
    for model in range(num_models):
        plt.plot(range(num_epochs), fdrnns_epoch_losses[model], label=f"Model {model + 1}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (-$U_T$)")
    plt.legend()
    plt.title(f"{asset}: Training Loss/Epoch based on First 4 Months of {year}")
    plt.savefig(f"{asset}_{year}_loss.png", facecolor="w")
    plt.close()

    # Plot accumulated profits over training period using best trained model after final epoch
    plt.figure(figsize=(20, 10))
    for model in range(num_models):
        plt.plot(train_df.index[seq_length-2:len(pnl_over_time)], fdrnns_pnl_over_time[model][1:], label=f"Model {model + 1}")
    plt.xlabel("Date")
    plt.ylabel("$U_T$")
    plt.legend()
    plt.title(f"{asset}: Accumulated Rewards for First 4 Months of {year}")
    plt.savefig(f"{asset}_{year}_train.png", facecolor="w")
    plt.close()

    # Save ONLY the best trained FDRNN model
    # Find best model based on which one achieved the highest final profit over the training period after having trained
    # each model for num_epochs epochs
    highest_pnl = max([fdrnns_pnl_over_time[model][-1] for model in range(num_models)])
    idx = [fdrnns_pnl_over_time[model][-1] for model in range(num_models)].index(highest_pnl)
    best_fdrnn = fdrnns[idx]

    # Save best model's training loss function
    np.save(f"{asset}_{year}_epoch_losses.npy", np.array(fdrnns_epoch_losses[idx], dtype=np.float32))
    # Save best model's training profit and loss over time
    np.save(f"{asset}_{year}_train_pnl_over_time.npy", np.array(fdrnns_pnl_over_time[idx], dtype=np.float32))
    # Save best model's parameters (weights, biases, etc.)
    torch.save(best_fdrnn.state_dict(), f"{asset}_{year}.pt")

    # Testing the best FDRNN model
    print("Beginning FDRNN testing...")
    start_time = timeit.default_timer()

    batch_generator = make_batch_generator(test_df, batch_size, seq_length)
    pnl_over_time = [0]
    with torch.no_grad():
        for batch, z_ts in batch_generator:
            delta_ts = best_fdrnn(batch)
            calc_U_t(delta_ts, z_ts, c, pnl_over_time)

    end_time = timeit.default_timer()
    print(f"Total testing time: {end_time - start_time}s\n")

    # Save best model's testing profit and loss over time
    np.save(f"{asset}_{year}_test_pnl_over_time.npy", np.array(pnl_over_time, dtype=np.float32))

    # Plot accumulated profits over testing period
    plt.figure(figsize=(20, 10))
    plt.plot(test_df.index[seq_length-2:len(pnl_over_time)], pnl_over_time[1:], label=f"Model {idx + 1}")
    plt.xlabel("Batch Idx")
    plt.ylabel("$U_T$")
    plt.legend()
    plt.title(f"{asset}: Accumulated Rewards for Last 8 Months of {year}")
    plt.savefig(f"{asset}_{year}_test.png", facecolor="w")
    plt.close()

    print(f"Successfully trained and tested FDRNN model on {asset} {year} dataset!")
