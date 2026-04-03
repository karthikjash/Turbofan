from hmmlearn import hmm
import numpy as np

def build_hmm(n_states=4, n_features=8):

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=100,
        init_params="",  # not initializing means and covars


        
        params="mc"
    )

    # Start probabilities
    model.startprob_ = np.array([1.0, 0.0, 0.0, 0.0])

    # Left-right transition matrix
    model.transmat_ = np.array([
        [0.7, 0.3, 0.0, 0.0],
        [0.0, 0.6, 0.4, 0.0],
        [0.0, 0.0, 0.6, 0.4],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Initialize emission parameters
    model.means_ = np.random.randn(n_states, n_features)
    model.covars_ = np.ones((n_states, n_features))

    return model
