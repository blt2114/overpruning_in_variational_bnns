import sys
from scipy import stats
import numpy as np
sys.path.append("../../../../")
sys.path.append("../")
import flow_network_bayes
import flow_network_bayes_struct
import argparse

def test(dataset, inference, summary_fn, w_prior_sigma=None,
        bayes_layers=None, nflows=1, lmbda=0.5, beta_std=0.,
        init_sigma_params=1e-5, lr=0.005, all_splits=False,
        log_base_dir="./logs/"):
    """Test takes the length scale and learning rate

    args:
        all_splits, if all_splits is set to true, we use 20 split and
        return the mean nlogl and sem of the nlogls
    """
    if init_sigma_params != 1e-5 : assert inference == 'VI_weight_noise'

    if inference == 'VI' or inference == 'VI_full_cov':
        learn_sigma_weights=True
    else:
        learn_sigma_weights=False

    if inference == 'VI_full_cov':
        net_class = flow_network_bayes_struct.flow_network_bayes_mvn
        inference = 'VI'
    elif inference == 'VI_weight_noise':
        net_class = flow_network_bayes.flow_network_bayes
        inference = 'VI'
    else:
        net_class = flow_network_bayes.flow_network_bayes

    n_samples = 20 if inference == 'VI' else 1
    if bayes_layers is not None: assert inference == 'VI'
    if lr != 0.005: assert inference == 'MLE'
    epochs = 2000 if inference == 'MLE' else 5000

    if lmbda != 0.5 or beta_std != 0.: assert n_flows is not 1

    data_dir_base = "../../DropoutUncertaintyExps/"

    net = net_class(
            summary_fn=summary_fn,
            lr=lr,
            init_sigma_params=init_sigma_params,
            length_scale=1.,
            n_hidden_units=[50],
            bayes_layers=bayes_layers,
            n_epochs=epochs,
            n_flows=nflows,
            w_prior_sigma=w_prior_sigma,
            dataset=dataset,
            log_base_dir=log_base_dir,
            display_freq=1000,
            data_dir_base=data_dir_base,
            n_samples=n_samples,
            batch_norm=False,
            log_image_summary=False,
            learn_ls=False,
            inference=inference,
            epoch_switch_opt=10**10,
            beta_std=beta_std,
            lmbda=lmbda,
            learn_sigma_weights=learn_sigma_weights
            )
    n_splits = 20 if all_splits else 1
    train_nlog_ls, test_nlog_ls, train_rmses, test_rmses = [], [], [], []
    for i in range(n_splits):
        net.split(i)
        net.summary_path = net.log_base_dir+net.summary_fn+"_split%d"%i
        returns = net.train()
        train_nlog_l, test_nlog_l = returns[0], returns[1]
        train_nlog_ls.append(train_nlog_l)
        test_nlog_ls.append(test_nlog_l)
        if len(returns) == 4:
            rmse_train, rmse_test = returns[2], returns[3]
            train_rmses.append(rmse_train)
    mean_train_nlogl = np.mean(np.array(train_nlog_ls))
    mean_test_nlogl = np.mean(np.array(test_nlog_ls))
    if all_splits:
        mean_test_sem = stats.sem(np.array(test_nlog_ls))
        return mean_test_nlogl, mean_test_sem
    else:
        return mean_train_nlogl, mean_test_nlogl
