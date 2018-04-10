import sys
import utils
import pickle
import tensorflow as tf
import flow_network_bayes
import numpy as np
from scipy import stats
import argparse

def main():
    ### Establish command line arguments
    parser = argparse.ArgumentParser()

    # Task Specification
    parser.add_argument("-dataset", dest="dataset", action="store",type=str,
            help="dataset name", required=True)
    parser.add_argument("-all_splits", dest="all_splits", action="store_true",
            default=False, help="if to test on all splits")

    # Optimization related parameters
    parser.add_argument("-batch_size", dest="batch_size", action="store",type=int,
            default=10**10, help="batch size")
    parser.add_argument("-epochs", dest="epochs", action="store",type=int,
            default=7000, help="number of epochs")
    parser.add_argument("-epoch_switch_opt", dest="epoch_switch_opt", action="store",type=int,
            default=10**10, help=("epoch when we start learning"
            "hyperparams(if they are set to being trainable) and to begin"
            "annealing in the KL (if we have chosen to do so)"))
    parser.add_argument("-lr", dest="lr", action="store", type=float,
            default=.005, help="learning rate")

    # Initialization Parameters
    parser.add_argument("-init_sigma_params", dest="init_sigma_params", action="store", type=float,
            default=1e-4, help="initial uncertainty in weights")
    parser.add_argument("-lmbda_init", dest="lmbda_init", action="store", type=float,
            default=1.0, help=("initial level of interpolation between",
            "homoscedastic and heteroscedastic."))

    # Model Specification
    parser.add_argument("-n_hidden", dest="n_hidden", action="store",type=str,
            default='50', help=("number of units in each hidden layer, comma"
            "separated"))

    # Inference Specification
    parser.add_argument("-inference", dest="inference", action="store",type=str,
            help="inference method (VI, MLE, or MAP)", required=True)
    parser.add_argument("-bayes_layers", dest="bayes_layers", action="store",type=str,
            default=None, help=("layers to learn approximate posterior in, comma"
            "separated"))
    parser.add_argument("-w_prior_sigma", dest="w_prior_sigma",
            action="store", type=float, default=None, help="w_prior_sigma")

    # Hyper-parameter learning control
    parser.add_argument("-learn_sigma_weights", dest="learn_sigma_weights", action="store_true",
            default=False, help=("set to learn uncertainty in weights, else "
            "weight noise"))
    parser.add_argument("-learn_ls", dest="learn_ls", action="store_true",
            default=False, help=("set to learn length_scale "
                "by marginal likelihood"))
    parser.add_argument("-learn_beta_std", dest="learn_beta_std", action="store_true",
            default=False, help=("set to learn beta_std "
                "by marginal likelihood"))

    # Logging related paramters
    parser.add_argument("-log_fn", dest="log_fn", action="store",type=str,
            help="root filename for tb logs", required=True)
    parser.add_argument("-log_base_dir", dest="log_base_dir", action="store",type=str,
            default='logs/', help="directory for tensorboard logs")
    parser.add_argument("-display_freq", dest="display_freq", action="store",type=int,
            default=100, help="display step frequency")
    parser.add_argument("-plot_title", dest="plot_title", action="store",type=str,
            help="title to use for plots", default=None)
    parser.add_argument("-plot_pts", dest="plot_pts", action="store_true",
            default=False, help=("plot points and intervals onto predictive"
            "distribution."))
    parser.add_argument("-plot_pruning", dest="plot_pruning", action="store_true",
            default=False, help="if to make pruning plots")


    ### Parse args
    try:
        args = parser.parse_args()
        print(args)
    except IOError as e:
        parser.error(e)
    n_hidden_units = [int(n) for n in args.n_hidden.split(",")]

    if args.bayes_layers is None:
        # This is learning approximate posterior for every layer.
        bayes_layers = None
    elif len(args.bayes_layers) == 0:
            bayes_layers = []
    else:
        bayes_layers = [int(l) for l in args.bayes_layers.split(",")]

    dataset = "./../DropoutUncertaintyExps/"+args.dataset
    n_samples = 20 if  args.inference == 'VI' else 1

    net = flow_network_bayes.flow_network_bayes(
            summary_fn=args.log_fn,
            init_sigma_params=args.init_sigma_params,
            n_hidden_units=n_hidden_units,
            n_epochs=args.epochs,
            n_flows=1,
            lr=args.lr,
            dataset=dataset,
            log_base_dir=args.log_base_dir,
            display_freq=args.display_freq,
            n_samples=n_samples,
            plot_pts=args.plot_pts,
            plot_title=args.plot_title,
            bayes_layers=bayes_layers,
            batch_size=args.batch_size,

            learn_ls=args.learn_ls,
            learn_sigma_weights=args.learn_sigma_weights,
            learn_beta_std=args.learn_beta_std,
            inference=args.inference,
            w_prior_sigma=args.w_prior_sigma,
            epoch_switch_opt=args.epoch_switch_opt,
            )
    with open(args.log_base_dir +"/"+args.log_fn+"_args.pkl",'w') as f:
        pickle.dump(args, f)
    print "saved args"


    if args.all_splits:
        n_splits = 20 if args.dataset[:4] != "prot" else 5
        train_nlog_ls, test_nlog_ls, train_rmses, test_rmses = [], [], [], []
        for i in range(n_splits):
            net.split(i)
            net.summary_path = net.log_base_dir+args.log_fn+"_split%d"%i
            returns = net.train()
            train_nlog_l, test_nlog_l = returns[0], returns[1]
            train_nlog_ls.append(train_nlog_l)
            test_nlog_ls.append(test_nlog_l)
            if len(returns) == 4:
                rmse_train, rmse_test = returns[2], returns[3]
                train_rmses.append(rmse_train)
                test_rmses.append(rmse_test)
            print "split %d"%i
            print "\tnlog_l --- Train: %f\tTest: %f"%(train_nlog_l,test_nlog_l)
        train_nlog_ls, test_nlog_ls = np.array(train_nlog_ls), np.array(test_nlog_ls)
        print "\n\nPerformance across all Splits"
        print "Train nlog_l:  mean-",np.mean(train_nlog_ls), "\tstderr-", stats.sem(train_nlog_ls)
        print "Test nlog_l:  mean-",np.mean(test_nlog_ls), "\tstderr-", stats.sem(test_nlog_ls)
        if len(returns) == 4:
            train_rmses, test_rmses = np.array(train_rmses), np.array(test_rmses)
            print "Train rmse:  mean-",np.mean(train_rmses), "\tstderr-", stats.sem(train_rmses)
            print "Test rmse:  mean-",np.mean(test_rmses), "\tstderr-", stats.sem(test_rmses)

    else:
        net.split(0)
        net.summary_path = net.log_base_dir+args.log_fn
        print net.train()

    if args.plot_pruning:
        utils.plot_pruning2(net,net.log_base_dir+args.log_fn+"_pruning")

if __name__ == "__main__":
    main()
