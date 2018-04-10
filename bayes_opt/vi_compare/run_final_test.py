import sys
import numpy as np
sys.path.append("../../")
import network_tester_base
import argparse

def main():
    ### Establish command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-log_fn", dest="log_fn", action="store",type=str,
            help="root filename for tb logs", required=True)
    parser.add_argument("-log_base_dir", dest="log_base_dir", action="store",type=str,
            help="directory for logs", required=True)
    parser.add_argument("-inference", dest="inference", action="store",type=str,
            help="inference method (VI, MLE, or MAP)", required=True)
    parser.add_argument("-dataset", dest="dataset", action="store",type=str,
            help="dataset name", required=True)
    parser.add_argument("-w_prior_sigma", dest="w_prior_sigma", action="store", type=float,
            default=None, help="prior on weights")
    parser.add_argument("-init_sigma_params", dest="init_sigma_params", action="store", type=float,
            default=1e-5, help=("initial parameter uncertainty(or weight"
            " noise parameter)"))
    parser.add_argument("-lr", dest="lr", action="store", type=float,
            default=.005, help="learning rate")

    ### Parse args
    try:
        args = parser.parse_args()
        print(args)
    except IOError as e:
        parser.error(e)

    log_fn = args.log_fn
    log_base_dir = args.log_base_dir

    f = open(log_base_dir+"/"+log_fn+"_results.txt", 'w')
    f.write("starting test")
    mean_nlogl_test, sem_nlogl_test = network_tester_base.test(
            dataset=args.dataset, inference=args.inference,
            init_sigma_params=args.init_sigma_params,
            lr=args.lr, summary_fn=log_fn, log_base_dir=log_base_dir,
            w_prior_sigma=args.w_prior_sigma, all_splits=True)
    f.write("mean_test_nlogl:\t%f\tsem_test_nlogl\t%f\n"%(mean_test_nlogl, sem_nlogl_test))
    f.close()
    print "mean_test_nlogl:\t%f\tsem_test_nlogl\t%f"%(mean_test_nlogl, sem_nlogl_test)

if __name__ == "__main__":
    main()
