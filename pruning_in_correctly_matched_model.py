import sys
import network
import utils
import matplotlib.pyplot as plt
import tensorflow as tf
import flow_network_bayes
import numpy as np
from scipy import stats
import argparse

def main():
    ### Establish command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-log_fn", dest="log_fn", action="store",type=str,
            help="root filename for tb logs", required=True)
    parser.add_argument("-log_base_dir", dest="log_base_dir", action="store",type=str,
            default='logs/', help="directory for tensorboard logs")
    parser.add_argument("-n_hidden", dest="n_hidden", action="store",type=str,
            default='50', help=("number of units in each hidden layer, comma"
            "separated"))

    ### Parse args
    try:
        args = parser.parse_args()
        print(args)
    except IOError as e:
        parser.error(e)
    n_hidden_units = [int(n) for n in args.n_hidden.split(",")]
    n_samples_to_plot = 3

    # run several experiments on synthetic datasets, each fitting to a different
    # number of points.
    n_pts_vals = [5, 25, 100]
    w_prior_sigma = 5.

    # We initialize our q to have relatively small uncertainty
    init_sigma_params = 0.05

    base_dir = args.log_base_dir

    for i in range(n_samples_to_plot):
        n_pts = n_pts_vals[i]
        print "sample %d/%d"%(i,n_samples_to_plot)
        tf.reset_default_graph()
        net = flow_network_bayes.flow_network_bayes(
                summary_fn=args.log_fn+"_%d"%i,
                n_hidden_units=n_hidden_units,
                n_epochs=10000,
                standardize_data=False,

                n_samples=20,

                # When set to 1, simply a Gaussian predictive distribution.
                n_flows=1,
                init_sigma_params=init_sigma_params,
                learn_sigma_weights=True, # set to False for weight noise model
                lr=1e-2,
                dataset='toy_small',
                log_base_dir=base_dir,
                display_freq=1000,
                init_sigma_obs=3.0,

                # set True to fit the level of observation noise by
                # minimizing the VFE
                learn_sigma_obs=False,

                # number of points in synthetic dataset
                n_pts=n_pts,
                log_image_summary=True,

                # prior over weights (same in all layers)
                w_prior_sigma=w_prior_sigma
                )

        z = tf.contrib.distributions.Normal(loc=0., scale=1.).sample(n_pts)
        sample_y = net.flows[0].project(z)

        n_pts_plot = 100
        x_plot = np.linspace(-3,3,n_pts_plot)
        y_plot = net.flows[0].project(x_plot[None]*0.)
        y_plot_sd2p = net.flows[0].project(2.*np.ones(x_plot.shape))
        y_plot_sd2m = net.flows[0].project(2.*-np.ones(x_plot.shape))
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            # Construct the training set
            sample_x = sess.run([z])[0]
            sample_y_val = sess.run([sample_y],
                    feed_dict={net.x:sample_x[:,None],
                        })[0][0]
            print "sample_x.shape", sample_x.shape
            print "sample_y_val.shape", sample_y_val.shape
            net.split(0, sample_x[:, None], sample_y_val[:, None])

            # Plot state before training
            y_plot_val, y_plot_sd2p_val, y_plot_sd2m_val = sess.run([y_plot,
                y_plot_sd2p, y_plot_sd2m], feed_dict={net.x:x_plot[:,None]})
            plt.clf()
            plt.scatter(net.X_train, net.Y_train,c='k',s=50)
            plt.plot(x_plot, y_plot_val[0],'g-', x_plot, y_plot_sd2m_val[0],'g--',
                    x_plot, y_plot_sd2p_val[0],'g--')
            plt.title("sample %d before training (data and predictive "
                    "distribution)"%i)
            plt.savefig(base_dir+args.log_fn+"_npts_%03d_pretraining_12-24.png"%n_pts)

            # Plot state after training
            net.summary_path = base_dir+args.log_fn+"_npts_%03d"%n_pts
            net.train(sess)
            # plot 5 samples from the posterior
            plt.clf()
            plt.scatter(net.X_train, net.Y_train, c='k',s=50)
            for _ in range(5):
                y_plot_val = sess.run([y_plot], feed_dict={net.x:x_plot[:,
                    None]})[0]
                plt.plot(x_plot, y_plot_val[0],c='g')
            plt.title("sample %d post training 1"%i)
            plt.savefig(base_dir+args.log_fn+"_npts_%03d_posttraining_12-24.png"%n_pts)
            n_unpruned = utils.plot_pruning2(net,
                    base_dir+args.log_fn+"_npts_%03d"%n_pts)
            print "N units not pruned: %d/%d"%(n_unpruned, 50)

            #net.train(sess)
            #plt.clf()
            #plt.scatter(sample_x, sample_y_val[0],c='k',s=3,marker='*')
            #for _ in range(5):
            #    y_plot_val = sess.run([y_plot], feed_dict={net.x:x_plot})[0]
            #    plt.plot(x_plot, y_plot_val[0],c='g')
            #plt.title("sample %d post training 2"%i)
            #plt.savefig("./func_sample_%d_posttraining2.png"%i)

if __name__ == "__main__":
    main()
