import pandas as pd
import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from jax import vmap
import jax.numpy as jnp
import jax.random as random

import numpyro
# numpyro.set_platform('gpu')
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import functools
import random as python_random


# helper function for HMC inference
def run_inference(model, rng_key, X, Y, num_warmup, num_samples, num_chains=1):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )

    mcmc.run(rng_key, X, Y)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


# helper function for prediction
def predict(model, rng_key, samples, X):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None)
    return model_trace["Y"]["value"]




def monte_carlo_fit(X, Y, model, num_warmup, num_samples, seed=0, device='cpu'):
    numpyro.set_platform(device)

    X = jnp.array(X)
    Y = jnp.array(Y)

    # do inference
    rng_key, rng_key_predict = random.split(random.PRNGKey(seed))
    samples = run_inference(model, rng_key, X, Y, num_warmup, num_samples)

    return samples, rng_key, rng_key_predict


def monte_carlo_pred(model, samples, rng_key, rng_key_predict, X_test, percentile_values = [2.25, 97.75]):
    # predict Y_test at inputs X_test
    num_samples = len(samples[list(samples.keys())[0]])
    num_chains = 1
    vmap_args = (
        samples,
        random.split(rng_key_predict, num_samples * num_chains),
    )
    predictions = vmap(
        lambda samples, rng_key: predict(functools.partial(model, with_noise=False), rng_key, samples, X_test)
    )(*vmap_args)
    predictions = predictions[..., 0]

    predictions_noisy = vmap(
        lambda samples, rng_key: predict(functools.partial(model, with_noise=True), rng_key, samples, X_test)
    )(*vmap_args)
    predictions_noisy = predictions_noisy[..., 0]


    mean = jnp.mean(predictions, axis=0)
    percentiles = np.percentile(predictions, percentile_values, axis=0)
    percentiles_noisy = np.percentile(predictions_noisy, percentile_values, axis=0)

    samples = [predictions[i,:] for i in range(predictions.shape[0])]

    return mean, percentiles, percentiles_noisy, samples
