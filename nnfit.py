#! /usr/bin/env python

import jax
import jax.numpy as np
import jax.random as rjax
import flax
from flax import nn, optim, serialization

from lib.params import mdn, mdstp, mpip

""" Train NN to learn 3D or 5D pdf """

class NN(nn.Module):
    def apply(self, x):
        x = nn.Dense(x, features=32)
        x = nn.sigmoid(x)
        x = nn.Dense(x, features=32)
        x = nn.sigmoid(x)
        x = nn.Dense(x, features=1)
        return nn.sigmoid(x)


@flax.struct.dataclass
class TrainState:
    optimizer: optim.Optimizer


def do_fit(data, steps=100):
    """ """
    norm_sample_size = 10**6
    norm_space = (
        (-2.5, 15),  # energy range (MeV)
        (0, 22),  # T(DD) range (MeV)
        (2.004, 2.026),  # m(Dpi) range (GeV)
    )

    _, initial_params = NN.init(rjax.PRNGKey(0), data)
    print('Model initialized:')
    print(jax.tree_map(np.shape, initial_params))

    rng = rjax.PRNGKey(10)
    rng, key1, key2, key3 = rjax.split(rng, 4)
    keys = [key1, key2, key3]
    norm_sample = np.column_stack([
        rjax.uniform(rjax.PRNGKey(key[0]), (norm_sample_size,), minval=lo, maxval=hi)\
            for key, (lo, hi) in zip(keys, norm_space)
    ])

    def loglh_loss(model):
        """ loss function for the unbinned maximum likelihood fit """
        return -np.sum(np.log(model(data))) +\
            data.shape[0] * np.log(np.sum(model(norm_sample)))

    model = nn.Model(NN, initial_params)
    adam = optim.Adam(learning_rate=0.03)
    optimizer = adam.create(model)

    for i in range(steps):
        loss, grad = jax.value_and_grad(loglh_loss)(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        print(f'{i}/{steps}: loss: {loss:.3f}')


    with open('nn_model_3d.dat', 'wb') as ofile:
        state = TrainState(optimizer=optimizer)
        data = flax.serialization.to_bytes(state)
        print(f'Model serialized, num bytes: {len(data)}')
        ofile.write(data)


def get_vars(data):
    """ """
    e, mddsq, md1pisq = [data[:,i] for i in range(3)]
    s = (e + mdn + mdstp)**2
    md2pisq = s + 2*mdn**2 + mpip**2 - mddsq - md1pisq
    tdd = (np.sqrt(mddsq) - 2*mdn)*10**3
    md1pi = np.sqrt(md1pisq)
    return (e, tdd, md1pi, mddsq, md1pisq, md2pisq)


def main():
    smc = np.load('mc_ddpip_3d_smeared.npy')
    print(smc.shape)

    data = np.column_stack(get_vars(smc)[:3])
    print(data.shape)

    do_fit(data, 100)


if __name__ == '__main__':
    main()
