""" Toy MC Generation tools """

import numpy as np

class MCProducer():
    """ """

    def __init__(self, pdf, phsp, maj=None):
        """  """
        self.pdf = pdf
        self.phsp = phsp
        self.ndim = len(phsp)
        self.chunk_size = 10**6
        self.maj = self.assess_maj() if maj is None else maj
        print(f'maj: {self.maj:.2f}')


    def __call__(self, chunks):
        """ """
        result = []
        total = 0
        for i in range(chunks):
            data = self.get_chunk()
            xi = self.get_xi()
            y = self.pdf(data)
            result.append(data[y>xi].reshape(-1, self.ndim))

            current = result[-1].shape[0]
            total += current
            print(f'{i}/{chunks}: {current:5d} events (total {total:7d})')

        return np.vstack(result)

    
    def get_chunk(self):
        """ Generates uniform sample in the X space """
        # Generate random number in the unit box
        data = np.random.random((self.chunk_size, self.ndim))

        # Rescale each dimension
        for i in range(self.ndim):
            rng = self.phsp[i]
            data[:,i] = rng[0] + data[:,i] * (rng[1] - rng[0])

        return data


    def get_xi(self):
        """ Random variable for the accept-reject algorithm """
        return np.random.random((self.chunk_size, 1)) * self.maj


    def assess_maj(self):
        """ """
        print('assessing majorant...')
        return 1.1*max([np.max(self.pdf(self.get_chunk())) for _ in range(10)])
