from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.local.state_local import StateLOCAL
from utils.beam_search import beam_search


class LOCAL(object):

    NAME = 'local'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1


        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) , None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return LOCALDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateLOCAL.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = LOCAL.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class LOCALDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution='unit'):
        super(LOCALDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            if distribution == 'unit':  # Sample points randomly in [0, 1] square
                self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]
            elif distribution == 'scale': # Sample y_max randomly in [0, 1], then sample points in the rectangular area
                y_max = torch.rand(num_samples)
                self.data = [torch.cat((torch.rand(size, 1), torch.rand(size, 1)*y_max[i]), dim=1) for i in range(num_samples)]
            else:
                raise NotImplementedError
                
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
