"""
Implement option to seed random state to start with observer-specific initial states.
"""
import torch


def seeded_random_states(target_shape, seeds, rand_fn=torch.randn):
    '''
    target_shape is batchsize x state_dim
    '''

    state_old = torch.get_rng_state()

    def sample_state(seed):
        torch.manual_seed(seed)
        return rand_fn(1, target_shape[-1])

    states = torch.cat([sample_state(s) for s in seeds])
    
    torch.set_rng_state(state_old)
    return states


def seeded_random_states_like(state_tensor, seeds, rand_fn=torch.randn):
    return seeded_random_states(state_tensor.size(), seeds, rand_fn=rand_fn)


if __name__ == "__main__":
    s1 = seeded_random_states((5, 20), [1, 2, 3])
    s2 = seeded_random_states((5, 20), [3, 2, 21])
    s1l = seeded_random_states_like(torch.zeros(5, 20), [1, 2, 3])
    s2l = seeded_random_states_like(torch.zeros(5, 20), [3, 2, 21])

    assert (s1 == s1l).all()
    assert (s2 == s2l).all()

    assert (s1[2, :] == s2[0, :]).all()
    assert (s1[1, :] == s2[1, :]).all()
