import torch

def seeded_random_states(target_shape, seeds, rand_fn=torch.randn, batch_dim=0, device=None):
    '''
    target_shape is state_dim with batch_size in dimension batch_dim
    '''

    state_old = torch.get_rng_state()

    def sample_state(seed):
        torch.manual_seed(seed)
        target_shape_list = list(target_shape)
        target_shape_list[batch_dim] = 1
        return rand_fn(target_shape_list, **dict(device=device) if device is not None else {})

    states = torch.cat([sample_state(s) for s in seeds], dim=batch_dim)
    
    torch.set_rng_state(state_old)
    return states

def seeded_random_states_like(state_tensor, seeds, rand_fn=torch.randn, batch_dim=0):
    return seeded_random_states(state_tensor.size(), seeds, rand_fn=rand_fn, batch_dim=batch_dim, device=state_tensor.device)

if __name__ == "__main__":
    s1 = seeded_random_states((5, 20), [1, 2, 3])
    s2 = seeded_random_states((5, 20), [3, 2, 21])
    s1l = seeded_random_states_like(torch.zeros(5, 20), [1, 2, 3])
    s2l = seeded_random_states_like(torch.zeros(5, 20), [3, 2, 21])

    assert (s1 == s1l).all()
    assert (s2 == s2l).all()

    assert (s1[2, :] == s2[0, :]).all()
    assert (s1[1, :] == s2[1, :]).all()
