import torch
import numpy as np

def get_actions(max_reward, n_actions):
        """"Returns a list of n_action floats mirrored around zero with regular increments. The difference between the first and last values from the second and second to last respectively might be different than the differences between the rest.

        Args:
            max_reward (_type_): _description_
            n_actions (_type_): _description_

        Returns:
            _type_: _description_
        """
        inc = 2*max_reward/(n_actions - 1)
        x = np.linspace(inc, max_reward, int(n_actions/2))
        # print("x", x)
        if x[-1] != max_reward:
            x = np.r_[x, max_reward]
        mapped_actions = np.r_[-x[::-1], 0, x]
        print("mapped actions", mapped_actions)
        return torch.Tensor(mapped_actions), torch.Tensor([i for i in range(n_actions)])


def unmap(action, actions_mapped):
    # actions_mapped = self.actions_mapped
    actions_mapped = actions_mapped.cpu().numpy()
    action = action.cpu().numpy()

    actions_unmapped = [np.searchsorted(actions_mapped, a) for a in action]
    return actions_unmapped
