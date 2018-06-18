import numpy as np


def adjust_action_table(action_table):
    """Adjusts the action table to a general view.

Args:
    action_table (np.array with shape=(n_actions, 4) or (4) or (0)):
        table of actions with columns ["dVx", "dVy", "dVz", "time to request"].

    Returns:
            result_table (np.array): general action table.

    """
    if not action_table.size:
        result_table = action_table
    else:
        result_table = action_table.reshape((-1, 4))
        result_table[-1, -1] = np.nan
        i = 0
        while True:
            if i == result_table.shape[0] - 1:
                break

            if result_table[i, 3] == 0:
                # merge actions if time to next action == 0
                result_table[i + 1] += result_table[i]
                result_table = np.delete(result_table, i, axis=0)
            elif np.all(result_table[i, :3] == 0) and i != 0:
                # merge empty actions
                result_table[i - 1] += result_table[i]
                result_table = np.delete(result_table, i, axis=0)
            else:
                i += 1

    return result_table
