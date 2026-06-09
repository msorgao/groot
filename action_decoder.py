        # Per-step temporal ensemble: for each position t in the chunk,
        # compute weighted average of its historical predictions
        action_chunk = np.zeros((self.temporal_size, 7))
        for t in range(self.temporal_size):
            weights = self.action_buffer_mask[:, t:t + 1] * self.temporal_weights
            total_weight = np.sum(weights)
            if total_weight > 0:
                action_chunk[t] = (
                    np.sum(self.action_buffer[:, t, :] * weights, axis=0) / total_weight
                )
            else:
                action_chunk[t] = pred_action[0, t]
 
        # Un-normalize each action in the chunk
        action_chunk = np.where(
            mask[None, :],
            0.5 * (action_chunk + 1) * (action_high[None, :] - action_low[None, :]) + action_low[None, :],
            action_chunk,
        )
 
        return action_chunk
