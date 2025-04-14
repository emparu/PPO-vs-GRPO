# PPO vs. GRPO Comparison on RL Environments

This repository contains implementations and comparisons of Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO) algorithms on standard reinforcement learning environments: CartPole and LunarLander.

The goal is to understand their performance differences outside the Large Language Model (LLM) domain where GRPO was introduced in the DeepSeekMath paper.

## Algorithm Modifications for Comparison

To facilitate a meaningful comparison, several modifications were made to the standard algorithms and the specific GRPO formulation:

### Common Loss Structure Modification

Both PPO and GRPO implementations here use a policy loss structure incorporating both KL divergence and entropy terms:

*   **Policy Loss ≈ - (Clipped Surrogate Objective) + β * KL(π_θ || π_ref) - η * Entropy(π_θ)**
*   **KL Divergence Penalty:** A penalty term using the KL divergence between the current policy (π_θ) and a reference policy (π_ref - the policy before the update iteration) is included, scaled by `KL_BETA` (β). This aims to stabilize training by preventing large policy shifts from one iteration to the next.
*   **Entropy Bonus:** The standard entropy bonus term, scaled by `ENTROPY_COEFF` (η), is subtracted from the loss (equivalent to adding `η * Entropy` to the objective being maximized). This encourages exploration by preventing the policy from becoming prematurely deterministic.

*Rationale:* The KL term regularizes the policy update, while the entropy term promotes exploration. The reference policy (`π_ref`) is simply the policy state before the current multi-epoch update begins.

### PPO Modifications

*   **Batch Collection:** Instead of collecting a fixed number of *steps* per iteration, this PPO implementation collects a fixed number of *rollouts* (`GROUP_SIZE`), mirroring the GRPO data collection process.
*   **Rationale:** This provides a more direct comparison regarding the amount and structure of data used per update cycle between PPO and GRPO. It can also be more sample-efficient in early training when episodes are short.
*   **Loss:** Includes the KL penalty described above in addition to the standard PPO clipped objective, value loss, and entropy bonus.

### GRPO Modifications

This GRPO implementation is adapted for standard RL environments and differs from the DeepSeek paper's LLM context:

*   **Loss Averaging:** The surrogate loss and KL divergence terms are averaged over *all steps* concatenated from the group's rollouts. The `(1 / |rollout_length|)` weighting per rollout described in the paper is *not* applied. This treats each transition step equally, common in many policy gradient implementations.
*   **Advantage Calculation:** Advantages (`Â_i,t`) are calculated as follows:
    1.  Compute raw discounted returns-to-go (G_t = Σ γ^k * r_{t+k}) for each step within its rollout.
    2.  Collect *all* G_t values from the entire group.
    3.  Normalize these collected G_t values using their group-wide mean and standard deviation. These normalized values serve as the advantages in the loss function.
*   **Rationale:** This advantage calculation method uses normalized returns-to-go, a standard technique in RL. It avoids issues encountered when trying to directly apply the paper's "Process Supervision" reward normalization in environments like CartPole (where constant rewards become zero after normalization).

      
### GRPO Modifications

This GRPO implementation removes certain features specific to the DeepSeek paper's LLM context to apply it to standard RL environments:

*   **Loss Averaging:** The surrogate loss and KL divergence terms are averaged over *all steps* concatenated from the group's rollouts. The `(1 / |rollout_length|)` weighting per rollout described in the paper's formula is *not* applied. This simplification treats each transition step equally, as is common in many policy gradient implementations.
*   **Advantage Calculation:** The paper's "Process Supervision" advantage calculation (involving pre-normalized rewards) is replaced with a standard RL approach using normalized returns-to-go:
    1.  Compute raw discounted returns-to-go (G_t = Σ γ^k * r_{t+k}) for each step within its rollout.
    2.  Collect *all* G_t values from the entire group.
    3.  Normalize these collected G_t values using their group-wide mean and standard deviation. These normalized values serve as the advantages (`Â_i,t`) in the loss function.
*   **Rationale for Advantage Change:** This standard advantage calculation (using normalized returns-to-go) was used because the paper's specific "Process Supervision" method, which involves normalizing individual step rewards *before* calculating advantages, appears highly tailored to their LLM step-verification training context. In standard RL environments, this approach is non-standard because the implicit reward signal guiding the training updates would differ significantly from the actual cumulative (discounted) reward that we aim to maximize.

## Implementation Notes

*   With these modifications, this GRPO implementation resembles a PPO variant that omits the critic network and uses group-normalized returns-to-go directly as advantages, alongside the KL penalty against the previous policy iteration.
*   The KL penalty calculation uses a reference model (`actor_ref`) which is a snapshot of the policy before the update epochs begin. While the *log probabilities* from the rollout phase (`log_probs_old`) could theoretically be reused, the current code uses `actor_ref` for recalculation within the update loop for structural consistency (this is slightly less computationally efficient but functionally equivalent for the purpose of the KL term).
*   Note: You can run these notebooks on Kaggle. Example (GRPO CartPole version): [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/emanuelruzak/grpo-cartpole)

## Results and Discussion

*   **LunarLander Performance Dynamics:** Both PPO and GRPO initially learn to land the agent safely, reaching an average reward plateau around ~150. This phase is often characterized by increasingly long episode lengths, as the agents learn careful, fuel-consuming landing procedures without prioritizing ending the episode quickly upon successful contact.
*   **Overcoming the Plateau:** PPO, in these experiments, consistently manages to overcome this ~150 reward barrier. It subsequently optimizes further, learning to reduce episode duration and fuel consumption, leading to average scores significantly above 200. GRPO, however, fails to transition past this plateau and remains stuck in the long-episode, moderate-reward regime.
*   **Hypothesis Testing (Bias towards Long Episodes):** A common observation in RL experimentation is that certain algorithm dynamics might inadvertently favor longer episodes. To test if this was the primary issue with GRPO, a converged PPO agent (score > 200, representing efficient landings) was used to initialize the GRPO agent's policy network. Subsequent training iterations using the GRPO update rule did *not* cause the agent to revert to longer episodes or significantly degrade its high score.
*   **Interpretation:** This test suggests that the GRPO update mechanism (as implemented) is not inherently biased towards creating unnecessarily long episodes. Instead, the failure to surpass the ~150 plateau indicates a potential difficulty in performing the specific type of exploration or optimization required to discover the more efficient landing strategies from the "safe but inefficient" landing policy. GRPO can *maintain* a good, efficient policy, but struggles to *find* it through learning when starting from less optimal policies stuck at the plateau. PPO's actor-critic structure might provide more effective gradients or exploration signals for this specific optimization phase.

## Conclusion

The results in these standard RL benchmarks, particularly LunarLander, suggest that GRPO, adapted in this manner, may be less effective than actor-critic PPO at navigating certain optimization challenges. Specifically, it appears to struggle with transitioning from moderately successful but inefficient policies (the ~150 reward plateau) to highly optimized, efficient solutions, even though it can maintain such solutions once found.

This contrasts with GRPO's reported success fine-tuning LLMs (DeepSeekMath), which may rely more on refinement of existing skills. LunarLander, however, requires exploring and discovering fundamentally new, efficient strategies. These results hint that GRPO might excel more at refinement than complex exploration, but this observation is preliminary based on these specific RL tests.

## References and Acknowledgements

*   The Group Relative Policy Optimization (GRPO) algorithm and its application to LLMs are detailed in:
    *   Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y.K., Wu, Y., & Guo, D. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300v3 [cs.CL]. [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
*   The Proximal Policy Optimization (PPO) algorithm is described in:
    *   Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347v2 [cs.LG]. [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
*   The base PPO implementation used in this repository was adapted from the code provided by FareedKhan-dev:
    *   [https://github.com/FareedKhan-dev/all-rl-algorithms](https://github.com/FareedKhan-dev/all-rl-algorithms)
