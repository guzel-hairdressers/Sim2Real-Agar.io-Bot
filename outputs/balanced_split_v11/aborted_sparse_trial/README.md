Aborted development trial; never evaluated or deployed.

The early pilot normalized the split loss by eligible actions in each minibatch. Rare feasible actions therefore received excessive weight. It was stopped before using any result. The replacement uses the usual mean over all rollout transitions, with unavailable actions contributing zero, and a lower stochastic exploration temperature to keep sampling closer to the existing deterministic decision boundary. Historical source, configuration and log are retained here.
