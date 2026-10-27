Pre-overflow experiment checkpoint

Only ppo_30720.pt is retained in Git for continuation and development evaluation. Its tensors and preceding updates are finite. Later local ppo.pt / ppo_61440.pt and the unfiltered history follow an infinite KL loss and must not be promoted; they are ignored. The original trainer and overflow log are under ../overflow_trial/. Stable continuation starts from these 30,720-transition weights with a new optimizer and fresh seeds under ../stable_training/. No model in this folder is accepted for deployment.
