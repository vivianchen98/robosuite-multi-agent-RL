# robosuite-multi-agent-RL

## Description
Multi-agent SAC implemented in the robosuite setting, with the aim at completing collaborative manipulation tasks (e.g. TwoArmHandover).

## Getting Started
### Installing
See dependencies needed in robosuite-benchmark (e.g. rlkit, viskit, etc.)

### Running experiment

Train
```
python scripts/train_multi.py --variant variant_multi_twoarmhandover.json
```
Rollout
```
python scripts/rollout.py --load_dir log/runs/TwoArmLift-PandaPanda-OSC-POSE-SEED1/TwoArmLift_PandaPanda_OSC_POSE_SEED1_2021_11_22_23_47_09_0000--s-0 --horizon 200 --camera frontview
```
### Acknowledgments

* [robosuite-benchmark](https://robosuite.ai/docs/algorithms/benchmarking)
* [rlkit](https://github.com/rail-berkeley/rlkit/tree/b7f97b2463df1c5a1ecd2d293cfcc7a4971dd0ab)