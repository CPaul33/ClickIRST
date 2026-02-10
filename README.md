# ClickIRST: Towards Interactive Segmentation for Infrared Small Target


## Datasets
We used the WideIRSTD for both training and testing, SIRST3 only for testing.

### Commands for runing demo
```
python demo.py --checkpoint=experiments/WideIRSTD_plainvit_base448/checkpoints/last_checkpoint.pth --gpu=0
```
  
### Commands for training
```
python train.py models/plainvit_base/plainvit_base448_WideIRSTD_itermask.py \
--batch-size=8 \
--gpus=0,1
```
### Commands for testinging
  ```
  python scripts/evaluate_model.py NoBRS --gpu=0 --checkpoint=experiments/ckick10/checkpoints/last_checkpoint.pth --eval-mode=cvpr --datasets="xxxxxx"
  ```

  ### Commands for ranking infrared image complexity
  ```
  python scripts/compute_irst_weights.py --dataset_path="xxxxxx"
  ```

  ## More code for other models will be open later
