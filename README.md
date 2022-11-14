# EN 520.655 Project 1

Nisarg A. Shah {nshah82@jhu.edu}
Yasiru Ranasinghe {dranasi1@jhu.edu}

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training
```
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```

## Fast Gradient Signed Method attack
```
# Start training with:
python fgsm.py
```

## Projected Gradient Descent L-2 norm attack
```
# Start training with:
python pgd_l2.py
```

## Projected Gradient Descent L-infinity norm attack
```
# Start training with:
python pgd_linf.py
```

## Carlini-Wagner attack
```
# Start training with:
python cw.py
```

## Adversarial training
```
# Start training with:
python adversarial_training.py
```