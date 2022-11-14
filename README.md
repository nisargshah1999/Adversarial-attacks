# EN 520.655 Project 1

Nisarg A. Shah<sup>1</sup> {nshah82@jhu.edu} and
Yasiru Ranasinghe<sup>1</sup> {dranasi1@jhu.edu}

<sup>1</sup> Johns Hopkins University

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
