import os

SKIP = 48 # how many experiments to skip - if you have run 5 set SKIP to 5 and it will skip the furst 5 experiments

# SGHMC - 24 experiments
for alpha in [0.1, 0.01, 0.001]:
    for eta in [1e-6, 2e-6, 4e-6, 8e-6]:
        for resample_n in [0, 100]:
            if SKIP:
                SKIP -= 1
                continue
            os.system('python bnn.py --updater {} --n-warmup {} --n-epochs {} --batch-size {} --hidden-size {} --lr {} --alpha {} --resample-n {}'.format(
                'SGHMC',
                50,
                800,
                500,
                100,
                eta,
                alpha,
                resample_n
            ))
#SGLD - 6 experiments
for eta in [1e-5, 2e-5, 4e-5]:
    for lr_decay in [0, 1]:
        if SKIP:
            SKIP -= 1
            continue
        os.system('python bnn.py --updater {} --n-warmup {} --n-epochs {} --batch-size {} --hidden-size {} --lr {} --lr-decay {}'.format(
            'SGLD',
            50,
            800,
            500,
            100,
            eta,
            lr_decay
        ))

#SGD - 12 experiments
for eta in [1e-5, 2e-5, 4e-5, 6e-5]:
    for reg in [0.1, 1.0, 10.0]:
        if SKIP:
            SKIP -= 1
            continue
        os.system('python bnn.py --updater {} --n-warmup {} --n-epochs {} --batch-size {} --hidden-size {} --lr {} --reg {}'.format(
            'SGD',
            50,
            800,
            500,
            100,
            eta,
            reg
        ))

#SGDMOM - 48 experiments
for wd in [0.0, 1e-6]:
    if wd == 0.0:
        for alpha in [0.1, 0.01, 0.001]:
            for eta in [1e-6, 2e-6, 4e-6, 8e-6]:   
                for reg in [0.1, 1.0, 10.0]:
                    if SKIP:
                        SKIP -= 1
                        continue
                    os.system('python bnn.py --updater {} --n-warmup {} --n-epochs {} --batch-size {} --hidden-size {} --lr {} --alpha {} --wd {} --reg {}'.format(
                        'SGDMOM',
                        50,
                        800,
                        500,
                        100,
                        eta,
                        alpha,
                        wd,
                        reg
                    ))
    if wd == 1e-6:
        for alpha in [0.1, 0.01, 0.001]:
            for eta in [1e-4, 2e-4, 4e-4, 8e-4]:   
                for reg in [0.1, 1.0, 10.0]:
                    if SKIP:
                        SKIP -= 1
                        continue
                    os.system('python bnn.py --updater {} --n-warmup {} --n-epochs {} --batch-size {} --hidden-size {} --lr {} --alpha {} --wd {} --reg {}'.format(
                        'SGDMOM',
                        50,
                        800,
                        500,
                        100,
                        eta,
                        alpha,
                        wd,
                        reg
                    ))