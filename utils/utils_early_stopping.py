"""
# --------------------------------------------
# Early Stopping
# --------------------------------------------
# Jiahao Huang (j.huang21@imperial.uk.ac)
# 30/Jan/2022
# --------------------------------------------
"""


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, delta = 0):

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.is_save = 0

    def __call__(self, psnr, model, epoch, step):

        # psnr(0 --> +) score(0 --> +)
        score = psnr
        # init score
        if self.best_score is None:
            self.best_score = score
            self.is_save = 0
        # new model is worse
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.is_save = 0
            if self.counter >= self.patience:
                self.early_stop = True
        # new model is better
        else:
            self.best_score = score
            self.is_save = 1
            self.counter = 0

        print(f'EarlyStopping counter of epoch {epoch} step {step} : {self.counter} out of {self.patience}')

        return self.is_save
