class Dataset(ImgDataset, datasets.CIFAR10):
    """CIFAR10 wrapper. Docs: `datasets.CIFAR10.`

    Parameters
    ----------
    kwargs:
        Additional arguments to `datasets.CIFAR10` and `ImgDataset`.

    Examples
    --------
    See SVHN for more examples.

    >>> data = CIFAR10(split="train") #doctest:+ELLIPSIS
    Files ...
    >>> from .helpers import get_mean_std
    >>> mean, std = get_mean_std(data)
    >>> list(std)
    [0.24703279, 0.24348423, 0.26158753]
    >>> (str(list(mean)) == str(data.mean)) and (str(list(std)) == str(data.std))
    True
    """

    shape = (3, 32, 32)
    n_classes = 10
    missing_px_color = COLOUR_BLACK
    n_train = 50000
    mean = [0.4914009, 0.48215896, 0.4465308]
    std = [0.24703279, 0.24348423, 0.26158753]

    def __init__(self, **kwargs):

        ImgDataset.__init__(self, **kwargs)
        datasets.CIFAR10.__init__(
            self,
            self.root,
            download=True,
            train=self.split == "train",
            transform=self.transform,
            target_transform=self.target_transform,
        )
        self.targets = to_numpy(self.targets)

        if self.is_random_targets:
            self.randomize_targets_()