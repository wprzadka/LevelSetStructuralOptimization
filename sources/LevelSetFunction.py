import abc
import numpy as np


class LevelSetFunction(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, x: np.ndarray, *args, **kwargs) -> float:
        """
        :param x: coordinate in which function value has to be computed
        :return: value of the level set function in x
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, velocity_values: np.ndarray, dt: float) -> None:
        """
        :param velocity_values: values of velocity field in the centers of mesh elements
        :param dt: time delta
        """
        raise NotImplementedError
