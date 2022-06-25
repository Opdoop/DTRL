"""
Attack Recipe Class
========================

"""

from abc import ABC, abstractmethod

from textattack.shared import Attack


class AttackRecipe(Attack, ABC):
    """A recipe for building an NLP adversarial attacks from the literature."""

    @staticmethod
    @abstractmethod
    def build(model):
        """Creates an attacks recipe from recipe-specific arguments.

        Allows for support of different configurations of a single
        attacks.
        """
        raise NotImplementedError()
