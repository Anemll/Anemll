from abc import ABC, abstractmethod

class BaseConverter(ABC):
    """Abstract base class for Apple Neural Engine model converters."""

    def __init__(self, model):
        self.model = model

    def preprocess(self):
        """Common preprocessing steps (e.g., tokenization, normalization)."""
        pass

    @abstractmethod
    def convert(self):
        """Each model must implement its own conversion logic."""
        pass

    def postprocess(self, num_workers=None):
        """Common postprocessing steps (e.g., optimizations).
        
        Args:
            num_workers: Optional number of workers for parallel processing.
                        If None, uses default single worker.
        """
        pass