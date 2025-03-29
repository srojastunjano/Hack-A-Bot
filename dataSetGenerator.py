from typing import Generator

n_iter=10
train_dir = 'processed_Train'

# Create representative dataset generator
def get_representative_dataset() -> Generator:
    """A function that loads the dataset and returns a representative dataset generator.

    Returns:
        Generator: A generator yielding batches of preprocessed images.
    """
    dataset = train_dir

    def representative_dataset() -> Generator:
        """A generator function that yields batch of preprocessed images.

        Yields:
            A batch of preprocessed images.
        """
        for _ in range(n_iter):
            yield dataset.take(1).get_single_element()[0].numpy()

    return representative_dataset

# Create a representative dataset generator
representative_dataset_gen = get_representative_dataset()
