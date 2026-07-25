
import torch
import torch.nn as nn


class DigitalTwinSimulator:
    """
    Digital Twin Engine for generative/reconstruction-based representations of lesions.

    Responsibility:
        Simulate a 'digital twin' of a skin lesion using a generative model (e.g., VAE or Autoencoder)
        to reconstruct the input or generate variations.

    Time Complexity:
        O(M) where M is the time complexity of the underlying generative model's forward pass.

    Determinism:
        Fully deterministic if `seed` is provided, otherwise dependent on global RNG state.

    Mathematical Formula:
        Given an input x, the simulator outputs:
        x' = G(E(x)) or G(z) where z ~ N(mu, sigma)
        where G is the generator/decoder and E is the encoder.

    Edge Cases:
        - If input tensor is empty, raises ValueError.
        - Handles optional random seed for reproducibility in sampling processes.
    """

    def __init__(
        self,
        generative_model: nn.Module,
        seed: int | None = None
    ) -> None:
        """
        Initializes the DigitalTwinSimulator.

        Args:
            generative_model (nn.Module): The generative model to use (must have a forward pass that returns reconstructed/simulated output).
            seed (Optional[int]): Random seed for reproducibility.
        """
        super().__init__()
        self.generative_model = generative_model
        self.seed = seed

        if self.seed is not None:
            torch.manual_seed(self.seed)

    def simulate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Simulate the digital twin for a given input.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            **kwargs: Additional arguments to pass to the generative model.

        Returns:
            torch.Tensor: The simulated digital twin tensor of shape (B, C, H, W).

        Raises:
            ValueError: If input tensor is empty.
        """
        if x.numel() == 0:
            raise ValueError("Input tensor x cannot be empty.")

        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.generative_model.eval()
        with torch.no_grad():
            output = self.generative_model(x, **kwargs)

            # Assuming the generative model might return a tuple (like VAE returns recon_x, mu, logvar)
            if isinstance(output, tuple):
                return output[0]
            return output
