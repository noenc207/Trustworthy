
import torch
import torch.nn as nn
import torch.optim as optim


class CounterfactualRecourseGenerator:
    """
    Counterfactual Generator (gradient-descent based latent space perturbation to flip class).

    Responsibility:
        Generate a counterfactual explanation by finding a minimal perturbation in the latent space
        such that the generated image flips its predicted class to a target class.

    Time Complexity:
        O(I * (G + C + B)) where I is the number of iterations, G is the generator forward pass,
        C is the classifier forward pass, and B is the backward pass complexity.

    Determinism:
        Fully deterministic if `seed` is provided, setting the random seed for optimization.

    Mathematical Formula:
        z* = argmin_z [ Loss(classifier(generator(z)), target) + lambda * ||z - z_orig||_2^2 ]
        To prevent vanishing gradients, small epsilon is added to the distance.

    Edge Cases:
        - target_class is the same as the original class (should still work, but distance penalty dominates).
        - Vanishing gradients in distance calculation: uses epsilon (1e-8) for numerical stability.
        - Unreachable target class within max_iter: returns the best z found.
        - Empty latent tensor: raises ValueError.
    """

    def __init__(
        self,
        classifier: nn.Module,
        generator: nn.Module,
        lambda_reg: float = 0.1,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        epsilon: float = 1e-8,
        seed: int | None = None,
    ) -> None:
        """
        Initializes the CounterfactualRecourseGenerator.

        Args:
            classifier (nn.Module): The classification model f(x).
            generator (nn.Module): The generative model decoder g(z).
            lambda_reg (float): Regularization weight for the distance penalty.
            learning_rate (float): Learning rate for latent space optimization.
            max_iter (int): Maximum number of gradient descent iterations.
            epsilon (float): Small value to prevent vanishing gradients/division by zero in distance.
            seed (Optional[int]): Random seed for reproducibility.
        """
        self.classifier = classifier
        self.generator = generator
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.seed = seed

    def generate(
        self, z_orig: torch.Tensor, target_class: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate counterfactual representation in the latent space.

        Args:
            z_orig (torch.Tensor): Original latent space representation of shape (B, D).
            target_class (torch.Tensor): Target class labels of shape (B,).

        Returns:
            torch.Tensor: Optimized counterfactual latent representation of shape (B, D).

        Raises:
            ValueError: If z_orig is empty.
        """
        if z_orig.numel() == 0:
            raise ValueError("Input latent tensor z_orig cannot be empty.")

        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.classifier.eval()
        self.generator.eval()

        # Clone and detach to create the starting point for optimization, require gradients
        z_opt = z_orig.clone().detach().requires_grad_(True)

        optimizer = optim.Adam([z_opt], lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.max_iter):
            optimizer.zero_grad()

            # Generate image from latent
            x_gen = self.generator(z_opt)

            # Predict class
            logits = self.classifier(x_gen)

            # Classification loss
            loss_cls = criterion(logits, target_class)

            # Distance loss with epsilon to prevent vanishing gradients
            # L2 distance squared
            diff = z_opt - z_orig
            dist_sq = torch.sum(diff * diff, dim=-1) + self.epsilon
            loss_dist = torch.mean(dist_sq)

            # Total loss
            loss = loss_cls + self.lambda_reg * loss_dist

            loss.backward()
            optimizer.step()

        return z_opt.detach()
