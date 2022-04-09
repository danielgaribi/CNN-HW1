import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops).
        #   Hint: Create a matrix M where M[i,j] is the margin-loss
        #   for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        N = y.size(dim=0)
        relu = torch.nn.ReLU()
        x_w_yi = x_scores.gather(1, y.reshape(-1, 1))
        M = x_scores - x_w_yi + self.delta
        L_with_yi = relu(M)
        L_sum = torch.sum(L_with_yi) - N * self.delta # remove N deltas, one for each y_i in the sum
        loss = (L_sum / N)
        
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx["x"] = x
        self.grad_ctx["y"] = y
        self.grad_ctx["M"] = M
        # ========================

        return loss

    def grad(self):

        # TODO: Implement SVM loss gradient calculation
        # Same notes as above. Hint: Use the matrix M from above, based on
        # it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        x = self.grad_ctx["x"]
        y = self.grad_ctx["y"]
        M = self.grad_ctx["M"]
        N = y.size(dim=0)
        
        M_is_grater = M > 0 # 1 if Mij>0 else 0
        M_is_grater = M_is_grater.float()
        M_is_grater_y_indexes = M_is_grater.gather(1, y.reshape(-1, 1))
        M_nof_ones_in_rows = torch.sum(M_is_grater, axis=1).reshape(-1,1)
        dL_j_is_y = M_nof_ones_in_rows - M_is_grater_y_indexes
        G = M_is_grater.scatter(1, y.reshape(-1, 1), (-1)*dL_j_is_y)
        grad = x.T @ G
        grad = grad * (1/N)
        
        # ========================

        return grad
