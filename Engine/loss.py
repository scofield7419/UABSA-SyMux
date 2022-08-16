from abc import ABC
import torch


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class SyMuxLoss(Loss):
    def __init__(self, pol_criterion, term_criterion, model, optimizer, scheduler, max_grad_norm):
        self._pol_criterion = pol_criterion
        self._term_criterion = term_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm

    def compute(self, term_logits, pol_logits, term_types, pol_types, term_sample_masks, pol_sample_masks):
        # term loss
        term_logits = term_logits.view(-1, term_logits.shape[-1])
        term_types = term_types.view(-1)
        term_sample_masks = term_sample_masks.view(-1).float()

        term_loss = self._term_criterion(term_logits, term_types)
        term_loss = (term_loss * term_sample_masks).sum() / term_sample_masks.sum()

        # polarity loss
        pol_sample_masks = pol_sample_masks.view(-1).float()
        pol_count = pol_sample_masks.sum()

        if pol_count.item() != 0:
            pol_logits = pol_logits.view(-1, pol_logits.shape[-1])
            pol_types = pol_types.view(-1, pol_types.shape[-1])

            pol_loss = self._pol_criterion(pol_logits, pol_types)
            pol_loss = pol_loss.sum(-1) / pol_loss.shape[-1]
            pol_loss = (pol_loss * pol_sample_masks).sum() / pol_count

            # joint loss
            train_loss = term_loss + pol_loss
        else:
            # corner case: no positive/negative polation samples
            train_loss = term_loss

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()
