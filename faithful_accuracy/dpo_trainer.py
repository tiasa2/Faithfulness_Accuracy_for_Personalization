"""DPO trainer used for curriculum training.

Adapted from RLHFlow Online-DPO-R1 (MIT), itself built on TRL's DPOTrainer: adds per-pair margins and
logs the correctness of chosen/rejected responses.
"""
import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from huggingface_hub.utils._deprecation import _deprecate_arguments
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from trl.trainer.utils import cap_exp
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl import DPOTrainer
from trl.trainer.dpo_config import DPOConfig, FDivergenceConstants, FDivergenceType
from trl.trainer.dpo_trainer import PreferenceCollator


class CustomPreferenceCollator(PreferenceCollator):
    EXTRA_KEYS = ["margin", "chosen_score", "rejected_score", "chosen_pos", "chosen_neg"]

    def torch_call(self, examples):
        output = super().torch_call(examples)
        for key in self.EXTRA_KEYS:
            if key in examples[0]:
                output[key] = [example[key] for example in examples]
        return output

class MyDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Optional[str] = None,
        args: Optional[DPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
        force_use_ref_model: bool = False,
    ):
        if data_collator is None:
            if padding_value is not None:
                pad_id = padding_value
            elif hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                pad_id = tokenizer.pad_token_id
            elif hasattr(tokenizer, 'tokenizer') and tokenizer.tokenizer.pad_token_id is not None:
                pad_id = tokenizer.tokenizer.pad_token_id
            else:
                pad_id = 0  # fallback
            data_collator = CustomPreferenceCollator(pad_token_id=pad_id)

        super().__init__(
            model=model,
            ref_model=ref_model,
            beta=beta,
            label_smoothing=label_smoothing,
            loss_type=loss_type,
            args=args,
            data_collator=data_collator,
            label_pad_token_id=label_pad_token_id,
            padding_value=padding_value,
            truncation_mode=truncation_mode,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            max_target_length=max_target_length,
            peft_config=peft_config,
            is_encoder_decoder=is_encoder_decoder,
            disable_dropout=disable_dropout,
            generate_during_eval=generate_during_eval,
            compute_metrics=compute_metrics,
            precompute_ref_log_probs=precompute_ref_log_probs,
            dataset_num_proc=dataset_num_proc,
            model_init_kwargs=model_init_kwargs,
            ref_model_init_kwargs=ref_model_init_kwargs,
            model_adapter_name=model_adapter_name,
            ref_adapter_name=ref_adapter_name,
            reference_free=reference_free,
            force_use_ref_model=force_use_ref_model,
        )
        self.use_dpo_data_collator = True
        print("----------------")
        print("Train dataset columns:", self.train_dataset.column_names)
        print(len(train_dataset))
        print("----------------")

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        self._signature_columns += [
            "margin",
            "chosen_score",
            "rejected_score",
            "chosen_pos",
            "chosen_neg",
        ]

    # keep only the extra pair fields through tokenization
    @staticmethod
    def tokenize_row(features, processing_class, max_prompt_length, max_completion_length, add_special_tokens):
        output = DPOTrainer.tokenize_row(
            features, processing_class, max_prompt_length, max_completion_length, add_special_tokens
        )
        for key in ["margin", "chosen_score", "rejected_score", "chosen_pos", "chosen_neg"]:
            if key in features:
                output[key] = features[key]
        return output

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        train_eval = "train" if "loss" in logs else "eval"
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        if start_time is not None:
            return Trainer.log(self, logs, start_time)
        return Trainer.log(self, logs)

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        chosen_logratios = policy_chosen_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_chosen_logps.to(self.accelerator.device)
        rejected_logratios = policy_rejected_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_rejected_logps.to(self.accelerator.device)

        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
            logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        else:
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            if self.reference_free:
                ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
            else:
                ref_logratios = reference_chosen_logps - reference_rejected_logps

            pi_logratios = pi_logratios.to(self.accelerator.device)
            ref_logratios = ref_logratios.to(self.accelerator.device)
            logits = pi_logratios - ref_logratios

            if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE.value:
                logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "robust":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                + F.logsigmoid(-self.beta * logits) * self.label_smoothing
            ) / (1 - 2 * self.label_smoothing)
        elif self.loss_type == "exo_pair":
            import math

            if self.label_smoothing == 0:
                self.label_smoothing = 1e-3
            losses = (self.beta * logits).sigmoid() * (
                F.logsigmoid(self.beta * logits) - math.log(1 - self.label_smoothing)
            ) + (-self.beta * logits).sigmoid() * (F.logsigmoid(-self.beta * logits) - math.log(self.label_smoothing))
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "bco_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean

            losses = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
                -(self.beta * rejected_logratios - delta)
            )
        elif self.loss_type == "sppo_hard":
            # In the paper (https://huggingface.co/papers/2405.00675), SPPO employs a soft probability approach, estimated using the PairRM score. The probability calculation is conducted outside of the trainer class. The version described here is the hard probability version, where P in Equation (4.7) of Algorithm 1 is set to 1 for the winner and 0 for the loser.
            a = policy_chosen_logps - reference_chosen_logps
            b = policy_rejected_logps - reference_rejected_logps

            losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2
        elif self.loss_type == "nca_pair":
            chosen_rewards = (policy_chosen_logps - reference_chosen_logps) * self.beta
            rejected_rewards = (policy_rejected_logps - reference_rejected_logps) * self.beta
            losses = (
                -F.logsigmoid(chosen_rewards)
                - 0.5 * F.logsigmoid(-chosen_rewards)
                - 0.5 * F.logsigmoid(-rejected_rewards)
            )
        elif self.loss_type == "aot_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
            rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)

            delta = chosen_logratios_sorted - rejected_logratios_sorted

            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "aot":
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps

            pi_logratios_sorted, _ = torch.sort(pi_logratios, dim=0)
            ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)

            delta = pi_logratios_sorted - ref_logratios_sorted

            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "apo_zero":
            # Use this loss when you believe the chosen outputs are better than your model's default output

            losses_chosen = 1 - F.sigmoid(self.beta * chosen_logratios)  # Increase chosen likelihood
            losses_rejected = F.sigmoid(self.beta * rejected_logratios)  # Decrease rejected likelihood

            losses = losses_chosen + losses_rejected

        elif self.loss_type == "apo_down":
            # Use this loss when you believe the chosen outputs are worse than your model's default output

            losses_chosen = F.sigmoid(self.beta * chosen_logratios)  # Decrease chosen likelihood
            losses_rejected = 1 - F.sigmoid(
                self.beta * (chosen_logratios - rejected_logratios)
            )  # Decrease rejected likelihood more

            losses = losses_chosen + losses_rejected

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'exo_pair', 'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_pair', 'apo_zero', 'apo_down']"
            )

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)


    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]):
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        if self.is_vision_model:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
            if "pixel_attention_mask" in concatenated_batch:
                model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        # Concatenate prompt + completion
        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        # loss_mask: only compute loss on completion tokens
        loss_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
            dim=1,
        )

        # Flush left to reduce memory usage
        for i in range(attention_mask.size(0)):
            first_one_idx = torch.nonzero(attention_mask[i])[0].item()
            input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
            attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
            loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)

        # Remove trailing all-zero columns
        empty_cols = torch.sum(attention_mask, dim=0) == 0
        first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else attention_mask.size(1) + 1
        input_ids = input_ids[:, :first_empty_col - 1]
        attention_mask = attention_mask[:, :first_empty_col - 1]
        loss_mask = loss_mask[:, :first_empty_col - 1]

        # Truncate right
        if self.args.max_length is not None:
            input_ids = input_ids[:, :self.args.max_length]
            attention_mask = attention_mask[:, :self.args.max_length]
            loss_mask = loss_mask[:, :self.args.max_length]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)

        # Offset logits by one to align with labels
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:].clone()
        loss_mask = loss_mask[:, 1:].bool()

        if logits.shape[:2] != labels.shape[:2]:
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute log probs
        labels[~loss_mask] = 0  # dummy; ignored via loss_mask
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        per_token_logps[~loss_mask] = 0

        all_logps = per_token_logps.sum(-1)

        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)

        output = {}
        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["mean_chosen_logits"] = logits[:num_examples][loss_mask[:num_examples]].mean()
        output["mean_rejected_logits"] = logits[num_examples:][loss_mask[num_examples:]].mean()

        chosen_logits = logits[:num_examples]
        chosen_labels = labels[:num_examples]
        output["nll_loss"] = F.cross_entropy(
            chosen_logits.reshape(-1, chosen_logits.shape[-1]),
            chosen_labels.reshape(-1),
            ignore_index=0,
        )

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output


    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        metrics = {}

        model_output = self.concatenated_forward(model, batch)

        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        ref_output = self.concatenated_forward(self.model, batch)
                else:
                    ref_output = self.concatenated_forward(self.ref_model, batch)
                ref_chosen_logps = ref_output["chosen_logps"]
                ref_rejected_logps = ref_output["rejected_logps"]

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_chosen_logps,
            ref_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output["nll_loss"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""

        # correctness of chosen / rejected responses (logged for monitoring)
        if "chosen_pos" in batch:
            c_pos = torch.tensor(batch["chosen_pos"], dtype=torch.float32, device=self.accelerator.device)
            metrics[f"{prefix}correctness/chosen_avg"] = c_pos.mean().item()
        if "chosen_neg" in batch:
            c_neg = torch.tensor(batch["chosen_neg"], dtype=torch.float32, device=self.accelerator.device)
            metrics[f"{prefix}correctness/rejected_avg"] = c_neg.mean().item()
        if "chosen_pos" in batch and "chosen_neg" in batch:
            metrics[f"{prefix}correctness/batch_gap"] = (c_pos - c_neg).mean().item()
        if "chosen_score" in batch and "rejected_score" in batch:
            c_scores_orig = torch.tensor(batch["chosen_score"], dtype=torch.float32, device=self.accelerator.device)
            r_scores_orig = torch.tensor(batch["rejected_score"], dtype=torch.float32, device=self.accelerator.device)
            metrics[f"{prefix}reward_model/chosen_avg"] = c_scores_orig.mean().item()
            metrics[f"{prefix}reward_model/rejected_avg"] = r_scores_orig.mean().item()
            model_prefers_chosen = chosen_rewards > rejected_rewards
            preferred_scores = torch.where(model_prefers_chosen, c_scores_orig, r_scores_orig)
            metrics[f"{prefix}reward_model/preferred_avg"] = preferred_scores.mean().item()

        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/chosen"] = model_output["chosen_logps"].detach().mean().cpu()
        metrics[f"{prefix}logps/rejected"] = model_output["rejected_logps"].detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = model_output["mean_chosen_logits"].detach().cpu()
        metrics[f"{prefix}logits/rejected"] = model_output["mean_rejected_logits"].detach().cpu()
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = model_output["nll_loss"].detach().mean().cpu()
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = model_output["aux_loss"].detach().cpu()

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        compute_loss_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()

        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss


    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()

        with torch.no_grad(), prediction_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)