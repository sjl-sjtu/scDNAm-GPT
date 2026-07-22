from src.model.scdnam_mixer_seq_simple import MambaLMHeadModel
from transformers.modeling_outputs import CausalLMOutput
from typing import Optional
import sys, torch
from torch import nn
import numpy as np
from src.mambaconfig import MambaConfig
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

sys.path.append("..")

import torch

class scDNAmGPTLMHeadModelwithLoss(MambaLMHeadModel):
    """
    A Language Model Head with additional support for loss calculations related to methylation.

    Attributes:
        pad_token_id: Token ID for the padding token.
        bos_token_id: Token ID for the beginning-of-sequence token ([BOS]).
        sep_token_id: Token ID for the sequence-separator token ([SEP]).
    """
    def __init__(self, config, tokenizer, initializer_cfg=None, device=None, dtype=None, use_dataug=False):
        super(scDNAmGPTLMHeadModelwithLoss, self).__init__(config, initializer_cfg, device, dtype)

        self.pad_token_id = tokenizer._convert_token_to_id("[PAD]")
        self.bos_token_id = tokenizer._convert_token_to_id("[BOS]")
        self.sep_token_id = tokenizer._convert_token_to_id("[SEP]")
        self.mask_token_id = tokenizer._convert_token_to_id("[MASK]")
        self.tokenizer = tokenizer
        self.use_dataug = use_dataug
        self.profile_ce_mem = True
        self.last_profile = {}

        # Define replacement conditions
        self.condition_unmethy = lambda x: (x > 6) & (x % 2 == 1)  # ID > 6 and odd
        # self.condition_methy = lambda x: (x > 6) & (x % 2 == 0)  # ID > 6 and even
        
        print("------------------------------------------\nShape of lm_head.weight: ", self.lm_head.weight.shape, "\n------------------------------------------")

    @classmethod
    def from_pretrained(cls, tokenizer, pretrained_model_name, device=None, dtype=None, **kwargs):
        """Load a pretrained model and configuration."""
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, tokenizer, device=device, dtype=dtype, **kwargs)
        if torch.cuda.device_count() == 1:
            device = "cuda:0"
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error: {e}")
            del state_dict['classify.weight']
            del state_dict['classify.bias']
            model.load_state_dict(state_dict, strict=False)
        return model

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dictionary and handle unexpected or missing keys."""
        load_result = super().load_state_dict(state_dict, strict=strict)

        if strict:
            missing_keys = [key for key in self.state_dict().keys() if key not in state_dict]
            unexpected_keys = [key for key in state_dict.keys() if key not in self.state_dict().keys()]
            load_result = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}

        self.tie_weights()
        return load_result

class scDNAmGPTForSequenceClassification(scDNAmGPTLMHeadModelwithLoss):
    """
    A class for sequence classification based on methylation data.

    Attributes:
        device: The device where the model is deployed (CPU/GPU).
        num_labels: The number of classification labels.
        classify: Linear layer for classification.
        attention: Multihead attention mechanism.
        dropout: Dropout layer to prevent overfitting.
        norm: Layer normalization to stabilize training.
    """

    def __init__(
        self,
        config,
        tokenizer,
        num_labels: int,
        attention_mechanism: dict = None,
        initializer_cfg=None,
        device=None,
        dtype=None,
        need_layer_hidden_states: Optional[bool]=False,
        cross_attn_every_hidden_states: Optional[bool]=False,
    ) -> None:
        """Initialize the sequence classification model."""
        super().__init__(config, tokenizer, initializer_cfg, device, dtype)

        self.device = device
        self.num_labels = num_labels
        self._keys_to_ignore_on_save = ['lm_head.weight']
        self.need_layer_hidden_states = need_layer_hidden_states
        self.cross_attn_every_hidden_states = cross_attn_every_hidden_states

        # Configure attention dimensions
        if attention_mechanism is not None:
            self.projection_dim = attention_mechanism.get("projection_dim", None)
        else:
            print("There is no attention_mechanism config")

        self.attention_embed_dim = (
            self.projection_dim
            if isinstance(self.projection_dim, int) and self.projection_dim > 0
            else config.d_model
        )
        self.attention_num_heads = attention_mechanism["attention_num_heads"]
        self.dropout_rate = attention_mechanism.get("dropout_rate", 0.0)

        # Optional projection layers
        if self.projection_dim:
            if cross_attn_every_hidden_states:
                # Create a list of projection layers for each backbone layer
                self.query_proj_layers = nn.ModuleList(
                    [
                        nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
                        for _ in range(len(self.backbone.layers) + 1)  # One projection layer per backbone layer
                    ]
                )

                self.key_proj_layers = nn.ModuleList(
                    [
                        nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
                        for _ in range(len(self.backbone.layers) + 1)  # One projection layer per backbone layer
                    ]
                )

                self.value_proj_layers = nn.ModuleList(
                    [
                        nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
                        for _ in range(len(self.backbone.layers) + 1)  # One projection layer per backbone layer
                    ]
                )
            else:
                self.query_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
                self.key_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
                self.value_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)


        if cross_attn_every_hidden_states:
            # Create a list of attention layers with the same length as backbone layers
            self.attention_layers = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim=self.attention_embed_dim,
                        num_heads=self.attention_num_heads,
                        batch_first=True,
                    )
                    for _ in range(len(self.backbone.layers) + 1)  # Number of attention layers = number of backbone layers
                ]
            )
        else:
            # Multihead attention
            self.attention = nn.MultiheadAttention(
                embed_dim=self.attention_embed_dim,
                num_heads=self.attention_num_heads,
                batch_first=True,
            )

        # Regularization and classification layers
        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm = nn.LayerNorm(self.attention_embed_dim)
    
        self.classify = nn.Linear(self.attention_embed_dim, num_labels, dtype=dtype).to(device)
        nn.init.xavier_uniform_(self.classify.weight)
        nn.init.zeros_(self.classify.bias)

    @classmethod
    def from_pretrained(cls, tokenizer, pretrained_model_name, device=None, dtype=None, strict: bool = False, **kwargs):
        """
        Load a pretrained model with optional strictness.
        
        Args:
            tokenizer: The tokenizer to use.
            pretrained_model_name: Path or HuggingFace model name.
            device: Target device (cuda/cpu).
            dtype: Data type.
            strict: Whether to enforce exact parameter matching. Default is False.
            **kwargs: Other model init args.
        """
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, tokenizer, device=device, dtype=dtype, **kwargs)

        if torch.cuda.device_count() == 1:
            device = "cuda:0"
        if not torch.cuda.is_available():
            device = torch.device("cpu")

        # Load raw state dict
        state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)

        if not strict:
            # Filter out mismatched or unknown keys
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        print(f"[Skip] Shape mismatch for key: {k}, checkpoint: {v.shape}, model: {model_state_dict[k].shape}")
                else:
                    print(f"[Skip] Unexpected key not in model: {k}")
            state_dict = filtered_state_dict

        # Load weights
        load_result = model.load_state_dict(state_dict, strict=strict)

        # Print warnings if not strict
        if not strict:
            if load_result.missing_keys:
                print("[Missing keys]:")
                for k in load_result.missing_keys:
                    print(f"  - {k}")
            if load_result.unexpected_keys:
                print("[Unexpected keys]:")
                for k in load_result.unexpected_keys:
                    print(f"  - {k}")

        return model

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Retrieve the state dictionary, excluding unnecessary keys."""
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        state.pop(prefix + 'lm_head.weight', None)
        return state

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dictionary and handle unexpected or missing keys."""
        load_result = super().load_state_dict(state_dict, strict=strict)

        if strict:
            missing_keys = [key for key in self.state_dict().keys() if key not in state_dict]
            unexpected_keys = [key for key in state_dict.keys() if key not in self.state_dict().keys()]
            load_result = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}

        self.tie_weights()
        return load_result


    def _validate_and_get_sep_indices(
        self,
        unmethy_input_ids,
        methy_input_ids,
        attention_mask,
    ):
        """
        Validate sequence boundaries and return the position of [SEP].

        Every sequence must start with [BOS], end with [SEP] at its last
        non-padding position, and use an attention mask with the same shape as
        the two input-ID tensors.

        Parameters
        ----------
        unmethy_input_ids : torch.Tensor
            Unmethylated token IDs with shape [batch_size, sequence_length].
        methy_input_ids : torch.Tensor
            Methylated token IDs with shape [batch_size, sequence_length].
        attention_mask : torch.Tensor
            Padding mask with 1/True for valid tokens and 0/False for padding.

        Returns
        -------
        sep_indices : torch.LongTensor
            Position of the terminal [SEP] token for every sequence.
        attention_mask : torch.BoolTensor
            Boolean attention mask on the same device as the input tensors.
        """
        if unmethy_input_ids.ndim != 2 or methy_input_ids.ndim != 2:
            raise ValueError(
                "unmethy_input_ids and methy_input_ids must have shape "
                "[batch_size, sequence_length]."
            )

        if unmethy_input_ids.shape != methy_input_ids.shape:
            raise ValueError(
                "unmethy_input_ids and methy_input_ids must have the same shape."
            )

        if attention_mask is None:
            raise ValueError("attention_mask must not be None during validation.")

        if attention_mask.shape != methy_input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as the input-ID tensors."
            )

        attention_mask = attention_mask.to(
            device=methy_input_ids.device,
            dtype=torch.bool,
        )

        sequence_lengths = attention_mask.long().sum(dim=1)
        invalid_length = sequence_lengths < 2
        if invalid_length.any():
            invalid_indices = torch.nonzero(
                invalid_length,
                as_tuple=False,
            ).flatten().tolist()
            raise ValueError(
                "Each sequence must contain at least [BOS] and [SEP]. "
                f"Invalid batch indices: {invalid_indices}"
            )

        # The dataset/collator uses right padding, so the final valid token is
        # located at sequence_length - 1 for each sequence.
        sep_indices = sequence_lengths - 1
        batch_indices = torch.arange(
            methy_input_ids.size(0),
            device=methy_input_ids.device,
        )

        invalid_first_mask = ~attention_mask[:, 0]
        if invalid_first_mask.any():
            invalid_indices = torch.nonzero(
                invalid_first_mask,
                as_tuple=False,
            ).flatten().tolist()
            raise ValueError(
                "The first position of every sequence must be a valid [BOS] "
                f"token. Invalid batch indices: {invalid_indices}"
            )

        invalid_bos = (
            (unmethy_input_ids[:, 0] != self.bos_token_id)
            | (methy_input_ids[:, 0] != self.bos_token_id)
        )
        if invalid_bos.any():
            invalid_indices = torch.nonzero(
                invalid_bos,
                as_tuple=False,
            ).flatten().tolist()
            raise ValueError(
                "Both unmethylated and methylated input sequences must start "
                f"with [BOS]. Invalid batch indices: {invalid_indices}"
            )

        invalid_sep_mask = ~attention_mask[batch_indices, sep_indices]
        if invalid_sep_mask.any():
            invalid_indices = torch.nonzero(
                invalid_sep_mask,
                as_tuple=False,
            ).flatten().tolist()
            raise ValueError(
                "The calculated terminal position is masked. Ensure that "
                "attention_mask uses right padding. "
                f"Invalid batch indices: {invalid_indices}"
            )

        invalid_sep = (
            unmethy_input_ids[batch_indices, sep_indices] != self.sep_token_id
        ) | (
            methy_input_ids[batch_indices, sep_indices] != self.sep_token_id
        )
        if invalid_sep.any():
            invalid_indices = torch.nonzero(
                invalid_sep,
                as_tuple=False,
            ).flatten().tolist()
            raise ValueError(
                "The last non-padding token of both unmethylated and "
                "methylated input sequences must be [SEP]. "
                f"Invalid batch indices: {invalid_indices}"
            )

        return sep_indices, attention_mask
    

    def forward(
        self,
        unmethy_input_ids,
        methy_input_ids,
        methy_ratios,
        labels=None,
        attention_mask=None,
        inference_params=None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Perform a forward pass of the sequence-classification model."""
        if attention_mask is None:
            attention_mask = methy_input_ids != self.pad_token_id

        # Explicitly verify that each sequence starts with [BOS] and that its
        # last non-padding position contains [SEP]. The validated [SEP] state
        # is used as the cross-attention query below.
        sep_indices, attention_mask = self._validate_and_get_sep_indices(
            unmethy_input_ids=unmethy_input_ids,
            methy_input_ids=methy_input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = self.backbone(
            unmethy_input_ids,
            methy_input_ids,
            methy_ratios,
            inference_params=inference_params,
            need_layer_hidden_states=(
                self.need_layer_hidden_states
                or self.cross_attn_every_hidden_states
            ),
            **kwargs,
        )

        if self.need_layer_hidden_states or self.cross_attn_every_hidden_states:
            hidden_states, layer_hidden_states = hidden_states

        if self.cross_attn_every_hidden_states:
            batch_size = layer_hidden_states[0].size(0)
            batch_indices = torch.arange(
                batch_size,
                device=layer_hidden_states[0].device,
            )
            key_padding_mask = ~attention_mask

            norm_attn_outputs = []
            attn_weights = []

            # Preserve the original selection of backbone hidden states.
            layer_hidden_states = (
                layer_hidden_states[:-2] + layer_hidden_states[-1:]
            )

            for i, current_hidden_states in enumerate(layer_hidden_states):
                # Use the validated terminal [SEP] hidden state as the query.
                sep_token_states = current_hidden_states[
                    batch_indices,
                    sep_indices,
                ]

                if hasattr(self, "query_proj_layers"):
                    q_proj = self.query_proj_layers[i](
                        sep_token_states.unsqueeze(1)
                    )
                    k_proj = self.key_proj_layers[i](current_hidden_states)
                    v_proj = self.value_proj_layers[i](current_hidden_states)
                else:
                    q_proj = sep_token_states.unsqueeze(1)
                    k_proj = current_hidden_states
                    v_proj = current_hidden_states

                attn_output, attn_weight = self.attention_layers[i](
                    q_proj,
                    k_proj,
                    v_proj,
                    key_padding_mask=key_padding_mask,
                )

                norm_attn_outputs.append(self.norm(attn_output))
                attn_weights.append(attn_weight)

            if return_dict:
                norm_attn_outputs_for_save = torch.cat(
                    norm_attn_outputs,
                    dim=1,
                )

            norm_attn_outputs = torch.sum(
                torch.cat(norm_attn_outputs, dim=1),
                dim=1,
            )
            attn_weights = torch.cat(attn_weights, dim=1)

        else:
            batch_size = hidden_states.size(0)
            batch_indices = torch.arange(
                batch_size,
                device=hidden_states.device,
            )

            # Use the validated terminal [SEP] hidden state as the query.
            sep_token_states = hidden_states[
                batch_indices,
                sep_indices,
            ]
            key_padding_mask = ~attention_mask

            if hasattr(self, "query_proj"):
                q_proj = self.query_proj(sep_token_states.unsqueeze(1))
                k_proj = self.key_proj(hidden_states)
                v_proj = self.value_proj(hidden_states)
            else:
                q_proj = sep_token_states.unsqueeze(1)
                k_proj = hidden_states
                v_proj = hidden_states

            attn_output, attn_weights = self.attention(
                q_proj,
                k_proj,
                v_proj,
                key_padding_mask=key_padding_mask,
            )
            attn_output = attn_output.squeeze(1)
            norm_attn_outputs = self.norm(attn_output)

            if return_dict:
                norm_attn_outputs_for_save = norm_attn_outputs

        logits = self.classify(norm_attn_outputs)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        else:
            loss = None

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return {
            "logits": logits,
            "labels": labels,
            "loss": loss,
        }

class scDNAmGPTForPredictscRNAseq(scDNAmGPTForSequenceClassification):

    def __init__(
        self,
        config,
        tokenizer,
        num_labels: int,
        attention_mechanism: dict = None,
        initializer_cfg=None,
        device=None,
        dtype=None,
        need_layer_hidden_states: bool = False,
        cross_attn_every_hidden_states: bool = False,
        **kwargs,
    ):
        super().__init__(
            config,
            tokenizer,
            num_labels,
            attention_mechanism=attention_mechanism,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
            need_layer_hidden_states=need_layer_hidden_states,
            cross_attn_every_hidden_states=cross_attn_every_hidden_states,
            **kwargs,
        )

    def forward(
        self,
        unmethy_input_ids,
        methy_input_ids,
        methy_ratios,
        labels=None,
        attention_mask=None,
        inference_params=None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Perform a forward pass of the gene-expression prediction model."""
        if attention_mask is None:
            attention_mask = methy_input_ids != self.pad_token_id

        # This subclass defines its own forward method, so boundary validation
        # must also be invoked explicitly here.
        sep_indices, attention_mask = self._validate_and_get_sep_indices(
            unmethy_input_ids=unmethy_input_ids,
            methy_input_ids=methy_input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = self.backbone(
            unmethy_input_ids,
            methy_input_ids,
            methy_ratios,
            inference_params=inference_params,
            need_layer_hidden_states=(
                self.need_layer_hidden_states
                or self.cross_attn_every_hidden_states
            ),
            **kwargs,
        )

        if self.need_layer_hidden_states or self.cross_attn_every_hidden_states:
            hidden_states, layer_hidden_states = hidden_states

        if self.cross_attn_every_hidden_states:
            batch_size = layer_hidden_states[0].size(0)
            batch_indices = torch.arange(
                batch_size,
                device=layer_hidden_states[0].device,
            )
            key_padding_mask = ~attention_mask

            norm_attn_outputs = []
            attn_weights = []

            # Preserve the original selection of backbone hidden states.
            layer_hidden_states = (
                layer_hidden_states[:-2] + layer_hidden_states[-1:]
            )

            for i, current_hidden_states in enumerate(layer_hidden_states):
                # Use the validated terminal [SEP] hidden state as the query.
                sep_token_states = current_hidden_states[
                    batch_indices,
                    sep_indices,
                ]

                if hasattr(self, "query_proj_layers"):
                    q_proj = self.query_proj_layers[i](
                        sep_token_states.unsqueeze(1)
                    )
                    k_proj = self.key_proj_layers[i](current_hidden_states)
                    v_proj = self.value_proj_layers[i](current_hidden_states)
                else:
                    q_proj = sep_token_states.unsqueeze(1)
                    k_proj = current_hidden_states
                    v_proj = current_hidden_states

                attn_output, attn_weight = self.attention_layers[i](
                    q_proj,
                    k_proj,
                    v_proj,
                    key_padding_mask=key_padding_mask,
                )

                norm_attn_outputs.append(self.norm(attn_output))
                attn_weights.append(attn_weight)

            if return_dict:
                norm_attn_outputs_for_save = torch.cat(
                    norm_attn_outputs,
                    dim=1,
                )

            norm_attn_outputs = torch.sum(
                torch.cat(norm_attn_outputs, dim=1),
                dim=1,
            )
            attn_weights = torch.cat(attn_weights, dim=1)

        else:
            batch_size = hidden_states.size(0)
            batch_indices = torch.arange(
                batch_size,
                device=hidden_states.device,
            )

            # Use the validated terminal [SEP] hidden state as the query.
            sep_token_states = hidden_states[
                batch_indices,
                sep_indices,
            ]
            key_padding_mask = ~attention_mask

            if hasattr(self, "query_proj"):
                q_proj = self.query_proj(sep_token_states.unsqueeze(1))
                k_proj = self.key_proj(hidden_states)
                v_proj = self.value_proj(hidden_states)
            else:
                q_proj = sep_token_states.unsqueeze(1)
                k_proj = hidden_states
                v_proj = hidden_states

            attn_output, attn_weights = self.attention(
                q_proj,
                k_proj,
                v_proj,
                key_padding_mask=key_padding_mask,
            )
            attn_output = attn_output.squeeze(1)
            norm_attn_outputs = self.norm(attn_output)

            if return_dict:
                norm_attn_outputs_for_save = norm_attn_outputs

        logits = self.classify(norm_attn_outputs)
        predicted_probs = logits

        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(predicted_probs, labels)
        else:
            loss = None

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return {
            "predicted_probs": predicted_probs,
            "labels": labels,
            "loss": loss,
        }
