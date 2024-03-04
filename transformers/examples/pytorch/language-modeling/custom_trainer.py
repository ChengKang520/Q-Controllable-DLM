import torch, wandb
from transformers import PreTrainedModel, GPT2PreTrainedModel, GPT2Config, GPT2Model, BertPreTrainedModel, BertModel, \
    GPT2LMHeadModel, BertForMaskedLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, NextSentencePredictorOutput
from transformers.models.bert.modeling_bert import BertOnlyNSPHead
from torch import nn
from transformers import Trainer, GPT2PreTrainedModel, PreTrainedModel, DataCollator, TrainingArguments, EvalPrediction, \
    TrainerCallback
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
import collections
from transformers.utils import logging
from  transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedTensorGatherer,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from quantize_layers import *
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib import cm

logger = logging.get_logger(__name__)

class Trainer_8dim(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, torch.nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional["PreTrainedTokenizerBase"] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            task = None, sep=None

    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                         compute_metrics, callbacks, optimizers)

        self.task = task
        self.sep = sep

        return



    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        # compute_loss
        if model.model_name.startswith('gpt2'):
            if model.tuning_mode == 'full':
                outputs = model(**inputs, predict_logp=loss)
            else:
                outputs = model(**inputs, predict_logp=loss, transformer_base_model=self.gpt2)


        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

class GPT2LMHeadModelCompress(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if hasattr(config, 'reduced_emb'):
            self.mlp_dim = config.reduced_emb
        else:
            self.mlp_dim = 8
        self.down_proj = nn.Sequential(nn.Linear(config.n_embd, self.mlp_dim * 4), nn.Tanh(),
                                       nn.Linear(self.mlp_dim * 4, self.mlp_dim))
        self.up_proj = nn.Sequential(nn.Linear(self.mlp_dim, self.mlp_dim * 4), nn.Tanh(),
                                       nn.Linear(self.mlp_dim * 4, config.n_embd))
        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    # @add_start_docstrings(PARALLELIZE_DOCSTRING)
    # def parallelize(self, device_map=None):
    #     self.device_map = (
    #         get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
    #         if device_map is None
    #         else device_map
    #     )
    #     assert_device_map(self.device_map, len(self.transformer.h))
    #     self.transformer.parallelize(self.device_map)
    #     self.lm_head = self.lm_head.to(self.transformer.first_device)
    #     self.model_parallel = True
    #
    # @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    # def deparallelize(self):
    #     self.transformer.deparallelize()
    #     self.transformer = self.transformer.to("cpu")
    #     self.lm_head = self.lm_head.to("cpu")
    #     self.model_parallel = False
    #     torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }


    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(input_ids.shape, 'input_ids', )

        input_embs = self.transformer.wte(input_ids) # input_embs
        down_proj = self.down_proj(input_embs)
        # print(down_proj.shape)
        input_embs2 = self.up_proj(down_proj)

        transformer_outputs = self.transformer(
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=input_embs2,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
    
    

class BERTModelCompress(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if hasattr(config, 'reduced_emb'):
            self.mlp_dim = config.reduced_emb
        else:
            self.mlp_dim = 8
        self.down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size//2), nn.Tanh(),
                                       nn.Linear(config.hidden_size//2, self.mlp_dim))
        self.up_proj = nn.Sequential(nn.Linear(self.mlp_dim, config.hidden_size//2), nn.Tanh(),
                                       nn.Linear(config.hidden_size//2, config.hidden_size))
        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()



    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }


    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(input_ids.shape, 'input_ids', )


        transformer_outputs = self.bert(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        down_proj = self.down_proj(hidden_states)
        embs2 = self.up_proj(down_proj)
        lm_logits = self.lm_head(embs2)
        # print(down_proj.shape, embs2.shape, lm_logits.shape)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # shift_logits = lm_logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()
            shift_logits = lm_logits
            shift_labels = labels
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=down_proj, #transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
class AR_for_cont(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if hasattr(config, 'sigma'):
            self.sigma = config.sigma
        else:
            self.sigma = 1.

        self.model_parallel = False
        self.device_map = None
            
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }


    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        noise=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(input_ids.shape, 'input_ids', )

        input_embs = self.transformer.wte(input_ids) # input_embs
        if noise is None:
            noise = torch.randn_like(input_embs)
        input_embs2 = input_embs + self.sigma * noise

        transformer_outputs = self.transformer(
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=input_embs2,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states) # might be better to be L2.

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    


class AutoEncoderWithNoise(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config, config2=None):
        super().__init__(config)
        self.bert = BertModel(config)
        if hasattr(config, 'reduced_emb'):
            self.mlp_dim = config.reduced_emb
        else:
            self.mlp_dim = 8

        if hasattr(config, 'sigma'):
            self.sigma = config.sigma
        else:
            self.sigma = 1.
        self.down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size // 2), nn.Tanh(),
                                       nn.Linear(config.hidden_size // 2, self.mlp_dim))
        self.up_proj = nn.Sequential(nn.Linear(self.mlp_dim, config.hidden_size // 2), nn.Tanh(),
                                     nn.Linear(config.hidden_size // 2, config.hidden_size))

        if hasattr(config, 'rounding_mode'):
            self.rounding_mode = config.rounding_mode
        else:
            self.rounding_mode = 'gpt2'

        if self.rounding_mode== 'gpt2':
            config2.vocab_size = config.vocab_size
            self.decoder = GPT2LMHeadModel(config2)
        elif self.rounding_mode == 'bert':
            self.decoder = BertForMaskedLM(config)
        elif self.rounding_mode == 'conv':
            raise NotImplementedError
        elif self.rounding_mode == 'mlp':
            self.decoder = nn.Sequential(nn.Linear(self.mlp_dim, config.hidden_size // 2), nn.Tanh(),
                                         nn.Linear(config.hidden_size // 2, config.hidden_size))
        else:
            assert False, 'invalid rounding_mode'
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(input_ids.shape, 'input_ids', )

        transformer_outputs = self.bert(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        down_proj = self.down_proj(hidden_states)
        down_proj_norm = torch.norm(down_proj, dim=-1) #bsz, seqlen
        clamped_norm = torch.clamp(down_proj_norm, min=1) #bsz, seqlen
        clamped_down_proj = down_proj / clamped_norm.unsqueeze(-1)
        gaussian_noise = torch.randn(clamped_down_proj.shape).to(clamped_down_proj.device) * self.sigma
        noised_z = clamped_down_proj + gaussian_noise


        if self.rounding_mode== 'gpt2':
            embs2 = self.up_proj(noised_z)
            decoder_outputs = self.decoder(input_ids, encoder_hidden_states=embs2)
            lm_logits = decoder_outputs.logits[:, :-1].contiguous()
            labels = labels[..., 1:].contiguous()
            # version 1. concatenate these at the beginning to attend to them all at once.


        elif self.rounding_mode== 'gpt2_v2':
            # version 2. concatenate these at each token position, one by one.
            embs2 = self.up_proj(noised_z)
            input_ids_embs = self.decoder.embeddings(input_ids)
            concat_embs = embs2 + input_ids_embs  # bsz, seqlen, dim
            decoder_outputs = self.decoder(inputs_embeds=concat_embs)
            lm_logits = decoder_outputs.logits[:, :-1].contiguous()
            labels = labels[..., 1:].contiguous()

        elif self.rounding_mode== 'gpt2_v3':
            # version 2. concatenate these at each token position, one by one.
            embs2 = self.up_proj(noised_z)
            input_ids_embs = self.decoder.embeddings(input_ids)
            concat_embs = torch.cat([embs2, input_ids_embs], dim=1)  # bsz, seqlen, dim
            decoder_outputs = self.decoder(inputs_embeds=concat_embs)
            lm_logits = decoder_outputs.logits[:, embs2.size(1) - 1:-1].contiguous()


        elif self.rounding_mode == 'bert':
            embs2 = self.up_proj(noised_z)
            decoder_outputs = self.decoder(inputs_embeds=embs2)
            lm_logits = decoder_outputs.logits

        elif self.rounding_mode == 'conv':
            raise NotImplementedError

        elif self.rounding_mode == 'mlp':
            embs2 = self.up_proj(noised_z)
            embs2 = self.decoder(embs2)
            lm_logits = self.lm_head(embs2)


        # print(down_proj.shape, embs2.shape, lm_logits.shape)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # shift_logits = lm_logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()
            shift_logits = lm_logits
            shift_labels = labels
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # print(lm_logits.shape)
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=down_proj,  # transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def half_forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(input_ids.shape, 'input_ids', )

        transformer_outputs = self.bert(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        down_proj = self.down_proj(hidden_states)
        down_proj_norm = torch.norm(down_proj, dim=-1)  # bsz, seqlen
        clamped_norm = torch.clamp(down_proj_norm, min=1)  # bsz, seqlen
        clamped_down_proj = down_proj / clamped_norm.unsqueeze(-1)
        # print(torch.norm(clamped_down_proj, dim=-1) , torch.norm(clamped_down_proj, dim=-1).shape)
        # gaussian_noise = torch.randn(clamped_down_proj.shape).to(clamped_down_proj.device) * self.sigma
        # noised_z = clamped_down_proj + gaussian_noise

        # print(down_proj.shape, embs2.shape, lm_logits.shape)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        loss = None
        lm_logits = None 

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # print(lm_logits.shape)
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=clamped_down_proj,  # transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class GPT2VAE(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if hasattr(config, 'sigma'):
            self.sigma = config.sigma
        else:
            self.sigma = 1.

        if hasattr(config, 'reduced_emb'):
            self.latent_dim = config.reduced_emb
        else:
            self.latent_dim = 8

        if hasattr(config, 'mlp_dim'):
            self.mlp_dim = config.mlp_dim
        else:
            self.mlp_dim = 128

        self.q = nn.Embedding(config.vocab_size,  self.latent_dim)
        self.g_theta = nn.Sequential(nn.Linear(self.latent_dim, self.mlp_dim), nn.Tanh(),
                                       nn.Linear(self.mlp_dim, config.vocab_size))




        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    # @add_start_docstrings(PARALLELIZE_DOCSTRING)
    # def parallelize(self, device_map=None):
    #     self.device_map = (
    #         get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
    #         if device_map is None
    #         else device_map
    #     )
    #     assert_device_map(self.device_map, len(self.transformer.h))
    #     self.transformer.parallelize(self.device_map)
    #     self.lm_head = self.lm_head.to(self.transformer.first_device)
    #     self.model_parallel = True
    #
    # @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    # def deparallelize(self):
    #     self.transformer.deparallelize()
    #     self.transformer = self.transformer.to("cpu")
    #     self.lm_head = self.lm_head.to("cpu")
    #     self.model_parallel = False
    #     torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }


    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(input_ids.shape, 'input_ids', )
        bsz, seqlen = input_ids.shape

        embs = self.q(input_ids)
        z_sample = torch.randn(bsz, seqlen, self.latent_dim).to(embs.device)
        z_embed = z_sample * self.sigma + embs

        logits = self.g_theta(z_embed)
        loss_fct = CrossEntropyLoss(reduction='none')
        loss_first = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss_first = loss_first.view(labels.shape)[:, 1:]


        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)
        lm_logits = lm_logits[..., :-1, :].contiguous() # bsz, seqlen-1, vocab

        embeds_q = self.q.weight
        vocab_size = embeds_q.size(0)


        # z_embed.shape # bsz * seqlen, dim
        # embeds_q # vocab, dim
        D = torch.sum(z_embed.view(-1, z_embed.size(-1))**2, axis=-1).unsqueeze(1).expand(-1, vocab_size) + \
            torch.sum(embeds_q**2, axis=-1).unsqueeze(0).expand(bsz*seqlen, -1) \
            - 2*torch.mm(z_embed.view(-1, z_embed.size(-1)),
                         embeds_q.transpose(0,1)).view(bsz*seqlen, vocab_size)
        D = D.view(bsz, seqlen, vocab_size)
        D = D[:, 1:].contiguous()
        nll = -torch.logsumexp(lm_logits - D / (2.0 * self.sigma**2), dim=-1) \
            + torch.logsumexp(lm_logits, dim=-1)

        loss = loss_first - nll
        # print(loss_first.shape, nll.shape, loss.shape)
        loss = loss.mean()



        # loss = None
        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     shift_logits = lm_logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

class Classifier_POS2(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config, diffusion=None):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.transformer.wte = nn.Embedding(config.vocab_size,config.input_emb_dim, )
        self.pos_wte = nn.Embedding(config.pos_vocab_size, config.input_emb_dim, )
        # self.lm_head = nn.Linear(config.input_emb_dim, config.vocab_size, bias=False)

        self.up_proj = nn.Sequential(nn.Linear(config.input_emb_dim, config.input_emb_dim * 4), nn.Tanh(),
                                     nn.Linear(config.input_emb_dim * 4, config.n_embd))
        print(diffusion)
        self.diffusion = diffusion
        if diffusion is not None:
            self.train_diff_steps = config.train_diff_steps
        else:
            self.train_diff_steps = 200
        self.time_embeddings = nn.Embedding(self.train_diff_steps+1, config.n_embd)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
        self.lm_head2 = nn.Linear(config.n_embd, config.vocab_size, bias=False)


    # def get_output_embeddings(self):
    #     return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        print(kwargs.keys(), input_ids)
        token_type_ids = kwargs.get("token_type_ids", None)
        pos_ids = kwargs.get("pos_ids", None)
        t = kwargs.get("t", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            pos_ids = input_ids[:, -1].unsqueeze(-1)
            input_ids = None
            t = None
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        attention_mask = None
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "pos_ids": pos_ids,
            "t":t,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
            self,
            input_ids=None,
            pos_ids=None,
            input_embs=None,
            pos_embs=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            t = 200,
            t_aware=True,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(input_ids.shape, 'input_ids', )

        if input_embs is None and past_key_values is None:
            # print(input_ids.shape, pos_ids.shape)
            input_embs = self.transformer.wte(input_ids)  # input_embs
            print(pos_ids)
            pos_embs = self.pos_wte(pos_ids)

        if past_key_values is not None: # decoding mode.
            print(pos_ids)
            pos_embs = self.pos_wte(pos_ids)
            assert t is None


        if self.diffusion is not None:
            if self.train_diff_steps > 0:
                # sample t
                t = torch.randint(-1, self.train_diff_steps, (input_embs.shape[0],)).to(input_embs.device)
                t_mask = (t >= 0)
                input_embs_rand = self.diffusion.q_sample(input_embs, t)
                input_embs[t_mask] = input_embs_rand[t_mask]
                t[~t_mask] = self.train_diff_steps
                time_emb = self.time_embeddings(t).unsqueeze(1)

                # t = torch.randint(0, self.train_diff_steps+1, (input_embs.shape[0],)).to(input_embs.device)
                # t_mask = (t < self.train_diff_steps)
                # input_embs_rand = self.diffusion.q_sample(input_embs, t)
                # input_embs[t_mask] = input_embs_rand[t_mask]
                # print(input_embs.shape, t[:3])
                # print(self.time_embeddings, t)
                # time_emb = self.time_embeddings(t).unsqueeze(1)

        if self.diffusion is None and t is not None:
            # print(t, input_embs.shape, 'should see this')
            t = torch.LongTensor([t]).expand(input_embs.size(0)).to(self.device)
            time_emb = self.time_embeddings(t).unsqueeze(1)

        if past_key_values is None:
            input_embs = torch.cat([input_embs, pos_embs], dim=1)
            input_embs = self.up_proj(input_embs)
            if t_aware:
                input_embs = torch.cat([time_emb, input_embs], dim=1)

        else:
            input_embs = pos_embs
            input_embs = self.up_proj(input_embs)

        transformer_outputs = self.transformer(
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=input_embs,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if t_aware and past_key_values is None:
            hidden_states = transformer_outputs[0][:, 1:, ]
            # print(hidden_states)
        else:
            hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # print(hidden_states.shape, self.lm_head2, self.lm_head2.weight.shape)
        lm_logits = self.lm_head2(hidden_states)
        # print(lm_logits.shape, hidden_states.shape, labels.shape)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class Classifier_Tree(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config, diffusion=None):
        super().__init__(config)
        # config.input_emb_dim = 16
        self.transformer = BertModel(config)
        self.transformer.embeddings.word_embeddings = nn.Embedding(config.vocab_size, config.input_emb_dim, )
        # self.pos_wte = nn.Embedding(config.pos_vocab_size, config.input_emb_dim, )
        # self.lm_head = nn.Linear(config.input_emb_dim, config.vocab_size, bias=False)

        print(self.transformer)

        # print('******************************************')
        # print('Freeze Pra-trained Language Models!')
        # for name, param in self.transformer.named_parameters():
        #     print(name)
        #     param.requires_grad = False
        #
        # print('Only Train these Weights:')
        # for name, param in self.transformer.named_parameters():
        #     if ('lora' in name) or ('inhibition' in name):
        #         print(name)
        #         param.requires_grad = True
        # print('******************************************')

        if config.quant_m == 'original':
            self.up_proj = nn.Sequential(
                nn.Linear(config.input_emb_dim, config.input_emb_dim * 4),
                nn.Tanh(),
                nn.Linear(config.input_emb_dim * 4, config.hidden_size)
            )
        else:
            print('*****************  quantized up_proj  *****************')
            print(config.quant_m)
            print('*****************  quantized up_proj  *****************')
            self.up_proj = nn.Sequential(
                LinearQuant(config.input_emb_dim, config.hidden_size, bias=True, H=1, W_LR_scale="Glorot", config_q=config.quant_m, weight_f_width=config.weight_f_width, weight_i_width=config.weight_i_width),
                )

        # print(diffusion)
        self.diffusion = diffusion
        if diffusion is not None:
            self.train_diff_steps = config.train_diff_steps
        else:
            self.train_diff_steps = 200
        self.time_embeddings = nn.Embedding(self.train_diff_steps+1, config.hidden_size)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
        # self.lm_head2 = nn.Linear(config.hidden_size, config.tree_vocab_size, bias=False)

        if config.quant_m == 'original':
            self.lm_head2 = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, config.tree_vocab_size, bias=False)
            )
        else:
            print('*****************  quantized lm_head2  *****************')
            self.lm_head2 = LinearQuant(config.hidden_size * 2, config.tree_vocab_size, bias=False, H=1, W_LR_scale="Glorot", config_q=config.quant_m,
                        weight_f_width=config.weight_f_width, weight_i_width=config.weight_i_width)




    # def get_output_embeddings(self):
    #     return self.lm_head

    def set_input_embeddings(self, new_embeddings):
        self.transformer.embeddings.word_embeddings = new_embeddings

    def get_input_embeddings(self):
        return self.transformer.embeddings.word_embeddings

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids=None,
            parse_chart=None,
            input_embs=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            t = 200,
            t_aware=True,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        labels = parse_chart

        # print('PARSING MODEL IS TRAINING')
        # print(input_ids.shape, 'input_ids', )

        if input_embs is None:
            input_embs = self.transformer.embeddings.word_embeddings(input_ids)  # input_embs


        if self.diffusion is not None:
            if self.train_diff_steps > 0:
                # sample t
                t = torch.randint(-1, self.train_diff_steps, (input_embs.shape[0],)).to(input_embs.device)
                t_mask = (t >= 0)
                input_embs_rand = self.diffusion.q_sample(input_embs, t)
                input_embs[t_mask] = input_embs_rand[t_mask]
                t[~t_mask] = self.train_diff_steps
                time_emb = self.time_embeddings(t).unsqueeze(1)


        if self.diffusion is None and t is not None:
            # print(t, input_embs.shape, 'should see this')
            t = torch.LongTensor([t]).expand(input_embs.size(0)).to(self.device)
            time_emb = self.time_embeddings(t).unsqueeze(1)

        input_embs = self.up_proj(input_embs)
        if t_aware:
            input_embs = torch.cat([time_emb, input_embs], dim=1)


        transformer_outputs = self.transformer(
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=input_embs,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if t_aware and past_key_values is None:
            hidden_states = transformer_outputs[0][:, 1:, ]
            # print(hidden_states)
        else:
            hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # print(hidden_states.shape, self.lm_head2, self.lm_head2.weight.shape)
        # span_features = torch.unsqueeze(hidden_states, 1) - torch.unsqueeze(hidden_states, 2)
        # lm_logits = self.lm_head2(span_features)
        span_features = torch.cat([torch.unsqueeze(hidden_states, 1).expand(-1, hidden_states.size(1), -1, -1),
                                   torch.unsqueeze(hidden_states, 2).expand(-1, -1, hidden_states.size(1), -1)],
                                  dim=-1)
        lm_logits = self.lm_head2(span_features)
        #[:, :-1, 1:]
        # because the first token cannot the end of a span, and
        # the second token cannot be the start of a span.
        # print(span_features.shape, 'span_features')
        # print(lm_logits.shape)
        # print(parse_chart.shape)
        # print(lm_logits.shape, hidden_states.shape, labels.shape)

        loss = None
        if parse_chart is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits
            shift_labels = parse_chart
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class Classifier_POS(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config, diffusion=None):
        super().__init__(config)
        self.transformer = BertModel(config)
        self.transformer.embeddings.word_embeddings = nn.Embedding(config.vocab_size, config.input_emb_dim)
        # self.pos_wte = nn.Embedding(config.pos_vocab_size, config.input_emb_dim, )
        # self.lm_head = nn.Linear(config.input_emb_dim, config.vocab_size, bias=False)

        print('*****************  apply_lora  *****************')
        if config.apply_lora:
            self.apply_lora = 'LoRA'
            print(self.apply_lora)
        else:
            self.apply_lora = 'Standard'
            print(self.apply_lora)
        print('*****************  apply_lora  *****************')


        self.quant_m = config.quant_m
        self.weight_i_width = config.weight_i_width
        self.weight_f_width = config.weight_f_width

        if config.quant_m == 'original':
            self.up_proj = nn.Sequential(
                nn.Linear(config.input_emb_dim, config.input_emb_dim * 4),
                nn.Tanh(),
                nn.Linear(config.input_emb_dim * 4, config.hidden_size)
            )
        else:
            print('*****************  quantized up_proj  *****************')
            self.up_proj = nn.Sequential(
                LinearQuant(config.input_emb_dim, config.hidden_size, bias=True, H=1, W_LR_scale="Glorot", config_q=config.quant_m, weight_f_width=config.weight_f_width, weight_i_width=config.weight_i_width),
                )

        print(diffusion)
        self.diffusion = diffusion
        if diffusion is not None:
            self.train_diff_steps = config.train_diff_steps
        else:
            self.train_diff_steps = 200
        self.time_embeddings = nn.Embedding(self.train_diff_steps+1, config.hidden_size)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

        if config.quant_m == 'original':
            self.lm_head2 = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                          nn.Tanh(),
                                          nn.Linear(config.hidden_size, config.pos_vocab_size, bias=False))

        else:
            print('*****************  quantized lm_head2  *****************')
            self.lm_head2 = LinearQuant(config.hidden_size, config.pos_vocab_size, bias=False, H=1, W_LR_scale="Glorot", config_q=config.quant_m,
                        weight_f_width=config.weight_f_width, weight_i_width=config.weight_i_width)


    # def get_output_embeddings(self):
    #     return self.lm_head

    def set_input_embeddings(self, new_embeddings):
        self.transformer.embeddings.word_embeddings = new_embeddings

    def get_input_embeddings(self):
        return self.transformer.embeddings.word_embeddings

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids=None,
            pos_ids=None,
            input_embs=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            t = 200,
            t_aware=True,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        labels = pos_ids
        # print(input_ids.shape, 'input_ids', )

        if input_embs is None:
            # print(input_ids.shape, pos_ids.shape)
            input_embs = self.transformer.embeddings.word_embeddings(input_ids)  # input_embs


        if self.diffusion is not None:
            if self.train_diff_steps > 0:
                # sample t
                t = torch.randint(-1, self.train_diff_steps, (input_embs.shape[0],)).to(input_embs.device)
                t_mask = (t >= 0)
                input_embs_rand = self.diffusion.q_sample(input_embs, t)
                input_embs[t_mask] = input_embs_rand[t_mask]
                t[~t_mask] = self.train_diff_steps
                time_emb = self.time_embeddings(t).unsqueeze(1)

                # t = torch.randint(0, self.train_diff_steps+1, (input_embs.shape[0],)).to(input_embs.device)
                # t_mask = (t < self.train_diff_steps)
                # input_embs_rand = self.diffusion.q_sample(input_embs, t)
                # input_embs[t_mask] = input_embs_rand[t_mask]
                # print(input_embs.shape, t[:3])
                # print(self.time_embeddings, t)
                # time_emb = self.time_embeddings(t).unsqueeze(1)

        if self.diffusion is None and t is not None:
            # print(t, input_embs.shape, 'should see this')
            t = torch.LongTensor([t]).expand(input_embs.size(0)).to(self.device)
            time_emb = self.time_embeddings(t).unsqueeze(1)

        input_embs = self.up_proj(input_embs)



        print('***********************************************  Embedding t-SNE  *********************************************')
        length_emd = 15
        tSNE_input_ids = input_ids[:, :length_emd].cpu().detach().numpy() #
        tSNE_input_embs = input_embs[:, :length_emd, :].cpu().detach().numpy() # .cpu()
        tSNE_input_labels = labels[:, :length_emd].cpu().detach().numpy()

        # print('**** tSNE_input_ids ****')
        # print(tSNE_input_ids.shape)
        tSNE_input_ids = np.squeeze(tSNE_input_ids.reshape(32*length_emd))
        # print(tSNE_input_ids)
        # print('**** tSNE_input_ids ****')

        # print('**** tSNE_input_embs ****')
        # print(tSNE_input_embs.shape)
        tSNE_input_embs = np.moveaxis(tSNE_input_embs, -1, 0)
        # print(tSNE_input_embs.shape)
        tSNE_input_embs = tSNE_input_embs.reshape(768, 32*length_emd)
        # print(tSNE_input_embs.shape)
        tSNE_input_embs = np.moveaxis(tSNE_input_embs, -1, 0)
        # print(tSNE_input_embs)
        # print('**** tSNE_input_embs ****')

        # print('**** tSNE_input_labels ****')
        # print(tSNE_input_labels.shape)
        tSNE_input_labels = np.squeeze(tSNE_input_labels.reshape(32*length_emd))
        # print(tSNE_input_labels)
        # print('**** tSNE_input_labels ****')

        x_min, x_max = np.min(tSNE_input_embs, 0), np.max(tSNE_input_embs, 0)
        tSNE_input_embs = (tSNE_input_embs - x_min) / (x_max - x_min)



        # target_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        target_names = [0, 1, 3, 4, 5, 7, 8, 10, 11, 13]
        tsne_label = tSNE_input_labels
        X_3d_tsne = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3).fit_transform(tSNE_input_embs)
        fig1 = plt
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'gray', 'orange', 'saddlebrown'
        markers = 'x', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'x' # '^', '^', '^', '^', '^'
        fig = fig1.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        fig.add_axes(ax)
        for i, c, m, label in zip(target_names, colors, markers, target_names):
            ax.scatter(X_3d_tsne[tsne_label==i, 0], X_3d_tsne[tsne_label==i, 1], X_3d_tsne[tsne_label==i, 2], color=c, marker=m, label=label)
            # ax.scatter(X_3d_tsne[tsne_label==i, 0], X_3d_tsne[tsne_label==i, 1], color=c, marker=m, label=label, zdir='z', zs=-30)
        ax.view_init(azim=-66, elev=12)
        _ = ax.text2D(0.05, 0.1, fontsize=12, s=self.apply_lora + '_' + self.quant_m + str(self.weight_i_width) + str(
            self.weight_f_width), transform=ax.transAxes, bbox=dict(facecolor='white', alpha=1, edgecolor='black', linewidth=0.2, boxstyle='square'))

        fig1.legend(loc = 'center left')
        for i, c, m, label in zip(target_names, colors, markers, target_names):
            # ax.plot(X_3d_tsne[tsne_label==i, 0], X_3d_tsne[tsne_label==i, 1], 'o', zs=-50, zdir='z', color=c, marker=m)
            ax.scatter(X_3d_tsne[tsne_label==i, 0], X_3d_tsne[tsne_label==i, 1], zs=-50, zdir='z', color=c, marker=m, label=label)


        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)
        # ax.set_zlabel('Z')
        ax.set_zlim(-50, 25)

        save_file = 'POS_' + self.apply_lora + '_' + self.quant_m + str(self.weight_i_width) + str(
            self.weight_f_width) + '_embedding_tSNE.png'
        # fig1.title('The t-SNE of the Embedding Vector')

        fig1.show()
        print(save_file)
        fig1.savefig(save_file)
        fig1.close()


        # plot_emd = plt
        # plot_emd.figure(figsize=(6, 5))
        # for i in range(tsne_data.shape[0]):
        #     plot_emd.scatter(tsne_data[i, 0], tsne_data[i, 1], color=plot_emd.cm.Set1(tsne_label[i] / 20.), label=tsne_label)
        #     # plot_emd.text(tsne_data[i, 0], tsne_data[i, 1], str(tsne_label[i]), color=plot_emd.cm.Set1(tsne_label[i] / 20.), fontdict={'weight': 'bold', 'size': 9})
        # plot_emd.show()
        # if self.apply_lora:
        #     save_file = 'POS_' + 'LoRA_'+ self.quant_m + str(self.weight_i_width) + str(
        #         self.weight_f_width) + '_embedding_tSNE2.png'
        # else:
        #     save_file = 'POS_' + self.quant_m + str(self.weight_i_width) + str(self.weight_f_width) + '_embedding_tSNE2.png'
        # plot_emd.savefig(save_file)
        # plot_emd.close()
        print('***********************************************  Embedding t-SNE  *********************************************')



        if t_aware:
            input_embs = torch.cat([time_emb, input_embs], dim=1)



        transformer_outputs = self.transformer(
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=input_embs,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if t_aware and past_key_values is None:
            hidden_states = transformer_outputs[0][:, 1:, ]
            # print(hidden_states)
        else:
            hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # print(hidden_states.shape, self.lm_head2, self.lm_head2.weight.shape)
        lm_logits = self.lm_head2(hidden_states)
        # print(lm_logits.shape, hidden_states.shape, labels.shape)

        loss = None
        if pos_ids is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits
            shift_labels = labels
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class Classifier_GPT2(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config, diffusion=None,
                 quant_m = 'original',
                 weight_i_width = 0,
                 weight_f_width = 0):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.transformer.wte = nn.Embedding(config.vocab_size,config.input_emb_dim, )
        # self.lm_head = nn.Linear(config.input_emb_dim, config.vocab_size, bias=False)


        if quant_m == 'original':
            self.up_proj = nn.Sequential(
                nn.Linear(config.input_emb_dim, config.input_emb_dim * 4),
                nn.Tanh(),
                nn.Linear(config.input_emb_dim * 4, config.n_embd)
            )
        else:
            print('*****************  quantized up_proj  *****************')
            print(quant_m)
            print('*****************  quantized up_proj  *****************')
            self.up_proj = nn.Sequential(
                LinearQuant(config.input_emb_dim, config.n_embd, bias=True, H=1, W_LR_scale="Glorot", config_q=quant_m, weight_f_width=weight_f_width, weight_i_width=weight_i_width),
                )

        print(diffusion)
        self.diffusion = diffusion
        if diffusion is not None:
            self.train_diff_steps = config.train_diff_steps
        else:
            self.train_diff_steps = 200
        self.time_embeddings = nn.Embedding(self.train_diff_steps+1, config.n_embd)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

        if quant_m == 'original':
            self.lm_head2 = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        else:
            print('*****************  quantized lm_head2  *****************')
            self.lm_head2 = LinearQuant(config.n_embd, config.vocab_size, bias=False, H=1, W_LR_scale="Glorot", config_q=quant_m,
                        weight_f_width=weight_f_width, weight_i_width=weight_i_width)


    # def get_output_embeddings(self):
    #     return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
            self,
            input_ids=None,
            input_embs=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            t = 200,
            t_aware=True,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(input_ids.shape, 'input_ids', )

        if input_embs is None:
            input_embs = self.transformer.wte(input_ids)  # input_embs

        # if self.diffusion is not None:
        if self.train_diff_steps > 0 and t is None:
            # sample t
            t = torch.randint(-1, self.train_diff_steps, (input_embs.shape[0],)).to(input_embs.device)
            t_mask = (t >= 0)
            input_embs_rand = self.diffusion.q_sample(input_embs, t)
            input_embs[t_mask] = input_embs_rand[t_mask]
            t[~t_mask] = self.train_diff_steps
            time_emb = self.time_embeddings(t).unsqueeze(1)

            # t = torch.randint(0, self.train_diff_steps+1, (input_embs.shape[0],)).to(input_embs.device)
            # t_mask = (t < self.train_diff_steps)
            # input_embs_rand = self.diffusion.q_sample(input_embs, t)
            # input_embs[t_mask] = input_embs_rand[t_mask]
            # print(input_embs.shape, t[:3])
            # print(self.time_embeddings, t)
            # time_emb = self.time_embeddings(t).unsqueeze(1)

        # if self.diffusion is None and t is not None:
            # print(t, input_embs.shape, 'should see this')
        t = torch.LongTensor([t]).expand(input_embs.size(0)).to(self.device)
        time_emb = self.time_embeddings(t).unsqueeze(1)

        input_embs = self.up_proj(input_embs)
        if t_aware:
            input_embs = torch.cat([time_emb, input_embs], dim=1)

        transformer_outputs = self.transformer(
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=input_embs,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if t_aware:
            hidden_states = transformer_outputs[0][:, 1:, ]
            # print(hidden_states)
        else:
            hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # print(hidden_states.shape, self.lm_head2, self.lm_head2.weight.shape)
        lm_logits = self.lm_head2(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class Classifier_Consistency(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config, diffusion=None):
        super().__init__(config)
        self.bert = BertModel(config)
        self.bert.embeddings.word_embeddings = nn.Embedding(config.vocab_size, config.input_emb_dim, )
        self.cls = BertOnlyNSPHead(config)
        self.up_proj = nn.Sequential(nn.Linear(config.input_emb_dim, config.input_emb_dim * 4), nn.Tanh(),
                                     nn.Linear(config.input_emb_dim * 4, config.hidden_size))
        print(diffusion)
        self.diffusion = diffusion
        if diffusion is not None:
            self.train_diff_steps = config.train_diff_steps
        else:
            self.train_diff_steps = 200
        self.time_embeddings = nn.Embedding(self.train_diff_steps+1, config.hidden_size)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def set_input_embeddings(self, new_embeddings):
        self.bert.embeddings.word_embeddings = new_embeddings

    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids=None,
            context_ids=None,
            type_ids=None,
            mid_ids=None,
            input_embs=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            t = 200,
            t_aware=True,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            assert context_ids is not None and (mid_ids is not None or input_embs is not None)
            context_embs = self.bert.embeddings.word_embeddings(context_ids)
            context_type_ids = torch.full_like(context_ids, 0)
        else: # training
            assert type_ids is not None
            context_input_embs = self.bert.embeddings.word_embeddings(input_ids)
            context_input_type_ids = type_ids
            # input_embs = context_input_embs[context_input_type_ids == 1]
            input_embs = context_input_embs.clone()

        # if input_embs is None: # testing.
        #     input_embs = self.transformer.embeddings.word_embeddings(mid_ids)  # input_embs




        if self.diffusion is not None: # training
            if self.train_diff_steps > 0:
                # sample t
                # print(input_embs.shape)
                t = torch.randint(-1, self.train_diff_steps, (input_embs.shape[0],)).to(input_embs.device)
                t_mask = (t >= 0)
                input_embs_rand = self.diffusion.q_sample(input_embs, t)
                input_embs[t_mask] = input_embs_rand[t_mask]
                t[~t_mask] = self.train_diff_steps
                time_emb = self.time_embeddings(t).unsqueeze(1)

        if self.diffusion is None and t is not None: # testing.
            # print(t, input_embs.shape, 'should see this')
            t = torch.LongTensor([t]).expand(input_embs.size(0)).to(self.device)
            time_emb = self.time_embeddings(t).unsqueeze(1)
        # print(context_input_type_ids.shape, input_embs.shape, context_input_embs.shape)
        # context_input_embs = torch.where((context_input_type_ids == 1), input_embs, context_input_embs)
        context_input_embs[context_input_type_ids == 1] = input_embs[context_input_type_ids == 1]
        # context_input_embs[context_input_type_ids == 1] = input_embs
        context_input_embs = self.up_proj(context_input_embs)

        input_embs = context_input_embs #torch.cat([context_embs, context_input_embs], dim=1)
        # token_type_ids = torch.cat([context_type_ids, input_type_ids], dim=1)
        if t_aware:
            # print(time_emb.shape, input_embs.shape, input_ids.shape, type_ids.shape)
            input_embs = torch.cat([time_emb, input_embs], dim=1)
            t_type_ids = torch.LongTensor([0]).unsqueeze(0).expand(input_embs.shape[0], -1).to(self.device)
            token_type_ids = torch.cat([t_type_ids,context_input_type_ids], dim=1)


        attention_mask = (token_type_ids != 2)

        transformer_outputs = self.bert(
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=input_embs,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if t_aware and past_key_values is None:
            hidden_states = transformer_outputs[0][:, 1:, ]
            # print(hidden_states)
        else:
            hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # print(hidden_states.shape, self.lm_head2, self.lm_head2.weight.shape)
        pooled_output = transformer_outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        # print(labels)
        # print(seq_relationship_scores.shape)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=hidden_states,
            attentions=transformer_outputs.attentions,
        )


class Classifier_Times(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config, diffusion=None):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.transformer.wte = nn.Embedding(config.vocab_size,config.input_emb_dim, )
        # self.lm_head = nn.Linear(config.input_emb_dim, config.vocab_size, bias=False)

        self.up_proj = nn.Sequential(nn.Linear(config.input_emb_dim, config.input_emb_dim * 4), nn.Tanh(),
                                     nn.Linear(config.input_emb_dim * 4, config.n_embd))
        print(diffusion)
        self.diffusion = diffusion
        if diffusion is not None:
            self.train_diff_steps = config.train_diff_steps
        else:
            self.train_diff_steps = 200
        self.time_embeddings = nn.Embedding(self.train_diff_steps, config.n_embd)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
        self.lm_head2 = nn.Linear(config.n_embd, config.vocab_size, bias=False)


    # def get_output_embeddings(self):
    #     return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
            self,
            input_ids=None,
            tgt_ids=None,
            input_embs=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            t = 0,
            t_aware=True,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(input_ids.shape, 'input_ids', )

        if input_embs is None: # training time!
            input_embs = self.transformer.wte(input_ids)  # input_embs
            tgt_embs = self.transformer.wte(tgt_ids)

        # print(self.diffusion is not None, self.train_diff_steps)
        if self.diffusion is not None: # training time
            if self.train_diff_steps > 0:
                # sample t
                t = torch.randint(0, self.train_diff_steps, (input_embs.shape[0],)).to(input_embs.device)
                t_mask = (t == 0)
                input_embs_rand = self.diffusion.q_sample(input_embs, t)
                posterior_mean, posterior_variance,\
                posterior_log_variance_clipped = self.diffusion.q_posterior_mean_variance( x_start=input_embs,
                                                                                           x_t=input_embs_rand, t=t)
                input_embs_rand_mid = posterior_mean + torch.sqrt(posterior_variance) * torch.randn_like(posterior_mean)
                # print(input_embs_rand_mid.shape, input_embs_rand.shape)

                input_embs_rand_mid[t_mask] = input_embs[t_mask]

                input_concat = torch.cat([input_embs_rand_mid, input_embs_rand, tgt_embs], dim=1)
                time_emb = self.time_embeddings(t).unsqueeze(1)
                input_embs = input_concat



        elif self.diffusion is None and t is not None: # test time, specifying t.
            # print(t, input_embs.shape)
            t = torch.LongTensor([t]).expand(input_embs.size(0)).to(self.device)
            time_emb = self.time_embeddings(t).unsqueeze(1)

        # print(input_embs.shape)
        input_embs = self.up_proj(input_embs)
        # print(input_embs.shape)
        if t_aware:
            input_embs = torch.cat([time_emb, input_embs], dim=1)

        transformer_outputs = self.transformer(
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=input_embs,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if t_aware:
            hidden_states = transformer_outputs[0][:, 1:, ]
        else:
            hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # print(hidden_states.shape, self.lm_head2, self.lm_head2.weight.shape)
        lm_logits = self.lm_head2(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )