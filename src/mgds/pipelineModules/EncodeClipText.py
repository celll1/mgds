from contextlib import nullcontext

import torch
from torch import Tensor
from transformers import CLIPTextModel, CLIPTextModelWithProjection

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class EncodeClipText(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            in_name: str,
            tokens_attention_mask_in_name: str | None,
            hidden_state_out_name: str,
            pooled_out_name: str | None,
            text_encoder: CLIPTextModel | CLIPTextModelWithProjection,
            add_layer_norm: bool,
            hidden_state_output_index: int | None = None,
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
            chunk_length: int = 75,
            max_embeddings_multiples: int = 3,
    ):
        super(EncodeClipText, self).__init__()
        self.in_name = in_name
        self.tokens_attention_mask_in_name = tokens_attention_mask_in_name
        self.hidden_state_out_name = hidden_state_out_name
        self.pooled_out_name = pooled_out_name
        self.text_encoder = text_encoder
        self.add_layer_norm = add_layer_norm
        self.hidden_state_output_index = hidden_state_output_index
        self.chunk_length = chunk_length
        self.max_embeddings_multiples = max_embeddings_multiples

        self.autocast_contexts = [nullcontext()] if autocast_contexts is None else autocast_contexts
        self.dtype = dtype

    @staticmethod
    def encode_clip(
            text_encoder: CLIPTextModel | CLIPTextModelWithProjection,
            tokens: Tensor | None = None,
            default_layer: int = 0,
            layer_skip: int = 0,
            add_output: bool = True,
            text_encoder_output: Tensor | None = None,
            add_pooled_output: bool = False,
            pooled_text_encoder_output: Tensor | None = None,
            use_attention_mask: bool = True,
            attention_mask: Tensor | None = None,
            add_layer_norm: bool = True,
            chunk_length: int = 75,
            max_embeddings_multiples: int = 3,
    ) -> tuple[Tensor, Tensor]:
        if tokens is None or tokens.numel() == 0:
            return None, None

        original_device = tokens.device
        text_encoder_device = next(text_encoder.parameters()).device

        chunks = [tokens[:, i:i + chunk_length] for i in range(0, tokens.shape[1], chunk_length)]
        chunk_embeddings = []
        pooled_outputs = []

        for _, chunk in enumerate(chunks):
            if chunk.numel() == 0:
                continue

            chunk = chunk.to(text_encoder_device)

            bos_tokens = torch.full((chunk.shape[0], 1),
                                text_encoder.config.bos_token_id,
                                dtype=chunk.dtype,
                                device=text_encoder_device)
            eos_tokens = torch.full((chunk.shape[0], 1),
                                text_encoder.config.eos_token_id,
                                dtype=chunk.dtype,
                                device=text_encoder_device)

            chunk = torch.cat([bos_tokens, chunk, eos_tokens], dim=1)

            if chunk.shape[1] < chunk_length + 2:
                padding = torch.full(
                    (chunk.shape[0], chunk_length + 2 - chunk.shape[1]),
                    text_encoder.config.eos_token_id,
                    dtype=chunk.dtype,
                    device=text_encoder_device
                )
                chunk = torch.cat([chunk, padding], dim=1)

            outputs = text_encoder(
                chunk,
                attention_mask=attention_mask if use_attention_mask else None,
                return_dict=True,
                output_hidden_states=True,
            )

            if add_output:
                embedding = outputs.hidden_states[default_layer - layer_skip]
                if add_layer_norm:
                    final_layer_norm = text_encoder.text_model.final_layer_norm
                    embedding = final_layer_norm(embedding)
                embedding = embedding.to(original_device)
                chunk_embeddings.append(embedding)

            if add_pooled_output:
                if hasattr(outputs, "text_embeds"):
                    pooled = outputs.text_embeds.to(original_device)
                    pooled_outputs.append(pooled)
                elif hasattr(outputs, "pooler_output"):
                    pooled = outputs.pooler_output.to(original_device)
                    pooled_outputs.append(pooled)

        if add_output:
            if chunk_embeddings and len(chunk_embeddings) > max_embeddings_multiples:
                chunk_embeddings = chunk_embeddings[:max_embeddings_multiples]
            text_encoder_output = torch.cat(chunk_embeddings, dim=1)
        else:
            text_encoder_output = None

        if add_pooled_output:
            if pooled_outputs and len(pooled_outputs) > max_embeddings_multiples:
                pooled_outputs = pooled_outputs[:max_embeddings_multiples]
            pooled_text_encoder_output = pooled_outputs[0] if pooled_outputs else None
        else:
            pooled_text_encoder_output = None

        return text_encoder_output, pooled_text_encoder_output

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        if self.pooled_out_name:
            return [self.hidden_state_out_name, self.pooled_out_name]
        else:
            return [self.hidden_state_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        tokens = self._get_previous_item(variation, self.in_name, index)
        tokens = tokens.unsqueeze(0)

        if self.tokens_attention_mask_in_name is not None:
            tokens_attention_mask = self._get_previous_item(variation, self.tokens_attention_mask_in_name, index)
            tokens_attention_mask = tokens_attention_mask.unsqueeze(0)
        else:
            tokens_attention_mask = None

        with self._all_contexts(self.autocast_contexts):
            if tokens_attention_mask is not None and self.dtype:
                tokens_attention_mask = tokens_attention_mask.to(dtype=self.dtype)

            hidden_state, pooled_state = self.encode_clip(
                text_encoder=self.text_encoder,
                tokens=tokens,
                default_layer=self.hidden_state_output_index,
                add_output=True,
                add_pooled_output=self.pooled_out_name is not None,
                use_attention_mask=tokens_attention_mask is not None,
                attention_mask=tokens_attention_mask,
                add_layer_norm=self.add_layer_norm,
                chunk_length=self.chunk_length,
                max_embeddings_multiples=self.max_embeddings_multiples,
            )

        if hidden_state is not None:
            hidden_state = hidden_state.squeeze()
        if pooled_state is not None:
            pooled_state = pooled_state.squeeze()

        return {
            self.hidden_state_out_name: hidden_state,
            self.pooled_out_name: pooled_state,
        }
