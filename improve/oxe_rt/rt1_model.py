import copy
import enum
import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


conv_kernel_init_fn = nn.initializers.variance_scaling(2.0, "fan_out", "normal")

dense_kernel_init_fn = nn.initializers.variance_scaling(1 / 3.0, "fan_out", "uniform")


class FilmConditioning(nn.Module):
    """FiLM conditioning layer."""

    num_channels: int

    @nn.compact
    def __call__(self, conv_filters, context):
        """Applies FiLM conditioning to the input.

        Args:
          conv_filters: array of shape (B, H, W, C), usually an output conv feature
            map.
          context: array of shape (B, context_size).

        Returns:
          array of shape (B, H, W, C) with the FiLM conditioning applied.
        """
        zero_init = nn.initializers.zeros_init()
        project_cond_add = nn.Dense(
            self.num_channels, kernel_init=zero_init, bias_init=zero_init
        )(context)
        project_cond_mul = nn.Dense(
            self.num_channels, kernel_init=zero_init, bias_init=zero_init
        )(context)

        project_cond_add = project_cond_add[:, None, None, :]
        project_cond_mul = project_cond_mul[:, None, None, :]

        result = (1 + project_cond_mul) * conv_filters + project_cond_add
        return result


class DepthwiseConv(nn.Module):
    """Depthwise convolution that matches tensorflow's conventions.

    In Tensorflow, the shapes of depthwise kernels don't match the shapes of a
    regular convolutional kernel of appropriate feature_group_count.
    It is safer to use this class instead of the regular Conv (easier port of
    tensorflow checkpoints, fan_out initialization of the previous layer will
    match the tensorflow behavior, etc...).

    Attributes:
      features: Number of convolution filters.
      kernel_size: Shape of the convolutional kernel.
      strides: A sequence of `n` integers, representing the inter-window strides.
      padding: Either the string `'SAME'`, the string `'VALID'`, or a sequence of
        `n` `(low, high)` integer pairs that give the padding to apply before and
        after each spatial dimension.
      input_dilation: `None`, or a sequence of `n` integers, giving the dilation
        factor to apply in each spatial dimension of `inputs`. Convolution with
        input dilation `d` is equivalent to transposed convolution with stride
        `d`.
      kernel_dilation: `None`, or a sequence of `n` integers, giving the dilation
        factor to apply in each spatial dimension of the convolution kernel.
        Convolution with kernel dilation is also known as 'atrous convolution'.
      feature_group_count: Unused attribute present in nn.Conv. Declare it to
        match the nn.Conv API.
      use_bias: Whether to add a bias to the output (default: True).
      dtype: The dtype of the computation (default: float32).
      precision: Numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: Initializer for the convolutional kernel.
      bias_init: Initializer for the bias.
    """

    features: int
    kernel_size: Tuple[int, int]
    strides: Optional[Tuple[int, int]] = None
    padding: Union[str, Sequence[int]] = "SAME"
    input_dilation: Optional[Sequence[int]] = None
    kernel_dilation: Optional[Sequence[int]] = None
    feature_group_count: int = 1
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    precision: Any = None
    kernel_init: Any = nn.initializers.lecun_normal()
    bias_init: Any = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Applies a convolution to the inputs.

        Args:
          inputs: Input data with dimensions (batch, spatial_dims..., features).

        Returns:
          The convolved data.
        """
        inputs = jnp.asarray(inputs, self.dtype)
        in_features = inputs.shape[-1]
        strides = self.strides

        if strides is None:
            strides = (1,) * (inputs.ndim - 2)

        kernel_shape = self.kernel_size + (self.features, 1)
        # Naming convention follows tensorflow.
        kernel = self.param("depthwise_kernel", self.kernel_init, kernel_shape)
        kernel = jnp.asarray(kernel, self.dtype)

        # Need to transpose to convert tensorflow-shaped kernel to lax-shaped kernel
        kernel = jnp.transpose(kernel, [0, 1, 3, 2])

        dimension_numbers = nn.linear._conv_dimension_numbers(
            inputs.shape
        )  # pylint:disable=protected-access

        y = jax.lax.conv_general_dilated(
            inputs,
            kernel,
            strides,
            self.padding,
            lhs_dilation=self.input_dilation,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=in_features,
            precision=self.precision,
        )

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,))
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias

        return y


# pytype: disable=attribute-error
# pylint:disable=unused-argument
class BlockConfig(object):
    """Class that contains configuration parameters for a single block."""

    def __init__(
        self,
        input_filters: int = 0,
        output_filters: int = 0,
        kernel_size: int = 3,
        num_repeat: int = 1,
        expand_ratio: int = 1,
        strides: Tuple[int, int] = (1, 1),
        se_ratio: Optional[float] = None,
        id_skip: bool = True,
        fused_conv: bool = False,
        conv_type: str = "depthwise",
    ):
        for arg in locals().items():
            setattr(self, *arg)


class ModelConfig(object):
    """Class that contains configuration parameters for the model."""

    def __init__(
        self,
        width_coefficient: float = 1.0,
        depth_coefficient: float = 1.0,
        resolution: int = 224,
        dropout_rate: float = 0.2,
        blocks: Tuple[BlockConfig, ...] = (
            # (input_filters, output_filters, kernel_size, num_repeat,
            #  expand_ratio, strides, se_ratio)
            # pylint: disable=bad-whitespace
            BlockConfig(32, 16, 3, 1, 1, (1, 1), 0.25),
            BlockConfig(16, 24, 3, 2, 6, (2, 2), 0.25),
            BlockConfig(24, 40, 5, 2, 6, (2, 2), 0.25),
            BlockConfig(40, 80, 3, 3, 6, (2, 2), 0.25),
            BlockConfig(80, 112, 5, 3, 6, (1, 1), 0.25),
            BlockConfig(112, 192, 5, 4, 6, (2, 2), 0.25),
            BlockConfig(192, 320, 3, 1, 6, (1, 1), 0.25),
            # pylint: enable=bad-whitespace
        ),
        stem_base_filters: int = 32,
        top_base_filters: int = 1280,
        activation: str = "swish",
        batch_norm: str = "default",
        bn_momentum: float = 0.99,
        bn_epsilon: float = 1e-3,
        # While the original implementation used a weight decay of 1e-5,
        # tf.nn.l2_loss divides it by 2, so we halve this to compensate in Keras
        weight_decay: float = 5e-6,
        drop_connect_rate: float = 0.2,
        depth_divisor: int = 8,
        min_depth: Optional[int] = None,
        use_se: bool = True,
        input_channels: int = 3,
        num_classes: int = 1000,
        model_name: str = "efficientnet",
        rescale_input: bool = True,
        data_format: str = "channels_last",
        final_projection_size: int = 0,
        classifier_head: bool = True,
        dtype: jnp.dtype = jnp.float32,
    ):
        """Default Config for Efficientnet-B0."""
        for arg in locals().items():
            setattr(self, *arg)


# pylint:enable=unused-argument


EN_MODEL_CONFIGS = {
    # (width, depth, resolution, dropout)
    "efficientnet-b3": ModelConfig(1.2, 1.4, 300, 0.3),
}


def round_filters(filters: int, config: ModelConfig) -> int:
    """Returns rounded number of filters based on width coefficient."""
    width_coefficient = config.width_coefficient
    min_depth = config.min_depth
    divisor = config.depth_divisor

    if not width_coefficient:
        return filters

    filters *= width_coefficient
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats: int, depth_coefficient: float) -> int:
    """Returns rounded number of repeats based on depth coefficient."""
    return int(math.ceil(depth_coefficient * repeats))


def conv2d(
    inputs: jnp.ndarray,
    num_filters: int,
    config: ModelConfig,
    kernel_size: Union[int, Tuple[int, int]] = (1, 1),
    strides: Tuple[int, int] = (1, 1),
    use_batch_norm: bool = True,
    use_bias: bool = False,
    activation: Any = None,
    depthwise: bool = False,
    train: bool = True,
    conv_name: Optional[str] = None,
    bn_name: Optional[str] = None,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Convolutional layer with possibly batch norm and activation.

    Args:
      inputs: Input data with dimensions (batch, spatial_dims..., features).
      num_filters: Number of convolution filters.
      config: Configuration for the model.
      kernel_size: Size of the kernel, as a tuple of int.
      strides: Strides for the convolution, as a tuple of int.
      use_batch_norm: Whether batch norm should be applied to the output.
      use_bias: Whether we should add bias to the output of the first convolution.
      activation: Name of the activation function to use.
      depthwise: If true, will use depthwise convolutions.
      train: Whether the model should behave in training or inference mode.
      conv_name: Name to give to the convolution layer.
      bn_name: Name to give to the batch norm layer.
      dtype: dtype for the computation.

    Returns:
      The output of the convolutional layer.
    """
    conv_fn = DepthwiseConv if depthwise else nn.Conv
    kernel_size = (
        (kernel_size, kernel_size)
        if isinstance(kernel_size, int)
        else tuple(kernel_size)
    )
    conv_name = conv_name if conv_name else "conv2d"
    bn_name = bn_name if bn_name else "batch_normalization"

    x = conv_fn(
        num_filters,
        kernel_size,
        tuple(strides),
        padding="SAME",
        use_bias=use_bias,
        kernel_init=conv_kernel_init_fn,
        name=conv_name,
        dtype=dtype,
    )(inputs)

    if use_batch_norm:
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=config.bn_momentum,
            epsilon=config.bn_epsilon,
            name=bn_name,
            dtype=dtype,
        )(x)

    if activation is not None:
        x = getattr(nn.activation, activation.lower())(x)
    return x


def stochastic_depth(
    inputs: jnp.ndarray,
    rng: jnp.ndarray,
    survival_probability: float,
    deterministic: bool = False,
) -> jnp.ndarray:
    """Applies stochastic depth.

    Args:
      inputs: The inputs that should be randomly masked.
      rng: A `jax.random.PRNGKey`.
      survival_probability: 1 - the probability of masking out a value.
      deterministic: If false the inputs are scaled by `1 / (1 - rate)` and
        masked, whereas if true, no mask is applied and the inputs are returned as
        is.

    Returns:
      The masked inputs.
    """
    if survival_probability == 1.0 or deterministic:
        return inputs

    mask_shape = [inputs.shape[0]] + [1 for _ in inputs.shape[1:]]
    mask = jax.random.bernoulli(rng, p=survival_probability, shape=mask_shape)
    mask = jnp.tile(mask, [1] + list(inputs.shape[1:]))
    return jax.lax.select(mask, inputs / survival_probability, jnp.zeros_like(inputs))


class SqueezeExcite(nn.Module):
    """SqueezeExite block (See: https://arxiv.org/abs/1709.01507.)

    Attributes:
      num_filters: Number of convolution filters.
      block: Configuration for this block.
      config: Configuration for the model.
      train: Whether the model is in training or inference mode.
    """

    num_filters: int
    block: BlockConfig
    config: ModelConfig
    train: bool

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Applies a convolution to the inputs.

        Args:
          inputs: Input data with dimensions (batch, spatial_dims..., features).

        Returns:
          The output of the squeeze excite block.
        """
        block = self.block
        config = self.config
        train = self.train
        dtype = config.dtype
        num_reduced_filters = max(1, int(block.input_filters * block.se_ratio))

        se = nn.avg_pool(inputs, inputs.shape[1:3])
        se = conv2d(
            se,
            num_reduced_filters,
            config,
            use_bias=True,
            use_batch_norm=False,
            activation=config.activation,
            conv_name="reduce_conv2d_0",
            train=train,
            dtype=dtype,
        )

        se = conv2d(
            se,
            self.num_filters,
            config,
            use_bias=True,
            use_batch_norm=False,
            activation="sigmoid",
            conv_name="expand_conv2d_0",
            train=train,
            dtype=dtype,
        )

        return inputs * se


class MBConvBlock(nn.Module):
    """Main building component of Efficientnet.

    Attributes:
      block: BlockConfig, arguments to create a Block.
      config: ModelConfig, a set of model parameters.
      train: Whether we are training or predicting.
    """

    block: BlockConfig
    config: ModelConfig
    train: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Mobile Inverted Residual Bottleneck.

        Args:
          inputs: Input to the block.

        Returns:
          The output of the block.
        """
        config = self.config
        block = self.block
        train = self.train
        use_se = config.use_se
        activation = config.activation
        drop_connect_rate = config.drop_connect_rate
        use_depthwise = block.conv_type != "no_depthwise"
        dtype = config.dtype

        rng = self.make_rng("random")

        filters = block.input_filters * block.expand_ratio

        x = inputs
        bn_index = 0

        if block.fused_conv:
            # If we use fused mbconv, skip expansion and use regular conv.
            x = conv2d(
                x,
                filters,
                config,
                kernel_size=block.kernel_size,
                strides=block.strides,
                activation=activation,
                conv_name="fused_conv2d_0",
                bn_name="batch_normalization_" + str(bn_index),
                train=train,
                dtype=dtype,
            )
            bn_index += 1
        else:
            if block.expand_ratio != 1:
                # Expansion phase
                kernel_size = (1, 1) if use_depthwise else (3, 3)
                x = conv2d(
                    x,
                    filters,
                    config,
                    kernel_size=kernel_size,
                    activation=activation,
                    conv_name="expand_conv2d_0",
                    bn_name="batch_normalization_" + str(bn_index),
                    train=train,
                    dtype=dtype,
                )
                bn_index += 1
            # Depthwise Convolution
            if use_depthwise:
                x = conv2d(
                    x,
                    num_filters=x.shape[-1],  # Depthwise conv
                    config=config,
                    kernel_size=block.kernel_size,
                    strides=block.strides,
                    activation=activation,
                    depthwise=True,
                    conv_name="depthwise_conv2d",
                    bn_name="batch_normalization_" + str(bn_index),
                    train=train,
                    dtype=dtype,
                )
                bn_index += 1

        # Squeeze and Excitation phase
        if use_se:
            assert block.se_ratio is not None
            assert 0 < block.se_ratio <= 1
            x = SqueezeExcite(
                num_filters=filters, block=block, config=config, train=train
            )(x)

        # Output phase
        x = conv2d(
            x,
            block.output_filters,
            config,
            activation=None,
            conv_name="project_conv2d_0",
            bn_name="batch_normalization_" + str(bn_index),
            train=train,
            dtype=dtype,
        )

        if (
            block.id_skip
            and all(s == 1 for s in block.strides)
            and block.input_filters == block.output_filters
        ):
            if drop_connect_rate and drop_connect_rate > 0:
                survival_probability = 1 - drop_connect_rate
                x = stochastic_depth(
                    x, rng, survival_probability, deterministic=not train
                )
            x = x + inputs

        return x


class Stem(nn.Module):
    """Initial block of Efficientnet.

    Attributes:
      config: ModelConfig, a set of model parameters.
      train: Whether we are training or predicting.
    """

    config: ModelConfig
    train: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Returns the output of the stem block.

        Args:
          inputs: The input to the block.

        Returns:
          Output of the block
        """
        config = self.config
        train = self.train
        x = conv2d(
            inputs,
            round_filters(config.stem_base_filters, config),
            config,
            kernel_size=(3, 3),
            strides=(2, 2),
            activation=config.activation,
            train=train,
            dtype=config.dtype,
        )
        return x


class Head(nn.Module):
    """Final block of Efficientnet.

    Attributes:
      config: A set of model parameters.
      train: Whether we are training or predicting.
    """

    config: Any
    train: bool = True

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Returns the output of the head block.

        Args:
          inputs: The input to the block.

        Returns:
          x: Classifier logits.
        """
        config = self.config
        train = self.train
        dtype = config.dtype
        # Build top.
        x = conv2d(
            inputs,
            round_filters(config.top_base_filters, config),
            config,
            activation=config.activation,
            train=train,
            dtype=dtype,
        )
        return x


# pytype: enable=attribute-error


class EfficientNetWithFilm(nn.Module):
    """EfficientNet with FiLM conditioning."""

    config: Any
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, context_input: jnp.ndarray, *, train: bool):
        """Returns the output of the EfficientNet model."""
        config = copy.deepcopy(self.config)
        config.dtype = self.dtype
        depth_coefficient = config.depth_coefficient
        blocks = config.blocks
        drop_connect_rate = config.drop_connect_rate

        inputs = jnp.asarray(inputs, self.dtype)

        # Build stem.
        x = Stem(config=config, train=train)(inputs)

        # Build blocks.
        num_blocks_total = sum(
            round_repeats(block.num_repeat, depth_coefficient) for block in blocks
        )
        block_num = 0

        for _, block in enumerate(blocks):
            assert block.num_repeat > 0
            # Update block input and output filters based on depth multiplier.
            block.input_filters = round_filters(block.input_filters, config)
            block.output_filters = round_filters(block.output_filters, config)
            block.num_repeat = round_repeats(block.num_repeat, depth_coefficient)

            # The first block needs to take care of stride and filter size increase
            drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
            config.drop_connect_rate = drop_rate

            x = MBConvBlock(block=block, config=config, train=train)(x)

            x = FilmConditioning(num_channels=x.shape[-1])(x, context_input)

            block_num += 1
            if block.num_repeat > 1:
                block.input_filters = block.output_filters
                block.strides = [1, 1]

                for _ in range(block.num_repeat - 1):
                    drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                    config.drop_connect_rate = drop_rate
                    x = MBConvBlock(block=block, config=config, train=train)(x)
                    x = FilmConditioning(num_channels=x.shape[-1])(x, context_input)

                    block_num += 1

        x = Head(self.config, train=train)(x)

        return x


class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    use_bias: bool = True
    kernel_init: Initializer = nn.initializers.xavier_uniform()
    bias_init: Initializer = nn.initializers.normal(stddev=1e-6)
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    precision: Optional[jax.lax.Precision] = None
    dtype: jnp.ndarray = jnp.float32

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, *, deterministic: bool):
        """Applies Transformer MlpBlock module."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            self.mlp_dim,
            dtype=self.dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            precision=self.precision,
        )(inputs)
        x = IdentityLayer(name="mlp1")(self.activation_fn(x))
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(
            actual_out_dim,
            dtype=self.dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            precision=self.precision,
        )(x)
        output = IdentityLayer(name="mlp2")(output)
        output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)
        return output


class TokenLearnerModuleV11(nn.Module):
    """TokenLearner module Version 1.1, using slightly different conv. layers.

    Instead of using 4 conv. layers with small channels to implement spatial
    attention, this version uses a MLP with gelu inbetween. It also uses softmax
    instead of sigmoid. We confirmed that this version works better in general.

    Attributes:
      num_tokens: Number of tokens.
      bottleneck_dim: The size of hidden units in the MLP for spatial attention.
      dropout_rate: Dropout rate.
    """

    num_tokens: int
    bottleneck_dim: int = 64
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        """Applies learnable tokenization to the 2D inputs.

        Args:
          inputs: Inputs of shape `[bs, h, w, c]`.
          deterministic: Weather we are in the deterministic mode (e.g inference
            time) or not.

        Returns:
          Output of shape `[bs, n_token, c]`.
        """
        if inputs.ndim == 4:
            n, h, w, c = inputs.shape
            inputs = jnp.reshape(inputs, [n, h * w, c])

        feature_shape = inputs.shape

        selected = inputs

        selected = nn.LayerNorm()(selected)

        selected = MlpBlock(
            mlp_dim=self.bottleneck_dim,
            out_dim=self.num_tokens,
            dropout_rate=self.dropout_rate,
            activation_fn=nn.gelu,
            name="token_masking",
        )(selected, deterministic=deterministic)

        selected = jnp.reshape(
            selected, [feature_shape[0], -1, self.num_tokens]
        )  # Shape: [bs, h*w, n_token].
        selected = jnp.transpose(selected, [0, 2, 1])  # Shape: [bs, n_token, h*w].
        selected = jax.nn.softmax(selected, axis=-1)

        feat = inputs
        feat = jnp.reshape(
            feat, [feature_shape[0], -1, feature_shape[-1]]
        )  # Shape: [bs, h*w, c].

        feat = jnp.einsum("...si,...id->...sd", selected, feat)

        return feat


class FFNOptions(enum.Enum):
    """Different choices of FFN block for ablation testing."""

    LINEAR = "linear"  # RT-1 Legacy
    SWIGLU = "swiglu"  # Match LLaMa


class TransformerBlock(nn.Module):
    """A self-attention transformer block.

    See the `_TransformerLayer` in
    google-research/robotics_transformer/transformer.py for the original
    tensorflow implementation.
    """

    layer_size: int = 128
    num_heads: int = 8
    feed_forward_hidden_size: int = 512
    feed_forward_output_size: int = 512
    ffn_option: FFNOptions = FFNOptions.SWIGLU
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, attn_mask: jnp.ndarray, *, train: bool):
        x1 = nn.LayerNorm()(x)

        x1 = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=(self.layer_size * self.num_heads),
            dropout_rate=self.dropout_rate,
        )(x1, x1, mask=attn_mask, deterministic=not train)

        x = x + x1

        y = nn.LayerNorm()(x)

        if self.ffn_option == FFNOptions.SWIGLU:
            h1 = nn.Dense(self.feed_forward_hidden_size, use_bias=False)(y)
            h1 = nn.swish(h1)
            gate = nn.Dense(self.feed_forward_hidden_size, use_bias=False)(y)
            ff_y = nn.Dense(self.feed_forward_output_size, use_bias=False)(h1 * gate)
        elif self.ffn_option == FFNOptions.LINEAR:
            ff_y = nn.Dense(self.feed_forward_output_size, use_bias=False)(y)
        else:
            raise ValueError(f"Unknown FFN option: {self.ffn_option}")

        ff_y = nn.Dropout(self.dropout_rate)(ff_y, deterministic=not train)
        x = x + ff_y
        return x


class Transformer(nn.Module):
    """Transformer architecture with dense positional embedding.

    See the `Transformer` in
    google-research/robotics_transformer/transformer.py for the original
    tensorflow implementation.
    """

    num_layers: int = 8
    layer_size: int = 128
    num_heads: int = 8
    feed_forward_hidden_size: int = 512
    feed_forward_output_size: int = 512
    ffn_option: FFNOptions = FFNOptions.SWIGLU
    dropout_rate: float = 0.1
    vocab_size: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray, attn_mask: jnp.ndarray, *, train: bool):
        bs, seqlen, *_ = x.shape

        pos = jnp.expand_dims(jnp.arange(0, seqlen, 1), 0)
        pos = jnp.tile(pos, [bs, 1])
        pos = jax.nn.one_hot(pos, seqlen)

        x = nn.Dense(self.feed_forward_output_size)(x)
        pos_emb = nn.Dense(self.feed_forward_output_size)(pos)
        x += pos_emb

        for _ in range(self.num_layers):
            x = TransformerBlock(
                layer_size=self.layer_size,
                num_heads=self.num_heads,
                feed_forward_hidden_size=self.feed_forward_hidden_size,
                feed_forward_output_size=self.feed_forward_output_size,
                dropout_rate=self.dropout_rate,
                ffn_option=self.ffn_option,
            )(x, attn_mask, train=train)

        output_tokens = nn.Dense(self.vocab_size)(x)
        return output_tokens


class TokenLearnerModuleV11(nn.Module):
    """TokenLearner module Version 1.1, using slightly different conv. layers.

    Instead of using 4 conv. layers with small channels to implement spatial
    attention, this version uses a MLP with gelu inbetween. It also uses softmax
    instead of sigmoid. We confirmed that this version works better in general.

    From google-research/scenic/projects/token_learner/model.py.

    Attributes:
      num_tokens: Number of tokens.
      bottleneck_dim: The size of hidden units in the MLP for spatial attention.
      dropout_rate: Dropout rate.
    """

    num_tokens: int
    bottleneck_dim: int = 64
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        """Applies learnable tokenization to the 2D inputs.

        Args:
          inputs: Inputs of shape `[bs, h, w, c]`.
          deterministic: Weather we are in the deterministic mode (e.g inference
            time) or not.

        Returns:
          Output of shape `[bs, n_token, c]`.
        """
        if inputs.ndim == 4:
            n, h, w, c = inputs.shape
            inputs = jnp.reshape(inputs, [n, h * w, c])

        feature_shape = inputs.shape

        selected = inputs

        selected = nn.LayerNorm()(selected)

        selected = MlpBlock(
            mlp_dim=self.bottleneck_dim,
            out_dim=self.num_tokens,
            dropout_rate=self.dropout_rate,
            activation_fn=nn.gelu,
            name="token_masking",
        )(selected, deterministic=deterministic)

        selected = jnp.reshape(
            selected, [feature_shape[0], -1, self.num_tokens]
        )  # Shape: [bs, h*w, n_token].
        selected = jnp.transpose(selected, [0, 2, 1])  # Shape: [bs, n_token, h*w].
        selected = jax.nn.softmax(selected, axis=-1)

        feat = inputs
        feat = jnp.reshape(
            feat, [feature_shape[0], -1, feature_shape[-1]]
        )  # Shape: [bs, h*w, c].

        feat = jnp.einsum("...si,...id->...sd", selected, feat)

        return feat


class ImageTokenizer(nn.Module):
    """Tokenizes images with EfficientNet+FiLM.

    This is based on the `RT1ImageTokenizer` implementation here:
    google-research/robotics_transformer/tokenizers/image_tokenizer.py

    The overall flow of the image tokenizer:
    * The input image batch dimensions are squashed, and the image is normalized.
    * The image is fed through the `EfficientNetWithFilm`.
    * A 1x1 convolution is applied to project to `num_features`.
    * Another final `FilmConditioning` layer is applied with the context.
    * `TokenLearnerModuleV11` is applied to project the tokens to `num_tokens`.
    """

    num_tokens: int = 8
    num_features: int = 512

    use_token_learner: bool = True

    @nn.compact
    def __call__(self, image: jnp.ndarray, context_input: jnp.ndarray, *, train: bool):
        """Tokenizes the image using an EfficientNet.

        Args:
          image: jnp.Array with batch and seqlen leading dimensions. We assume the
            input image is of size 300x300, since the EfficientNet takes in images
            of that size.
          context_input: jnp.Array with shape (batch * seqlen, size).
          train: Training mode.

        Returns:
          shape (batch, seqlen, num_tokens, num_features) array.
        """
        bs, seqlen, *_ = image.shape

        # The efficientnet-b3 model uses 300x300 images.
        efficientnet_config = EN_MODEL_CONFIGS["efficientnet-b3"]
        image = jnp.reshape(image, [bs * seqlen, 300, 300, 3])
        image -= jnp.array(MEAN_RGB)
        image /= jnp.array(STDDEV_RGB)

        # Apply film in EfficientNet.
        x = EfficientNetWithFilm(efficientnet_config)(
            image, context_input=context_input, train=train
        )

        # 1x1 conv. This corresponds to the 1x1 conv here:
        # google-research/robotics_transformer/film_efficientnet/pretrained_efficientnet_encoder.py
        var_init = nn.initializers.variance_scaling(
            scale=1.0,
            mode="fan_in",
            distribution="truncated_normal",
        )
        x = nn.Conv(
            features=self.num_features,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            kernel_init=var_init,
        )(x)

        x = FilmConditioning(num_channels=self.num_features)(x, context_input)

        if self.use_token_learner:
            x = TokenLearnerModuleV11(num_tokens=self.num_tokens)(
                x, deterministic=not train
            )

        x = jnp.reshape(x, [bs, seqlen, self.num_tokens, -1])

        return x


def tokenize_action(
    actions: Dict[str, jnp.ndarray],
    vocab_size: int,
    world_vector_range: Tuple[float, float] = (-1.0, 1.0),
) -> jnp.ndarray:
    """Tokenizes the action for the RT-1 task.

    :
    terminate_episode: (3,) int32,
      mode 0: terminate episode
      mode 1: arm + gripper

      mode 2: base
    world_vector: (3,) [-1.0, 1.0] (RT-1) or [-2.0, 2.0] (RT-1-X)
    rotation_delta: (3,) [-np.pi, np.pi]
    gripper_closedness_action: (1,) [-1, 1]
    base_displacement_vertical_rotation: (1,) [-np.pi, np.pi]
    base_displacement_vector: (2,) [-1.0, 1.0]

    Args:
      actions: The raw action dictionary.
      vocab_size: The vocab size of the tokenized actions.
      world_vector_range: The bounds to use for the world_vector token.

    Returns:
      the tokenized action.
    """
    action_tokens = []

    # Handle the discrete one first.
    terminate_episode = actions["terminate_episode"]
    terminate_episode = jnp.argmax(terminate_episode, axis=-1)
    terminate_episode = jnp.expand_dims(terminate_episode, -1)
    terminate_episode = terminate_episode.astype(jnp.int32)
    action_tokens.append(terminate_episode)

    for act_name, act_min, act_max in [
        ("world_vector", world_vector_range[0], world_vector_range[1]),
        ("rotation_delta", -np.pi / 2, np.pi / 2),
        ("gripper_closedness_action", -1.0, 1.0),
        ("base_displacement_vertical_rotation", -np.pi, np.pi),
        ("base_displacement_vector", -1.0, 1.0),
    ]:
        act = actions[act_name]
        act = jnp.clip(act, act_min, act_max)
        act = (act - act_min) / (act_max - act_min)
        act = act * (vocab_size - 1)
        act = act.astype(jnp.int32)
        action_tokens.append(act)

    tokenized = jnp.concatenate(action_tokens, axis=-1)
    return tokenized


def detokenize_action(
    tokenized_actions: jnp.ndarray,
    vocab_size: int,
    world_vector_range: Tuple[float, float] = (-1.0, 1.0),
) -> Dict[str, jnp.ndarray]:
    """De-tokenizes the action for the RT-1 task.

    See `tokenize_action` for information on the action structure.

    Args:
      tokenized_actions: The tokenized action vector.
      vocab_size: The vocab size of the tokenized actions.
      world_vector_range: The bounds to use for the world_vector token.

    Returns:
      the detokenized action dictionary.
    """
    terminate_episode = tokenized_actions[:, 0]
    terminate_episode = jax.nn.one_hot(terminate_episode, 3)

    raw_actions = dict(
        world_vector=tokenized_actions[:, 1:4].astype(jnp.float32),
        rotation_delta=tokenized_actions[:, 4:7].astype(jnp.float32),
        gripper_closedness_action=tokenized_actions[:, 7:8].astype(jnp.float32),
        base_displacement_vertical_rotation=tokenized_actions[:, 8:9].astype(
            jnp.float32
        ),
        base_displacement_vector=tokenized_actions[:, 9:11].astype(jnp.float32),
    )

    act_dict = {"terminate_episode": terminate_episode.astype(jnp.int32)}
    for act_name, act_min, act_max in [
        ("world_vector", world_vector_range[0], world_vector_range[1]),
        ("rotation_delta", -np.pi / 2, np.pi / 2),
        ("gripper_closedness_action", -1.0, 1.0),
        ("base_displacement_vertical_rotation", -np.pi, np.pi),
        ("base_displacement_vector", -1.0, 1.0),
    ]:
        act = raw_actions[act_name]
        act = act / (vocab_size - 1)
        act = act * (act_max - act_min)
        act = act + act_min
        act_dict[act_name] = act

    return act_dict


class RT1(nn.Module):
    """Full RT-1 and RT-1-X architecture."""

    num_layers: int = 8
    layer_size: int = 128
    num_heads: int = 8
    feed_forward_hidden_size: int = 512
    feed_forward_output_size: int = 512
    ffn_option: FFNOptions = FFNOptions.SWIGLU
    dropout_rate: float = 0.1
    vocab_size: int = 256
    num_image_tokens: int = 8
    num_action_tokens: int = 11
    image_num_features: int = 512

    world_vector_range: Tuple[float, float] = (-1.0, 1.0)

    use_token_learner: bool = True

    # By default, mask out previous actions.
    include_prev_timesteps_actions: bool = False

    sow_intermediates: bool = False

    def setup(self):
        self.image_tokenizer = ImageTokenizer(
            num_tokens=self.num_image_tokens,
            num_features=self.image_num_features,
            use_token_learner=self.use_token_learner,
        )

    def tokenize_image(self, image: jnp.ndarray, context: jnp.ndarray, *, train: bool):
        bs, seqlen, *_ = image.shape
        context = jnp.reshape(context, [bs * seqlen, -1])
        return self.image_tokenizer(image, context_input=context, train=train)

    @nn.compact
    def __call__(
        self,
        obs: Dict[str, jnp.ndarray],
        act: Dict[str, jnp.ndarray],
        obs_tokens: Optional[jnp.ndarray] = None,
        act_tokens: Optional[jnp.ndarray] = None,
        *,
        train: bool,
    ):
        bs = obs["image"].shape[0]
        seqlen = obs["image"].shape[1]

        # Depending on whether `obs_tokens` is passed, we either run the full
        # sequence of images through the image tokenizer, or simply use the
        # image tokens passed into this function. `obs_tokens` is usually passed
        # during an inference call when caching tokens from previous elements of
        # the input sequence.
        if obs_tokens is None:
            # Get image + language fused tokens.
            image = obs["image"]
            lang = obs["natural_language_embedding"]
            lang = jnp.reshape(lang, [bs * seqlen, -1])
            context_image_tokens = self.image_tokenizer(
                image=image, context_input=lang, train=train
            )
        else:
            context_image_tokens = obs_tokens

        if self.sow_intermediates:
            self.sow("intermediates", "image_tokens", context_image_tokens)

        # We either tokenize the action ourselves using `tokenize_action_fn` or
        # use the tokens passed into this function. `act_tokens` is usually supplied
        # during an inference call when caching tokens from previous actions.
        if act_tokens is None:
            action_tokens = tokenize_action(
                act, self.vocab_size, self.world_vector_range
            )  # pylint: disable=too-many-function-args
        else:
            action_tokens = act_tokens

        if self.include_prev_timesteps_actions:
            # Always zero out the final action tokens.
            previous_action_tokens = action_tokens[:, : (seqlen - 1), :]
            zero_action_tokens = jnp.zeros((bs, 1, self.num_action_tokens))
            action_tokens = jnp.concatenate(
                [previous_action_tokens, zero_action_tokens], axis=-2
            )

            # Project the actions to the token dimension.
            action_tokens = jax.nn.one_hot(action_tokens, num_classes=self.vocab_size)
            action_tokens = nn.Dense(self.image_num_features)(action_tokens)
        else:
            # If we're not including the previous actions, then we can zero out
            # the action tokens. We do it here to ensure tokens are consistently
            # zero regardless of the input actions passed to the function.
            action_tokens = jnp.zeros(
                (bs, seqlen, self.num_action_tokens, self.image_num_features)
            )

        # Assemble the input tokens into a single sequence.
        full_tokens = jnp.concatenate([context_image_tokens, action_tokens], axis=-2)

        num_action_tokens = action_tokens.shape[-2]
        full_tokens = jnp.reshape(
            full_tokens,
            [bs, seqlen * (self.num_image_tokens + num_action_tokens), -1],
        )

        attn_mask = self._construct_attn_mask(
            seqlen * (self.num_image_tokens + self.num_action_tokens)
        )
        output_tokens = Transformer(
            num_layers=self.num_layers,
            layer_size=self.layer_size,
            num_heads=self.num_heads,
            feed_forward_hidden_size=self.feed_forward_hidden_size,
            feed_forward_output_size=self.feed_forward_output_size,
            dropout_rate=self.dropout_rate,
            vocab_size=self.vocab_size,
            ffn_option=self.ffn_option,
        )(full_tokens, attn_mask=attn_mask, train=train)

        return output_tokens

    def _get_action_index_for_token(self, k: int, num_tokens: int):
        """Returns action associated with the token at given position `k`.

        If k is not an action token then it returns -1.
        If k is part of the first action in the sequence then returns 0 etc.

        Based on `_get_action_index_for_token` here:
        google-research/robotics_transformer/transformer_network.py

        Args:
          k: an int that represents the position in the sequence.
          num_tokens: The total number of tokens in the sequence.
        Returns:
          The index of the action that this position belongs to, or if this
          position is part of an image token then returns -1.
        """
        if k < 0 or k >= num_tokens:
            return -1

        single_time_step_num_tokens = self.num_image_tokens + self.num_action_tokens
        n = k
        if n % single_time_step_num_tokens < self.num_image_tokens:
            return -1

        return int(n / single_time_step_num_tokens)

    def _construct_attn_mask(self, num_tokens: ...):
        """Generate mask for action prediction loss.

        This masks out all action tokens.

        Based on `_generate_masks` here:
        google-research/robotics_transformer/transformer_network.py

        Args:
          num_tokens: The number of tokens with which to construct the input mask.

        Returns:
          A (num_tokens, num_tokens) attention mask.
        """
        default_attn_mask = np.tril(np.ones((num_tokens, num_tokens), np.int32))
        action_mask = np.zeros(shape=(num_tokens, num_tokens), dtype=np.int32)

        for i in range(num_tokens):
            for j in range(num_tokens):
                action_i = self._get_action_index_for_token(i, num_tokens)
                action_j = self._get_action_index_for_token(j, num_tokens)
                mask = 0
                if action_i != -1 and action_j != -1:
                    # Ignore actions of previous steps.
                    if action_j < action_i:
                        mask = 1
                    # If we're not auto-regression, ignore action dimensions of current
                    # step.
                    if action_j == action_i and j <= i:
                        mask = 1
                # i not is an action, but j is an action token.
                # Hence, also mask j when predicting i, to prevent accidental
                # dependency between output and masked dimensions because the output
                # can still depend on the masked dimensions when predictions of the
                # transformer layers after the first layer depends on the masked
                # dimensions.
                elif action_j != -1:
                    if not self.include_prev_timesteps_actions and j < i:
                        mask = 1
                action_mask[i, j] = mask
        return default_attn_mask - action_mask
