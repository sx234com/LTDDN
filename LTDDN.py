import torch.nn as nn
import torch
import math
from torch.nn import init
import re
import collections

__all__ = ['LTDDN']

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'width_divisor', 'min_width', 'drop_connect_rate', 'image_size'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def ltddn_params(model_name):
    """Map LTDDN model name to parameter coefficients."""
    params_dict = {
        # (widthi_coefficient, depth_coefficient, image_size, dropout_rate)
        'LTDDN': (1.0, 1.0, 224, 0.2),
    }
    return params_dict[model_name]


def ltddn(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2,
                 drop_connect_rate=0.2, image_size=None, num_classes=1000):
    """Creates an ltddn model."""
    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        width_divisor=8,
        min_width=None,
        image_size=image_size,
    )
    return blocks_args, global_params


class BlockDecoder(object):
    """Block Decoder for readability."""

    @staticmethod
    def _decode_block_string(block_string):
        """Gets a block through a string notation of arguments."""
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=None,
            strides=[int(options['s'][0]), int(options['s'][1])])

    @staticmethod
    def decode(string_list):
        """Decodes a list of string notations to specify blocks inside the network.
        Args:
            string_list: a list of strings, each string is a notation of block.
        Returns:
            A list of namedtuples to represent blocks arguments.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args


def get_model_params(model_name, override_params=None):
    """Get the block args and global params for a given model."""
    if model_name.startswith('LTDDN'):
        width_coefficient, depth_coefficient, image_size, dropout_rate = (ltddn_params(model_name))
        blocks_args, global_params = ltddn(width_coefficient, depth_coefficient,
                                                  dropout_rate, image_size=image_size)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)

    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)

    return blocks_args, global_params


def round_filters(filters, global_params):
    """Calculate and round number of filters based on width multiplier."""
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.width_divisor
    min_width = global_params.min_width
    filters *= multiplier
    min_width = min_width or divisor
    new_filters = max(min_width, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(x, drop_connect_rate, training):
    if not training:
        return x
    keep_prob = 1.0 - drop_connect_rate
    batch_size = x.shape[0]
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
    binary_mask = torch.floor(random_tensor)
    x = (x / keep_prob) * binary_mask
    return x


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


#The feature channel selection submodule (FCSS)
class FCCSModule(nn.Module):

    def __init__(self, kernel_size=3, a=4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()
        self.a = a  # Select the proportion of the number of channels

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # Global Average Pooling
        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1

        # Calculates the weights of the channels
        channel_weights = y.squeeze(-1)  # bs,c,1 -> bs,c

        # Sort according to the weight of the channel
        sorted_weights, sorted_indices = torch.sort(channel_weights, dim=1, descending=True)

        # Get the channel index for the top 1/a
        num_channels = sorted_indices.size(1)
        num_select_channels = int(num_channels // self.a) 

        top_k_indices = sorted_indices[:, :num_select_channels]

        # Returns the selected channels
        selected_channels = x.gather(1, top_k_indices.unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3)))

        return selected_channels,num_select_channels

#FeatureReuseModule
class FeatureReuseModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=4, dw_size=3, stride=1, relu=True):
        super(FeatureReuseModule, self).__init__()
        self.oup = oup
        selected_num=int(inp // ratio)
        self.selected=selected_num

        remain_num=self.oup-selected_num

        if(selected_num>0):
            self.FCCS=FCCSModule(kernel_size=3,a=ratio)
            # FES
            self.tranditional_conv = nn.Sequential(
                nn.Conv2d(selected_num, selected_num, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(selected_num),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )

            # FRS
            self.feature_reuse = nn.Sequential(
                nn.Conv2d(selected_num, remain_num, dw_size, 1, dw_size // 2, groups=selected_num, bias=False),
                nn.BatchNorm2d(remain_num),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
        else:
            selected_num = math.ceil(oup / ratio)
            remain_num = selected_num * (ratio - 1)
            # FES
            self.tranditional_conv = nn.Sequential(
                nn.Conv2d(inp, selected_num, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(selected_num),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )

            # FRS
            self.feature_reuse = nn.Sequential(
                nn.Conv2d(selected_num, remain_num, dw_size, 1, dw_size // 2, groups=selected_num, bias=False),
                nn.BatchNorm2d(remain_num),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )


    def forward(self, x):
        if(self.selected>0):
            #FCSS
            x,selected_num=self.FCCS(x)

        x1 = self.tranditional_conv(x)
        x2 = self.feature_reuse(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class CFAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()

        # feature focus submodule (FFM)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

        # key feature extraction submodule (KFEM)
        self.mxpool = nn.AdaptiveMaxPool2d(1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # Feature Focus Submodule (FFM)
        y_ffm = self.gap(x)  # Global Average Pooling: bs,c,1,1
        y_ffm = y_ffm.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y_ffm = self.conv(y_ffm)  # bs,1,c
        y_ffm = self.sigmoid(y_ffm)  # bs,1,c
        y_ffm = y_ffm.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        output_ffm = x * y_ffm.expand_as(x)  # Apply the feature focus to the input

        # Key Feature Extraction Submodule (KFEM)
        y_kfem = self.mxpool(x)  # Global Max Pooling: bs,c,1,1
        output_kfem = x * y_kfem.expand_as(x)  # Apply the key feature extraction to the input

        # Fusion: Add both outputs together
        output = output_ffm + output_kfem  # Element-wise addition

        return output


# feature enhancement mobile inverted residual bottleneck convolution module (FEMB)
class FEMBCM(nn.Module):
    """
    Feature enhancement mobile inverted residual bottleneck convolution module
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    """
    def __init__(self, block_args, global_params,image_size=None):
        super(FEMBCM, self).__init__()

        self._block_args = block_args
        self._momentum = 1 - global_params.batch_norm_momentum
        self._epsilon = global_params.batch_norm_epsilon
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        #print("进入block的图片大小:"+str(image_size))

        # feature extranction branch (FEB)
        # Expansion convolution
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._momentum, eps=self._epsilon)

        # Depthwise convolution
        k = self._block_args.kernel_size
        s = self._block_args.strides
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, padding=(k - 1) // 2, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._momentum, eps=self._epsilon)

        self.cfa=CFAttention(kernel_size=3)

        # Point convolution
        final_oup = self._block_args.output_filters
        self._project_conv = nn.Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._momentum, eps=self._epsilon)
        self._relu = nn.ReLU6(inplace=True)

        #detail feature compensation branch (DCB)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2)
        self.sigmoid = nn.Sigmoid()
        self.maxpool_dcb = nn.AdaptiveMaxPool2d(int(image_size))


    def forward(self, x, drop_connect_rate=None):
        """
        :param x: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # feature extranction branch (FEB)
        # Expansion Convolution, Depthwise Convolution , CFA , Point Convolution
        identity = x
        if self._block_args.expand_ratio != 1:
            x = self._relu(self._bn0(self._expand_conv(x)))

        x_dw = self._relu(self._bn1(self._depthwise_conv(x)))
        x_dw = self.cfa(x_dw)

        # detail feature compensation branch (DCB)
        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        x1 = x * y.expand_as(x)
        x_dcb = self.maxpool_dcb(x1)

        x = torch.add(x_dw, x_dcb)  # 特征融合 Feature Fusion

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and all([stride == 1 for stride in self._block_args.strides]) and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, drop_connect_rate, training=self.training)
            x += identity  # skip connection
        return x


class LTDDN_Model(nn.Module):
    """
    An EfficientNet-lite model.
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks
    """
    def __init__(self, blocks_args=None, global_params=None):
        super(LTDDN_Model, self).__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        momentum = 1 - self._global_params.batch_norm_momentum
        epsilon = self._global_params.batch_norm_epsilon

        # Stem Feature Reuse Module Layer
        out_channels = 32
        self.stem = nn.Sequential(
            FeatureReuseModule(3, out_channels, kernel_size=3, ratio=8, dw_size=3, stride=2, relu=True),
            CFAttention(kernel_size=3),
            nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=epsilon),
            nn.ReLU6(inplace=True),
        )

        image_size=112

        # Build blocks
        self.blocks = nn.ModuleList([])
        for i, block_args in enumerate(self._blocks_args):
            # Update block input and output filters based on width multiplier.
            block_args = block_args._replace(
                input_filters=block_args.input_filters if i == 0 \
                        else round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=block_args.num_repeat if i == 0 or i == len(self._blocks_args) - 1 \
                        else round_repeats(block_args.num_repeat, self._global_params)
            )


            if block_args.strides[0] > 1:
                num_stride = block_args.strides[-1]
                #print(num_stride)
                image_size/=2

            # The first block needs to take care of stride and filter size increase.
            self.blocks.append(FEMBCM(block_args, self._global_params,image_size=image_size))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, strides=[1, 1])
            for _ in range(block_args.num_repeat - 1):
                self.blocks.append(FEMBCM(block_args, self._global_params,image_size=image_size))

        # Head Feature Reuse Module Layer
        in_channels = block_args.output_filters
        out_channels = 1280
        self.head = nn.Sequential(
            FeatureReuseModule(in_channels, out_channels, kernel_size=3, ratio=8, dw_size=3, stride=1, relu=True),
            CFAttention(kernel_size=3),
            nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=epsilon),
            nn.ReLU6(inplace=True),
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        if self._global_params.dropout_rate > 0:
            self.dropout = nn.Dropout(self._global_params.dropout_rate)
        else:
            self.dropout = None
        self.fc = torch.nn.Linear(out_channels, self._global_params.num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0/float(n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.stem(x)
        for idx, block in enumerate(self.blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.blocks)
            x = block(x, drop_connect_rate)
        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)

        return x


def LTDDN(override_params=None, **kwargs):
    model_name = 'LTDDN'
    blocks_args, global_params = get_model_params(model_name, override_params)
    model = LTDDN_Model(blocks_args, global_params, **kwargs)
    return model
