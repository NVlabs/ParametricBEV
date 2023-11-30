from mmcv.cnn import ConvModule
from torch import nn

from mmdet.models import NECKS
from IPython import embed
from mmcv.runner import auto_fp16

from pdb import set_trace as st

class ResModule2D(nn.Module):

    def __init__(self, n_channels, norm_cfg=dict(type='BN2d'), groups=1):
        super().__init__()
        self.conv0 = ConvModule(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU', inplace=True))
        self.conv1 = ConvModule(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.activation = nn.ReLU(inplace=True)
    
    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C, N_x, N_y, N_z).

        Returns:
            torch.Tensor: 5d feature map.
        """
        identity = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = identity + x
        x = self.activation(x)
        return x


@NECKS.register_module()
class PDBEVNeck(nn.Module):
    """Neck for PDBev.
    """

    def __init__(self, 
                 in_channels,
                 out_channels,
                 num_layers=2,
                 norm_cfg=dict(type='BN2d'),
                 stride=2,
                 is_transpose=True):
        super().__init__()
        
        self.is_transpose = is_transpose
        for i in range(3):
            print('neck transpose: {}'.format(is_transpose))
        
        model = nn.ModuleList()
        model.append(ResModule2D(in_channels, norm_cfg))
        model.append(ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))
        
        for i in range(num_layers):
            model.append(ResModule2D(out_channels, norm_cfg))
            model.append(ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))
        
        self.model = nn.Sequential(*model)
    
    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C_in, N_x, N_y, N_z).

        Returns:
            list[torch.Tensor]: of shape (N, C_out, N_y, N_x).
        """
        N,C,X,Y = x.shape
        # x = x.permute(0,2,3,4,1).reshape(N,X,Y,-1).permute(0,3,1,2)
        # [N,X,Y,Z,C] -> [N,X,Y,Z*C] -> [N,Z*C,X,Y]
        x = self.model.forward(x)
        
        if self.is_transpose:
            # Anchor3DHead axis order is (y, x).
            return [x.transpose(-1, -2)]
        else:
            return [x]
