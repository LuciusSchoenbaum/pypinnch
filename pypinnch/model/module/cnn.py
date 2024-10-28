




import torch

from .module_impl import Module

from .module_impl.helper_modules import \
    ConvBlock_Basic, \
    ConvBlock_Init, \
    ConvBlock_Out



class CNN(Module):
    """

    ResNet (similar to ResNet18)
    due to Ekhi Ajuria, CERFACS, 07.09.2020

    Only input when called is number of data (input) channels.

    - Perform 4 levels of convolution
    - When returning to the original size, concatenate output of matching sizes
    - The smaller domains are upsampled to the desired size with the F.upsample function.

    Inputs are shape (batch, channels, height, width)
    Outputs are shape (batch,1, height, width)

    The number of input (data) channels is selected when the model is created.
    the number of output (target) channels is fixed at 1, although this could be changed in the future.

    The data can be any size (i.e. height and width).

    The model can be trained on data of a given size (H and W) and then used on data of any other size,
    although the best results so far have been obtained with test data of similar size to the training data

    Parameters:

        net:
            Instance of FNN network.
        dtype:
            data type specified by driver.

    """


    def __init__(
            self,
            net,
            dtype,
    ):
        super().__init__()
        # todo dtypes
        self.convN_1 = ConvBlock_Init(net.indim, 32)
        self.convN_2 = ConvBlock_Basic(32, 32)
        self.convN_3 = ConvBlock_Basic(32, 64)
        self.convN_4 = ConvBlock_Basic(64, 64)
        self.convN_5 = ConvBlock_Basic(64, 64)
        self.convN_6 = ConvBlock_Basic(64, 64)
        self.convN_7 = ConvBlock_Basic(64, 64)
        self.convN_8 = ConvBlock_Basic(64, 32)
        self.convN_9 = ConvBlock_Basic(64, 32)
        self.final = ConvBlock_Out(32, 1)


    def forward(self, inputs):
        convN_1out = self.convN_1(inputs)
        convN_2out = self.convN_2(convN_1out)
        convN_3out = self.convN_3(convN_2out+convN_1out)
        convN_4out = self.convN_4(
            convN_3out+torch.cat(
                (convN_2out, torch.zeros_like(convN_2out)),
                dim=1,
            )
        )
        convN_5out = self.convN_5(convN_4out+convN_3out)
        convN_6out = self.convN_6(convN_5out+convN_4out)
        #convN_6out = self.convN_6(convN_5out+torch.cat((convN_4out,torch.zeros_like(convN_4out)),dim=1))
        convN_7out = self.convN_7(convN_6out+convN_5out)
        convN_8out = self.convN_8(convN_7out+convN_6out)
        convN_9out = self.convN_9(
            torch.cat(
                (convN_8out, torch.zeros_like(convN_8out)),
                dim=1,
            ) + convN_7out
        )
        #convN_9out = self.convN_9(convN_8out+convN_7out)
        final_out = self.final(convN_9out)
        return final_out


