import math
import torch
import torch.nn as nn



class FullSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16,select_strategy='mean',spectrum='reduction'):
        super(FullSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w
        self.select_strategy=select_strategy
        self.spectrum = spectrum


        self.dct_layer = FullSpectralDCTLayer(dct_h, dct_w, channel,strategy=self.select_strategy,spectrum=self.spectrum)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class FullSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, channel,strategy = 'mean',spectrum = 'reduction'):
        super(FullSpectralDCTLayer, self).__init__()
        self.lenth = height
        self.strategy = strategy
        self.spectrum = spectrum
        # fixed DCT init
        if self.spectrum == 'reduction':
            self.register_buffer('weight', self.get_dct_filter(self.lenth,height, width,  channel))
        elif self.spectrum =='full':
            self.register_buffer('weight', self.get_dct_filter(self.lenth*self.lenth, height, width, channel))
            # Full strategy takes a HUGE memory, the reduction strategy is recommended.


    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        length,channel,height,width=self.weight.shape
        batch_size = x.shape[0]
        full_freq_tensor=torch.ones(batch_size,channel,length).to('cuda')
        for l in range(length):
            temp_x = x*self.weight[l,:,:,:]
            result = torch.sum(temp_x, dim=[2, 3])
            full_freq_tensor[:,:,l] = result

        if self.strategy == 'mean':
            full_freq_tensor = torch.mean(full_freq_tensor,dim=2)
        elif self.strategy == 'max':
            full_freq_tensor,indexs = torch.max(full_freq_tensor,dim=2)
        else:
            raise ValueError("Wrong strategy")
        return full_freq_tensor

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, length,tile_size_x, tile_size_y, channel):
        dct_filter = torch.zeros(length,channel, tile_size_x, tile_size_y)

        if self.spectrum=='reduction':
            mapper_x = [i for i in range(length)]
            mapper_y = [i for i in range(length)]

        elif self.spectrum =='full':
            mapper_x = [i//tile_size_x for i in range(length)]
            mapper_y = [i % tile_size_x for i in range(length)]
        else:
            raise  ValueError("Wrong spectrum selection")
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):

            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i, :, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) \
                                                 * self.build_filter(t_y, v_y, tile_size_y)
        return dct_filter


if __name__ == "__main__":
    device= 'cuda'
    b,c,h,w=8,64,16,16
    input_tensor = torch.randn(b, c, h, w)
    model = FullSpectralAttentionLayer(c, h, w)
    input_tensor = input_tensor.to(device)
    model = model.to(device)
    out = model(input_tensor)
    print(out.shape)