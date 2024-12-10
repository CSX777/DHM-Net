import torch
import torch.nn as nn
from loss import batch_episym



class DCAA_Block(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(DCAA_Block, self).__init__()
        self.conv1=nn.Conv2d(inchannel, outchannel, (1, 1))
        self.fc_qkv=nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.BatchNorm2d(outchannel),
            nn.GELU(),

        )
        self.eb = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.GELU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )
        self.n_avg=nn.AdaptiveAvgPool2d((None,1))
        self.gelu=nn.GELU()
        self.sigmiod=nn.Sigmoid()
    def forward(self, x):
        feature=x
        feature_1 = self.conv1(feature)
        w_2  =self.fc_qkv(self.gelu(self.n_avg(feature_1)))
        confidence_scores = self.sigmiod(self.conv1(w_2))
        confidence_vector = torch.sum(w_2, dim=1, keepdim=True).sigmoid()
        confidence_scores_1=x*confidence_scores*confidence_vector
        out =  self.eb(confidence_scores_1)+x
        return self.gelu(out)

def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv

def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4 [32,1,500,4] logits[32,2,500,1]
    mask = logits[:, 0, :, 0] #[32,500] logits的第一层
    weights = logits[:, 1, :, 0] #[32,500] logits的第二层

    mask = torch.sigmoid(mask)
    weights = torch.exp(weights) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)

    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)

    # Recover essential matrix from self-adjoing eigen

    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat






class DHM_layer(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super(DHM_layer, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel
        self.HM = nn.Sequential(
                nn.Conv2d(self.in_channel*2 , self.in_channel, (1, 1)),
                nn.BatchNorm2d(self.in_channel),
                nn.GELU(),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
                nn.BatchNorm2d(self.in_channel),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((None, 1)),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 1))
            )

    def GET_h_index(self, input_x, k):
        xx_input = torch.sum(input_x ** 2, dim=1, keepdim=True)
        inner_x = -2 * torch.matmul(input_x.transpose(2, 1), input_x)
        pairwise_distance = -xx_input - inner_x - xx_input.transpose(2, 1)
        _, index = pairwise_distance.topk(k=k, dim=-1)
        return index[:, :, :]

    def heper_construt(self, x, k):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        # ---------------------------------------------------OBTAIN INDEX
        idx_out = self.GET_h_index(x, k=k)
        H_matrix = idx_out
        device = x.device
        # --------------------------------------------------
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = H_matrix + idx_base
        H_matrix_flatten = idx.view(-1)
        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()
        H_feature = x.view(batch_size * num_points, -1)[H_matrix_flatten, :]
        H_feature = H_feature.view(batch_size, num_points, k, num_dims)
        # -------------------------------------HTON
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        H_feature = torch.cat((x, H_feature * (x - H_feature)), dim=3).permute(0, 3, 1, 2).contiguous()
        # -------------------------------------
        return H_feature

    def forward(self, features):
        out = self.heper_construt(features, self.knn_num)
        out = self.HM(out)
        features=features*out.sigmoid()
        return features

class GCN_Block(nn.Module):
    def __init__(self, in_channel):
        super(GCN_Block, self).__init__()
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

    def attention(self, w):
        w = torch.relu(torch.tanh(w)).unsqueeze(-1)
        A = torch.bmm(w.transpose(1, 2), w)
        return A

    def graph_aggregation(self, x, w):
        B, _, N, _ = x.size()
        with torch.no_grad():
            A = self.attention(w)
            I = torch.eye(N).unsqueeze(0).to(x.device).detach()
            A = A + I
            D_out = torch.sum(A, dim=-1)
            D = (1 / D_out) ** 0.5
            D = torch.diag_embed(D)
            L = torch.bmm(D, A)
            L = torch.bmm(L, D)
        out = x.squeeze(-1).transpose(1, 2).contiguous()
        out = torch.bmm(L, out).unsqueeze(-1)
        out = out.transpose(1, 2).contiguous()
        return out

    def forward(self, x, w):
        out = self.graph_aggregation(x, w)
        out = self.conv(out)
        return out

class CF_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, k_num=8, sampling_rate=0.5):
        super(CF_Block, self).__init__()
        self.initial = initial
        self.in_channel = 4 if self.initial is True else 6
        self.out_channel = out_channel
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate

        self.mlp_conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        self.gcn = GCN_Block(self.out_channel)

        self.INIT_0 = nn.Sequential(
            DCAA_Block(self.out_channel, self.out_channel),
            DCAA_Block(self.out_channel, self.out_channel),
            DCAA_Block(self.out_channel, self.out_channel),
            DCAA_Block(self.out_channel, self.out_channel),
            DHM_layer(self.k_num, self.out_channel),
            DCAA_Block(self.out_channel, self.out_channel),
            DCAA_Block(self.out_channel, self.out_channel),
            DCAA_Block(self.out_channel, self.out_channel),
            DCAA_Block(self.out_channel, self.out_channel),
        )
        self.INIT_1 = nn.Sequential(
            DCAA_Block(self.out_channel, self.out_channel),
        )
        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))

        if self.predict == True:
            self.INIT_2= DCAA_Block(self.out_channel, self.out_channel)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        B, _, N , _ = x.size()
        indices = indices[:, :int(N*self.sr)]
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices)
            w_out = torch.gather(weights, dim=-1, index=indices)
        indices = indices.view(B, 1, -1, 1)

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1))
            return x_out, y_out, w_out, feature_out

    def forward(self, x, y):
        B, _, N , _ = x.size()
        out = x.transpose(1, 3).contiguous()
        #----------------------------Increase of dimension
        out = self.mlp_conv(out)
        #-----------------------DHM_FEATURE_INIT
        h_feature = self.INIT_0(out)
        w0 = self.linear_0(h_feature).view(B, -1)
        # --------------------------------UPDATE  GCN
        out_g = self.gcn(h_feature, w0.detach())
        out = out_g + h_feature
        #--------------------------
        out = self.INIT_1(out)
        #----------------------------CONFIDIENT
        w1 = self.linear_1(out).view(B, -1)

        if self.predict == False:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            indices_1 = indices
            w1_ds = w1_ds[:, :int(N*self.sr)]
            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices, None, self.predict)
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds],indices_1
        else:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            indices_2=indices
            w1_ds = w1_ds[:, :int(N*self.sr)]
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out, self.predict)

            out = self.INIT_2(out)
            w2 = self.linear_2(out)
            e_hat = weighted_8points(x_ds, w2)

            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat,indices_2

class DHMNet(nn.Module):
    def __init__(self, config):
        super(DHMNet, self).__init__()

        self.CF_0 = CF_Block(initial=True, predict=False, out_channel=128, k_num=9, sampling_rate=config.sr)  # set sampling_rate=0.5
        self.CF_1 = CF_Block(initial=False, predict=True, out_channel=128, k_num=6, sampling_rate=config.sr) # K1=9,K2=6

    def forward(self, x, y):
        B, _, N, _ = x.shape
        #------------THE FIRST CF
        x1, y1, ws0, w_ds0,indices_1 = self.CF_0(x, y)
        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1)
        # ------------THE SECOND CF
        x2, y2, ws1, w_ds1, e_hat,indices_2 = self.CF_1(x_, y1)
        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat)
        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat,indices_1,indices_2

