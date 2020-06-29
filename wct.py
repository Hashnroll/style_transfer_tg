import torch
import torch.nn as nn

class WCT(nn.Module):
  def __init__(self, alpha, encoders, decoders):
    super().__init__()
    self.alpha = alpha
    self.encoders = encoders
    self.decoders = decoders

  def stylize(self, feat_c, feat_s):
    def whiten_and_color(feat_c, feat_s):
      c_size = feat_c.size()
      c_mean = torch.mean(feat_c, 1)
      c_mean = c_mean.unsqueeze(1).expand(c_size)
      feat_c = feat_c - c_mean

      content_conv = (feat_c @ feat_c.t()).div(c_size[1] -  1) + \
                      torch.eye(c_size[0]).to('cuda')
      _ , c_e, c_v = torch.svd(content_conv, some=False)

      k_c = c_size[0]
      for i in range(c_size[0]):
          if c_e[i] < 0.00001:
              k_c = i
              break

      s_size = feat_s.size()
      s_mean = torch.mean(feat_s, 1)
      feat_s = feat_s - s_mean.unsqueeze(1).expand(s_size)
      style_conv = (feat_s @ feat_s.t()).div(s_size[1] - 1)
      _, s_e, s_v = torch.svd(style_conv, some=False)

      k_s = s_size[0]
      for i in range(s_size[0]):
        if s_e[i] < 0.00001:
          k_s = i
          break

      c_d = 1 / torch.sqrt(c_e[0:k_c])
      step1 = c_v[:,0:k_c] @ torch.diag(c_d)
      step2 = step1 @ c_v[:,0:k_c].t()
      whiten_feat_c = step2 @ feat_c

      s_d = torch.sqrt(s_e[0:k_s])
      target = s_v[:,0:k_s] @ torch.diag(s_d) @ s_v[:,0:k_s].t() @ whiten_feat_c
      target = target + s_mean.unsqueeze(1).expand_as(target)
      return target

    C = feat_c.size(0)

    target = whiten_and_color(feat_c.view(C, -1), feat_s.view(C, -1))

    target = target.view_as(feat_c)
    result = self.alpha * target + (1.0 - self.alpha) * feat_c
    return result.unsqueeze(0)

  def forward(self, img_content, img_style):
    with torch.no_grad():
      for enc, dec in zip(self.encoders, self.decoders):
        content = enc(img_content).squeeze(0)
        style = enc(img_style).squeeze(0)
        content_style = self.stylize(content.squeeze(0), style.squeeze(0))
        img_content = dec(content_style)
    return img_content.squeeze(0)