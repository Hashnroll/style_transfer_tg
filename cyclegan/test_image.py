import torch.utils.data.distributed

from cyclegan.cyclegan_pytorch.models import Generator

model = Generator().to('cuda')

model.load_state_dict(torch.load("cyclegan/weights/cezanne2photo/netG_A2B.pth"))

model.eval()

