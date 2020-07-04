import torch.utils.data.distributed

from cyclegan.cyclegan_pytorch.models import Generator

model = Generator().to('cuda')

model.load_state_dict(torch.load("cyclegan/weights/vangogh.pth"))

model.eval()

