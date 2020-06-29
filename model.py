import torch
import torch.nn as nn

enc1 = nn.Sequential(
	nn.Conv2d(3, 3, kernel_size=1),
	nn.ReflectionPad2d(1),
	nn.Conv2d(3, 64, kernel_size=3),
	nn.ReLU()
)
enc1.load_state_dict(torch.load("vgg19/vgg19_1/vgg_normalised_conv1_1.pth"))
dec1 = nn.Sequential(
	nn.ReflectionPad2d(1),
	nn.Conv2d(64, 3, kernel_size=3)
)
dec1.load_state_dict(torch.load("vgg19/vgg19_1/feature_invertor_conv1_1.pth"))
enc2 = nn.Sequential(
	nn.Conv2d(3, 3, kernel_size=1),
	nn.ReflectionPad2d(1),
	nn.Conv2d(3, 64, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(64, 64, kernel_size=3),
	nn.ReLU(),
	nn.MaxPool2d(2, 2, 0, ceil_mode=True),
	nn.ReflectionPad2d(1),
	nn.Conv2d(64, 128, kernel_size=3),
	nn.ReLU()
)
enc2.load_state_dict(torch.load("vgg19/vgg19_2/vgg_normalised_conv2_1.pth"))
dec2 = nn.Sequential(
	nn.ReflectionPad2d(1),
	nn.Conv2d(128, 64, kernel_size=3),
	nn.ReLU(),
	nn.Upsample(scale_factor=2),
	nn.ReflectionPad2d(1),
	nn.Conv2d(64, 64, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(64, 3, kernel_size=3)
)
dec2.load_state_dict(torch.load("vgg19/vgg19_2/feature_invertor_conv2_1.pth"))
enc3 = nn.Sequential(
	nn.Conv2d(3, 3, 1),
	nn.ReflectionPad2d(1),
	nn.Conv2d(3, 64, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(64, 64, kernel_size=3),
	nn.ReLU(),
	nn.MaxPool2d(2, 2, 0, ceil_mode=True),
	nn.ReflectionPad2d(1),
	nn.Conv2d(64,128, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(128,128, kernel_size=3),
	nn.ReLU(),
	nn.MaxPool2d(2, 2, 0, ceil_mode=True),
	nn.ReflectionPad2d(1),
	nn.Conv2d(128, 256, kernel_size=3),
	nn.ReLU()
)
enc3.load_state_dict(torch.load("vgg19/vgg19_3/vgg_normalised_conv3_1.pth"))
dec3 = nn.Sequential(
	nn.ReflectionPad2d(1),
	nn.Conv2d(256,128, kernel_size=3),
	nn.ReLU(),
	nn.Upsample(scale_factor=2),
	nn.ReflectionPad2d(1),
	nn.Conv2d(128,128, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(128, 64, kernel_size=3),
	nn.ReLU(),
	nn.Upsample(scale_factor=2),
	nn.ReflectionPad2d(1),
	nn.Conv2d(64, 64, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(64, 3, kernel_size=3)
)
dec3.load_state_dict(torch.load("vgg19/vgg19_3/feature_invertor_conv3_1.pth"))
enc4 = nn.Sequential(
	nn.Conv2d(3, 3,(1, 1)),
	nn.ReflectionPad2d(1),
	nn.Conv2d(3, 64, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(64, 64, kernel_size=3),
	nn.ReLU(),
	nn.MaxPool2d(2, 2, 0, ceil_mode=True),
	nn.ReflectionPad2d(1),
	nn.Conv2d(64,128, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(128,128, kernel_size=3),
	nn.ReLU(),
	nn.MaxPool2d(2, 2, 0, ceil_mode=True),
	nn.ReflectionPad2d(1),
	nn.Conv2d(128, 256, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(256, 256, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(256, 256, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(256, 256, kernel_size=3),
	nn.ReLU(),
	nn.MaxPool2d(2, 2, 0, ceil_mode=True),
	nn.ReflectionPad2d(1),
	nn.Conv2d(256, 512, kernel_size=3),
	nn.ReLU()
)
enc4.load_state_dict(torch.load("vgg19/vgg19_4/vgg_normalised_conv4_1.pth"))
dec4 = nn.Sequential(
	nn.ReflectionPad2d(1),
	nn.Conv2d(512, 256, kernel_size=3),
	nn.ReLU(),
	nn.Upsample(scale_factor=2),
	nn.ReflectionPad2d(1),
	nn.Conv2d(256, 256, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(256, 256, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(256, 256, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(256,128, kernel_size=3),
	nn.ReLU(),
	nn.Upsample(scale_factor=2),
	nn.ReflectionPad2d(1),
	nn.Conv2d(128,128, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(128, 64, kernel_size=3),
	nn.ReLU(),
	nn.Upsample(scale_factor=2),
	nn.ReflectionPad2d(1),
	nn.Conv2d(64, 64, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(64, 3, kernel_size=3)
)
dec4.load_state_dict(torch.load("vgg19/vgg19_4/feature_invertor_conv4_1.pth"))
enc5 = nn.Sequential(
	nn.Conv2d(3, 3, 1),
	nn.ReflectionPad2d(1),
	nn.Conv2d(3, 64, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(64, 64, kernel_size=3),
	nn.ReLU(),
	nn.MaxPool2d(2, 2, 0, ceil_mode=True),
	nn.ReflectionPad2d(1),
	nn.Conv2d(64,128, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(128,128, kernel_size=3),
	nn.ReLU(),
	nn.MaxPool2d(2, 2, 0, ceil_mode=True),
	nn.ReflectionPad2d(1),
	nn.Conv2d(128, 256, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(256, 256, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(256, 256, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(256, 256, kernel_size=3),
	nn.ReLU(),
	nn.MaxPool2d(2, 2, 0, ceil_mode=True),
	nn.ReflectionPad2d(1),
	nn.Conv2d(256, 512, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(512, 512, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(512, 512, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(512, 512, kernel_size=3),
	nn.ReLU(),
	nn.MaxPool2d(2, 2, 0, ceil_mode=True),
	nn.ReflectionPad2d(1),
	nn.Conv2d(512, 512, kernel_size=3),
	nn.ReLU()
)
enc5.load_state_dict(torch.load("vgg19/vgg19_5/vgg_normalised_conv5_1.pth"))
dec5 = nn.Sequential(
	nn.ReflectionPad2d(1),
	nn.Conv2d(512, 512, kernel_size=3),
	nn.ReLU(),
	nn.Upsample(scale_factor=2),
	nn.ReflectionPad2d(1),
	nn.Conv2d(512, 512, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(512, 512, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(512, 512, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(512, 256, kernel_size=3),
	nn.ReLU(),
  nn.Upsample(scale_factor=2),
	nn.ReflectionPad2d(1),
	nn.Conv2d(256, 256, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(256, 256, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(256, 256, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(256, 128, kernel_size=3),
	nn.ReLU(),
  nn.Upsample(scale_factor=2),
	nn.ReflectionPad2d(1),
	nn.Conv2d(128, 128, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(128, 64, kernel_size=3),
	nn.ReLU(),
  nn.Upsample(scale_factor=2),
	nn.ReflectionPad2d(1),
	nn.Conv2d(64, 64, kernel_size=3),
	nn.ReLU(),
	nn.ReflectionPad2d(1),
	nn.Conv2d(64, 3, kernel_size=3)
)
dec5.load_state_dict(torch.load("vgg19/vgg19_5/feature_invertor_conv5_1.pth"))

encoders = [enc5, enc4, enc3, enc2, enc1]
for enc in encoders:
  enc.to('cuda')
decoders = [dec5, dec4, dec3, dec2, dec1]
for dec in decoders:
  dec.to('cuda')