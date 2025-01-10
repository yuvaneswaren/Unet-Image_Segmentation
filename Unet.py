import torch
import torch.nn as nn




class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((2, 2))

        #encoder 1 64x64x3 ->  32x32x64 
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)

        #encoder 2 32x32x64 ->  16x16x128 
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)

        #encoder 3 16x16x128 ->  8x8x256 
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)

        #encoder 4 8x8x256 ->  4x4x512 
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)

        #encoder 5 4x4x512 ->  2x2x512
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)

        #encoder 6 2x2x512 ->  1x1x512
        self.convB_1 = nn.Conv2d(512, 512, kernel_size=2, padding=0)
        #self.bnB_1 = nn.BatchNorm2d(512)
        self.convB_2 = nn.Conv2d(512, 512, kernel_size=1, padding=0)
        #self.bnB_2 = nn.BatchNorm2d(512)

        #decoder 1 1x1x512 ->  2x2x512
        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, padding=0)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)

        #decoder 2 2x2x512 ->  4x4x512
        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0)

        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6_1 = nn.BatchNorm2d(512)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6_2 = nn.BatchNorm2d(512)

        #decoder 3 4x4x512 ->  8x8x256
        self.conv7_c = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)

        self.conv7_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7_1 = nn.BatchNorm2d(256)
        self.conv7_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7_2 = nn.BatchNorm2d(256)

        #decoder 4 8x8x256 ->  16x16x128
        self.conv8_c = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)

        self.conv8_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn8_1 = nn.BatchNorm2d(128)
        self.conv8_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn8_2 = nn.BatchNorm2d(128)

        #decoder 5 16x16x128 ->  32x32x64
        self.conv9_c = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.up5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)

        self.conv9_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn9_1 = nn.BatchNorm2d(64)
        self.conv9_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn9_2 = nn.BatchNorm2d(64)

        self.match_size = nn.ConvTranspose2d(64, 64, kernel_size=2,stride=2, padding=0)
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, Input):
        #Encoder Block 1
        x = self.conv1_1(Input)
        x = self.bn1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)

        p1 = self.pool(x)
        print('x1=',x.shape)
        print('p1=',p1.shape)

        # Encoder Block 2
        x = self.conv2_1(p1)
        x = self.bn2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu(x)

        p2 = self.pool(x)
        print('x2=',x.shape)
        print('p2=',p2.shape)

        # Encoder Block 3
        x = self.conv3_1(p2)
        x = self.bn3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu(x)

        p3 = self.pool(x)
        print('x3=',x.shape)
        print('p3=',p3.shape)

        # Encoder Block 4
        x = self.conv4_1(p3)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)

        p4 = self.pool(x)
        print('x4=',x.shape)
        print('p4=',p4.shape)

        # Encoder Block 5
        x = self.conv5_1(p4)
        x = self.bn5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.relu(x)

        p5 = self.pool(x)
        print('x5=',x.shape)
        print('p5=',p5.shape)

        # Bottleneck - Encoder Block 6
        x = self.convB_1(p5)
        #x = self.bnB_1(x)
        #x = self.relu(x)
        x = self.convB_2(x)
        #x = self.bnB_2(x)
        x = self.relu(x)
        print('xB=',x.shape)

        #Decoder block 1
        #print(x.shape)
        x = self.up1(x)
        print(x.shape)
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.relu(x)

        # Decoder block 2
        print(x.shape)
        print(p5.shape)
        x = torch.cat([x, p5], axis=1)
        x = self.up2(x)
        x = self.conv6_1(x)
        x = self.bn6_1(x)
        x = self.relu(x)
        x = self.conv6_2(x)
        x = self.bn6_2(x)
        x = self.relu(x)

        # Decoder block 3
        print(x.shape)
        print(p4.shape)
        x = torch.cat([x, p4], axis=1)
        x = self.conv7_c(x)
        x = self.up3(x)
        x = self.conv7_1(x)
        x = self.bn7_1(x)
        x = self.relu(x)
        x = self.conv7_2(x)
        x = self.bn7_2(x)
        x = self.relu(x)

        # Decoder block 4
        print(x.shape)
        print(p3.shape)
        x = torch.cat([x, p3], axis=1)
        x = self.conv8_c(x)
        x = self.up4(x)
        x = self.conv8_1(x)
        x = self.bn8_1(x)
        x = self.relu(x)
        x = self.conv8_2(x)
        x = self.bn8_2(x)
        x = self.relu(x)

        # Decoder block 5
        print(x.shape)
        print(p2.shape)
        x = torch.cat([x, p2], axis=1)
        x = self.conv9_c(x)
        x = self.up5(x)
        x = self.conv9_1(x)
        x = self.bn9_1(x)
        x = self.relu(x)
        x = self.conv9_2(x)
        x = self.bn9_2(x)
        x = self.relu(x)

        print(x.shape)

        #Final Output
        x=self.match_size(x)
        print("match dimension", x.shape)
        outputs = self.outputs(x)
        return outputs
    
from torchsummary import summary

def test():
    x = torch.randn((100,3,64,64))
    model=UNet()
    preds = model(x)
    print(preds.shape)
    preds.shape == x.shape
    # summary(model, input_size=(3,64,64))

if __name__ == "__main__":
    test()


