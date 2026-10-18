import torch
import torch.nn as nn

def ex1_conv2d_shape():
    conv = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
    x =torch.randn(4,1,28,28)
    y = conv(x)

    assert y.shape==(4,6,24,24)
    print(f"✅ ex1 passed — Conv2d(1, 6, 5) on (4,1,28,28) → {tuple(y.shape)}")

def ex2_conv_params():
    conv = nn.Conv2d(1,6,5)
    expected = 1 * 6 * 5 * 5 + 6
    actual = sum(p.numel() for p in conv.parameters())

    assert actual == expected
    print(f"✅ ex2 passed — Conv2d(1,6,5) has {actual} params (expected {expected})")

def ex3_maxpool():
    pool = nn.MaxPool2d(2)
    x = torch.randn(4,6,24,24)
    y = pool(x)

    assert y.shape==(4,6,12,12)
    print(f"✅ ex3 passed — MaxPool2d(2) on (4,6,24,24) → {tuple(y.shape)}")

def ex4_flatten():
    x = torch.randn(4,6,12,12)
    y = x.view(x.size(0),-1)

    assert y.shape == (4,6 * 12 * 12)
    print(f"✅ ex4 passed — flatten (4,6,12,12) → {tuple(y.shape)}")

def ex5_lenet_forward():
    model = nn.Sequential(
        nn.Conv2d(1,6,5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(6,16,5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16*4*4,120),
        nn.ReLU(),
        nn.Linear(120,84),
        nn.ReLU(),
        nn.Linear(84,10),

    )

    x = torch.randn(4,1,28,28)
    y = model(x)
    assert y.shape == (4,10)
    print(f"✅ ex5 passed — LeNet forward: (4,1,28,28) → {tuple(y.shape)}")


def ex6_conv_kernel_visual():
    kernel = torch.tensor([[-1., -2., -1.],
                           [ 0.,  0.,  0.],
                           [ 1.,  2.,  1.]])

    conv = nn.Conv2d(1,1,3,bias=False)
    with torch.no_grad():
        conv.weight[0,0] = kernel

    img = torch.zeros(1,1,5,5)
    img[0,0,:2,:] = 1.0
    out = conv(img)

    assert out.abs().max() > 0.5
    print(f"✅ ex6 passed — Sobel filter detects horizontal edge")


def ex7_mlp_vs_cnn_params():
    """CNN has fewer params than equivalent MLP."""
    mlp = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    mlp_params = sum(p.numel() for p in mlp.parameters())

    cnn = nn.Sequential(
        nn.Conv2d(1, 6, 5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(6, 16, 5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 10),
    )
    cnn_params = sum(p.numel() for p in cnn.parameters())

    print(f"✅ ex7 passed — MLP: {mlp_params:,} | CNN: {cnn_params:,}")
    print(f"   CNN has {(1 - cnn_params/mlp_params)*100:.0f}% fewer params")


if __name__ == "__main__":
    print("=" * 50)
    print("Day 15 Exercises: CNNs")
    print("=" * 50 + "\n")

    ex1_conv2d_shape()
    ex2_conv_params()
    ex3_maxpool()
    ex4_flatten()
    ex5_lenet_forward()
    ex6_conv_kernel_visual()
    ex7_mlp_vs_cnn_params()

    print("\n🎉 All Day 15 exercises complete!")