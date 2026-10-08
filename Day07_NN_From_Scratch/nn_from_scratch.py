import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,)),])

train_ds = datasets.MNIST(root='./data' , train=True, download=True,transform = transform)
test_ds = datasets.MNIST(root='./data' , train=False, download=True,transform = transform)

X_train = train_ds.data.numpy().reshape(-1,784).astype(np.float32) / 255.0
y_train = train_ds.targets.numpy()
X_test = test_ds.data.numpy().reshape(-1,784).astype(np.float32) / 255.0
y_test = test_ds.targets.numpy()

mean,std = X_train.mean(),X_train.std()
X_train = (X_train- mean) / std
X_test = (X_test - mean) / std

print(f"Train: {X_train.shape} | Test: {X_test.shape}")


rng = np.random.default_rng(42)
D,H,K = 784,128,10
w1 = (rng.standard_normal((D, H)) * np.sqrt(2.0 / D)).astype(np.float32)
b1 = np.zeros(H,dtype=np.float32)
w2 = (rng.standard_normal((H, K)) * np.sqrt(2.0 / H)).astype(np.float32)
b2 = np.zeros(K,dtype=np.float32)

print(f'Parameters: w1{w1.shape} , b1:{b1.shape} , w2 : {w2.shape} , b2 : {b2.shape}')
print(f"Total parameters: {w1.size + b1.size + w2.size + b2.size:,}")

def forward(X):
    z1 = X @ w1 + b1
    a1 = np.maximum(0,z1)
    z2 = a1 @ w2 + b2
    z2_shift = z2 - z2.max(axis=1,keepdims=True)
    e = np.exp(z2_shift)
    probs = e / e.sum(axis=1,keepdims=True)
    cache = (X,z1,a1,z2,probs)
    return probs,cache

def backward(cache,y):
    X,z1,a1,z2,probs = cache
    N = X.shape[0]

    dz2 = probs.copy()
    dz2[np.arange(N),y] -= 1
    dz2 /=N

    dw2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ w2.T
    dz1 = da1 * (z1 > 0)
    dw1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    return dw1,db1,dw2,db2

def cross_entropy(probs,y):
    N = len(y)
    return -np.log(probs[np.arange(N),y] + 1e-8).mean()


def iterate_batches(X,y,batch_size=64,shuffle=True):
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle :
        rng.shuffle(idx)
    for start in range(0,N,batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx],y[batch_idx]

def evaluate(X,y):
    probs, _  = forward(X)
    preds = probs.argmax(axis=1)
    return cross_entropy(probs,y),(preds==y).mean()


epochs = 10 
lr = 0.1
batch_size = 64

train_losses , test_losses = [],[]
train_accs,test_accs = [],[]

for epoch in range(epochs):
    running_loss,correct,total = 0.0,0,0
    for X_batch, y_batch in iterate_batches(X_train,y_train,batch_size):
        probs,cache = forward(X_batch)
        loss = cross_entropy(probs,y_batch)

        dw1,db1,dw2,db2 = backward(cache,y_batch)

        w1 -= lr * dw1; b1 -= lr * db1
        w2 -= lr * dw2; b2 -= lr * db2

        running_loss += loss * X_batch.shape[0]
        correct += (probs.argmax(1)== y_batch).sum().item()
        total+=X_batch.shape[0]


    train_loss = running_loss / total
    train_acc = correct / total
    test_acc ,test_loss = evaluate(X_test,y_test)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1:2d} | " f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
          f"Test Loss {test_loss:.4f} | Test Acc {test_acc:.4f}")


fig , axes = plt.subplots(1,2,figsize=(12,4))

axes[0].plot(train_losses,label='Train')
axes[0].plot(test_losses,label='Test')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(train_accs,label='Train')
axes[1].plot(test_accs,label='Test')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy')
axes[1].legend()
axes[1].grid(True)


plt.tight_layout()
plt.savefig("nn_scratch_results.png",dpi=100)
plt.show()

print(f"\n✅ Done! Final test accuracy: {test_accs[-1]:.4f}")
print("Saved nn_scratch_results.png")