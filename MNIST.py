import matplotlib.pyplot as plt
import torch
import torch.nn.functional
import torchvision
from torch.utils import data
from torchvision import transforms

if __name__ == "__main__":
    def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """设置matplotlib的轴
        Defined in :numref:`sec_calculus`"""
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
            ylim=None, xscale='linear', yscale='linear',
            fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
        """绘制数据点
        Defined in :numref:`sec_calculus`"""
        if legend is None:
            legend = []

        axes = axes if axes else plt.gca()

        # 如果X有一个轴，输出True
        def has_one_axis(X):
            return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                    and not hasattr(X[0], "__len__"))

        if has_one_axis(X):
            X = [X]
        if Y is None:
            X, Y = [[]] * len(X), X
        elif has_one_axis(Y):
            Y = [Y]
        if len(X) != len(Y):
            X = X * len(Y)
        axes.cla()
        for x, y, fmt in zip(X, Y, fmts):
            if len(x):
                axes.plot(x, y, fmt)
            else:
                axes.plot(y, fmt)
        set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        
    # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
    # 并除以255使得所有像素的数值均在0到1之间
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data_fashion_mnist", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data_fashion_mnist", train=False, transform=trans, download=True)

    def get_fashion_mnist_labels(labels):
        """返回Fashion-MNIST数据集的文本标签"""
        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [text_labels[int(i)] for i in labels]

    def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
        """绘制图像列表"""
        figsize = (num_cols * scale, num_rows * scale)
        _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()
        for i, (ax, img) in enumerate(zip(axes, imgs)):
            if torch.is_tensor(img):
                # 图片张量
                ax.imshow(img.numpy())
            else:
                # PIL图片
                ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if titles:
                ax.set_title(titles[i])
        return axes

    batch_size = 256

    def get_dataloader_workers():
        """使用4个进程来读取数据"""
        return 4

    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                num_workers=get_dataloader_workers())

    def load_data_fashion_mnist(batch_size, resize=None):
        """下载Fashion-MNIST数据集,然后将其加载到内存中"""
        trans = [transforms.ToTensor()]
        if resize:
            trans.insert(0, transforms.Resize(resize))
        trans = transforms.Compose(trans)
        mnist_train = torchvision.datasets.FashionMNIST(
            root="./data_fashion_mnist", train=True, transform=trans, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(
            root="./data_fashion_mnist", train=False, transform=trans, download=True)
        return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                                num_workers=get_dataloader_workers()),
                data.DataLoader(mnist_test, batch_size, shuffle=False,
                                num_workers=get_dataloader_workers()))
        
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    num_inputs = 784
    num_outputs = 10

    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    torch.nn.init.xavier_uniform_(W)
    b = torch.zeros(num_outputs, requires_grad=True)

    def softmax(X):
        X_exp = torch.exp(X)
        partition = X_exp.sum(1, keepdim=True)
        return X_exp / partition  # 这里应用了广播机制

    '''
    def net(X):
        return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
    '''
    
    net = torch.nn.Sequential(torch.nn.Flatten(),
                        torch.nn.Linear(784, 256),
                        torch.nn.ReLU(),
                        torch.nn.Linear(256, 10))
    
    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)
    '''
    def cross_entropy(y_hat, y):
        return - torch.log(y_hat[range(len(y_hat)), y])
    '''
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    
    def accuracy(y_hat, y):
        """计算预测正确的数量"""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())

    def evaluate_accuracy(net, data_iter):
        """计算在指定数据集上模型的精度"""
        if isinstance(net, torch.nn.Module):
            net.eval()  # 将模型设置为评估模式
        metric = Accumulator(2)  # 正确预测数、预测总数
        with torch.no_grad():
            for X, y in data_iter:
                metric.add(accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]

    class Accumulator:
        """在n个变量上累加"""
        def __init__(self, n):
            self.data = [0.0] * n

        def add(self, *args):
            self.data = [a + float(b) for a, b in zip(self.data, args)]

        def reset(self):
            self.data = [0.0] * len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]
        
    def train_epoch(net, train_iter, loss, updater):
        """训练模型一个迭代周期"""
        # 将模型设置为训练模式
        if isinstance(net, torch.nn.Module):
            net.train()
        # 训练损失总和、训练准确度总和、样本数
        metric = Accumulator(3)
        for X, y in train_iter:
            # 计算梯度并更新参数
            y_hat = net(X)
            l = loss(y_hat, y)
            if isinstance(updater, torch.optim.Optimizer):
                # 使用PyTorch内置的优化器和损失函数
                updater.zero_grad()
                l.mean().backward()
                updater.step()
            else:
                # 使用定制的优化器和损失函数
                l.sum().backward()
                updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        # 返回训练损失和训练精度
        return metric[0] / metric[2], metric[1] / metric[2]

    class Animator:
        """在动画中绘制数据"""
        def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                    ylim=None, xscale='linear', yscale='linear',
                    fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                    figsize=(3.5, 2.5)):
            # 增量地绘制多条线
            if legend is None:
                legend = []
            self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
            if nrows * ncols == 1:
                self.axes = [self.axes, ]
            # 使用lambda函数捕获参数
            self.config_axes = lambda: set_axes(
                self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
            self.X, self.Y, self.fmts = None, None, fmts

        def add(self, x, y):
            # 向图表中添加多个数据点
            if not hasattr(y, "__len__"):
                y = [y]
            n = len(y)
            if not hasattr(x, "__len__"):
                x = [x] * n
            if not self.X:
                self.X = [[] for _ in range(n)]
            if not self.Y:
                self.Y = [[] for _ in range(n)]
            for i, (a, b) in enumerate(zip(x, y)):
                if a is not None and b is not None:
                    self.X[i].append(a)
                    self.Y[i].append(b)
            self.axes[0].cla()
            for x, y, fmt in zip(self.X, self.Y, self.fmts):
                self.axes[0].plot(x, y, fmt)
            self.config_axes()
            
    def train_model(net, train_iter, test_iter, loss, num_epochs, lr):
        """训练模型"""
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
        
        # 创建优化器
        optimizer = torch.optim.SGD(net.parameters(), lr)

        for epoch in range(num_epochs):
            train_metrics = train_epoch(net, train_iter, loss, optimizer)
            test_acc = evaluate_accuracy(net, test_iter)
            animator.add(epoch + 1, train_metrics + (test_acc,))
        
        train_loss, train_acc = train_metrics
        assert train_loss < 0.5, train_loss
        assert train_acc <= 1 and train_acc > 0.7, train_acc
        assert test_acc <= 1 and test_acc > 0.7, test_acc
    '''
    def train_model(net, train_iter, test_iter, loss, num_epochs, updater):
        
        """训练模型"""
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
        for epoch in range(num_epochs):
            train_metrics = train_epoch(net, train_iter, loss, updater)
            test_acc = evaluate_accuracy(net, test_iter)
            animator.add(epoch + 1, train_metrics + (test_acc,))
        train_loss, train_acc = train_metrics
        assert train_loss < 0.5, train_loss
        assert train_acc <= 1 and train_acc > 0.7, train_acc
        assert test_acc <= 1 and test_acc > 0.7, test_acc
    '''
    lr = 0.1

    def sgd(params, lr, batch_size):
        """小批量随机梯度下降"""
        with torch.no_grad():
            for param in params:
                param -= lr * param.grad / batch_size
                param.grad.zero_()

    def updater(batch_size):
        return torch.optim.SGD(net.parameters(), lr)

    num_epochs = 5
    train_model(net, train_iter, test_iter, loss, num_epochs, lr)

    def predict_model(net, test_iter, n=100):
        ac = 0
        """预测标签"""
        for X, y in test_iter:
            break
        trues = get_fashion_mnist_labels(y)
        preds = get_fashion_mnist_labels(softmax(net(X)).argmax(axis=1))
        for i in range(len(trues)):
            ac += (trues[i] == preds[i])
        print("ac:%.3f" % (ac / len(trues)))
        
    predict_model(net, test_iter)