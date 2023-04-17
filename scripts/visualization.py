import torch
import torchvision
from time import time
import numpy as np

# t-SNE required
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

# model
# from main_discover import Discoverer


def get_data(dataset_name, dataset_dir):
    # get dataset Class
    dataset_class = getattr(torchvision.datasets, dataset_name)
    train_set = dataset_class(root=dataset_dir, train=True, download=False, transform=torchvision.transforms.ToTensor())
    test_set = dataset_class(root=dataset_dir, train=False, download=False, transform=torchvision.transforms.ToTensor())
    # concatenate all data
    dataset = train_set + test_set
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=600, shuffle=False, num_workers=4)

    return data_loader


def prepare_logits(data_loader, ckpt_path, save_path):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # load model
    model = Discoverer.load_from_checkpoint(ckpt_path).to(device)
    model.eval()

    logits_list = []
    labels_list = []
    with torch.no_grad():
        count = 0
        for images, labels in data_loader:
            count += 1
            print(images.size(), labels.size(), count)

            images = images.to(device)
            # forward
            y_hat = model(images)

            # gather outputs
            y_hat["logits_lab"] = (y_hat["logits_lab"].unsqueeze(0).expand(4, -1, -1))

            # concat: (num_heads, batch_size, num_labeled_classes + num_unlabeled_classes)
            logits_head = torch.cat([y_hat["logits_lab"], y_hat["logits_unlab"]], dim=-1)
            logits_multihead = torch.cat([head for head in logits_head], dim=-1)
            print(logits_head.shape, logits_multihead.shape)

            logits_list.append(logits_multihead.cpu())
            labels_list.append(labels)

    outputs = {
        'logits': torch.cat(logits_list),
        'labels': torch.cat(labels_list)
    }
    print(outputs['logits'].shape, outputs['labels'].shape)
    torch.save(outputs, save_path)
    print('###################Save logits successfully###########################')


def fashion_scatter(x, colors):
    # choose a color palette with seaborn
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))
    # sns.palplot(sns.color_palette(palette))

    # create a scatter plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []
    for i in range(num_classes):
        # Position of each label at median of data points.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def t_sne(save_path):
    # read logits
    outputs = torch.load(save_path)
    logits = outputs['logits']
    labels = outputs['labels']
    print(logits.shape, labels.shape)

    # seaborn setting
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    time_start = time()

    fashion_tsne = TSNE(n_components=2, init='pca', random_state=123, n_jobs=-1).fit_transform(logits)

    f, ax, sc, txts = fashion_scatter(fashion_tsne, labels)
    f.show()
    f.savefig(save_path.replace('pth', 'pdf'))
    # plt.show()
    print('t-SNE done! Time elapsed: {} seconds'.format(time() - time_start))


if __name__ == '__main__':
    dataset_name = 'CIFAR10'
    dataset_dir = '/data/fzc'
    ckpt_path = './CIFAR10_baseline.ckpt'
    save_path = './CIFAR10_baseline.pth'

    print('#################### Start #########################')
    # data_loader = get_data(dataset_name, dataset_dir)
    # prepare_logits(data_loader, ckpt_path, save_path)
    t_sne(save_path)
    print('####################### Over #####################')
