import torch
import torchvision
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# model
from main_discover import Discoverer


def get_data(dataset_name, dataset_dir, num_labeled_classes, num_unlabeled_classes):
    # get dataset class
    dataset_class = getattr(torchvision.datasets, dataset_name)
    test_set = dataset_class(root=dataset_dir, train=False, download=False, transform=torchvision.transforms.ToTensor())

    # labeled classes
    labeled_classes = range(num_labeled_classes)
    test_lab_indices = np.where(np.isin(np.array(test_set.targets), labeled_classes))[0]
    test_lab_subset = torch.utils.data.Subset(test_set, test_lab_indices)
    # unlabeled classes
    unlabeled_classes = range(num_labeled_classes, num_labeled_classes + num_unlabeled_classes)
    test_unlab_indices = np.where(np.isin(np.array(test_set.targets), unlabeled_classes))[0]
    test_unlab_subset = torch.utils.data.Subset(test_set, test_unlab_indices)
    print(len(test_set), len(test_lab_subset), len(test_unlab_subset))

    lab_dataloader = torch.utils.data.DataLoader(test_lab_subset, batch_size=20, shuffle=False, num_workers=4)
    unlab_dataloader = torch.utils.data.DataLoader(test_unlab_subset, batch_size=20, shuffle=False, num_workers=4)

    return {'lab': lab_dataloader, 'unlab': unlab_dataloader}


def compute_confusion_matrix(data_loader, ckpt_path):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # load model
    model = Discoverer.load_from_checkpoint(ckpt_path).to(device)
    model.eval()

    # initial confusion matrix
    confusion_aware = {}
    confusion_agnostic = {}
    key = '{}_{}_{}'  # subset_label_misclassification

    with torch.no_grad():
        for subset in data_loader.keys():
            count = 0
            for images, labels in data_loader[subset]:
                images = images.to(device)
                # forward
                outputs = model(images)
                # print(outputs['logits_lab'].shape, outputs['logits_unlab'].shape)

                # pass through labeled head
                if subset == 'lab':
                    preds_aware = outputs["logits_lab"]  # task-aware
                    preds_agnostic = torch.cat(
                        [outputs["logits_lab"],
                         outputs["logits_unlab"][0],  # select first head
                         ],
                        dim=-1
                    )  # task-agnostic
                # pass through unlabeled head
                else:
                    preds_aware = outputs["logits_unlab"][0]  # task-aware, select first head
                    preds_agnostic = torch.cat(
                        [
                            outputs["logits_lab"].unsqueeze(0).expand(4, -1, -1),
                            outputs["logits_unlab"],
                        ],
                        dim=-1,
                    )  # task-agnostic
                preds_aware = preds_aware.max(dim=-1)[1]
                preds_agnostic = preds_agnostic.max(dim=-1)[1]
                print('Subset: {}, Count: {}'.format(subset, count))
                count += 1

                # label matching for clustering
                if subset == 'unlab':
                    labels_aware = labels - labels.min()  # align labels for clustering on task-aware
                    preds_agnostic_best_head = preds_agnostic[0]  # select first head
                    # hungarian algorithm
                    mapping_aware = compute_best_mapping(labels_aware.cpu().numpy(), preds_aware.cpu().numpy())
                    mapping_agnostic = compute_best_mapping(labels.cpu().numpy(), preds_agnostic_best_head.cpu().numpy())
                    # calibrate predictions
                    temp_preds_aware = torch.zeros_like(preds_aware)
                    temp_preds_agnostic = torch.zeros_like(preds_agnostic_best_head)
                    for i, j in mapping_aware:
                        temp_preds_aware[preds_aware == i] = j
                    for i, j in mapping_agnostic:
                        temp_preds_agnostic[preds_agnostic_best_head == i] = j
                    preds_aware = temp_preds_aware
                    preds_agnostic = temp_preds_agnostic
                # print(preds_aware)
                # print(labels_aware)
                # print(preds_agnostic)
                # print(labels)

                # count misclassification
                # task-aware
                count_aware = preds_aware != labels.to(device) if subset == 'lab' else preds_aware != labels_aware.to(device)
                if count_aware.sum().item():
                    mis_indices_aware = torch.nonzero(count_aware).squeeze()
                    if count_aware.sum().item() == 1:
                        label_aware = labels[mis_indices_aware].item() if subset == 'lab' else labels_aware[mis_indices_aware].item()
                        mis_aware = preds_aware[mis_indices_aware].item()
                        if key.format(subset, label_aware, mis_aware) not in confusion_aware.keys():
                            confusion_aware.update({key.format(subset, label_aware, mis_aware): 1})
                        else:
                            confusion_aware[key.format(subset, label_aware, mis_aware)] += 1
                    else:
                        mis_labels_aware = labels[mis_indices_aware] if subset == 'lab' else labels_aware[mis_indices_aware]
                        for label_aware, mis_aware in zip(mis_labels_aware, preds_aware[mis_indices_aware]):
                            if key.format(subset, label_aware, mis_aware) not in confusion_aware.keys():
                                confusion_aware.update({key.format(subset, label_aware, mis_aware): 1})
                            else:
                                confusion_aware[key.format(subset, label_aware, mis_aware)] += 1

                # task-agnostic
                count_agnostic = preds_agnostic != labels.to(device)
                if count_agnostic.sum().item():
                    mis_indices_agnostic = torch.nonzero(count_agnostic).squeeze()
                    if count_agnostic.sum().item() == 1:
                        label_agnostic = labels[mis_indices_agnostic].item()
                        mis_agnostic = preds_agnostic[mis_indices_agnostic].item()
                        if key.format(subset, label_agnostic, mis_agnostic) not in confusion_agnostic.keys():
                            confusion_agnostic.update({key.format(subset, label_agnostic, mis_agnostic): 1})
                        else:
                            confusion_agnostic[key.format(subset, label_agnostic, mis_agnostic)] += 1
                    else:
                        for label_agnostic, mis_agnostic in zip(labels[mis_indices_agnostic], preds_agnostic[mis_indices_agnostic]):
                            if key.format(subset, label_agnostic, mis_agnostic) not in confusion_agnostic.keys():
                                confusion_agnostic.update({key.format(subset, label_agnostic, mis_agnostic): 1})
                            else:
                                confusion_agnostic[key.format(subset, label_agnostic, mis_agnostic)] += 1

    # save as CSV files
    # print(confusion_aware)
    # print(confusion_agnostic)
    file_prefix = ckpt_path.split('_')[1]
    confusion_aware = pd.DataFrame(confusion_aware, index=[0])
    confusion_aware.to_csv(file_prefix + '_confusion_aware.csv')
    confusion_agnostic = pd.DataFrame(confusion_agnostic, index=[0])
    confusion_agnostic.to_csv(file_prefix + '_confusion_agnostic.csv')


def compute_best_mapping(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    return np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))


if __name__ == '__main__':
    dataset_name = 'CIFAR10'
    dataset_dir = '/data/fzc'
    num_labeled_classes = 5
    num_unlabeled_classes = 5
    ckpt_path = ['./CIFAR10_baseline.ckpt', './CIFAR10_inter.ckpt']

    print('#################### Start #########################')
    data_loader = get_data(dataset_name, dataset_dir, num_labeled_classes, num_unlabeled_classes)
    for ckpt in ckpt_path:
        compute_confusion_matrix(data_loader, ckpt)
        print('#################### {} is Done #########################'.format(ckpt))
    print('####################### Over #####################')
