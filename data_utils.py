import os
import torch
from torchvision import datasets, transforms, models

import clip
from pytorchcv.model_provider import get_model as ptcv_get_model

DATASET_ROOTS = {
    "imagenet_train": "YOUR_PATH/CLS-LOC/train/",
    "imagenet_val": "YOUR_PATH/ImageNet_val/",
    "cub_train":"data/CUB/train",
    "cub_val":"data/CUB/test"
}

BIOCLIP_CKPT = os.environ.get("LFCBM_BIOCLIP_CKPT",
                              "/workspace/models/bioclip/open_clip_pytorch_model.bin")

LABEL_FILES = {"places365":"data/categories_places365_clean.txt",
               "imagenet":"data/imagenet_classes.txt",
               "cifar10":"data/cifar10_classes.txt",
               "cifar100":"data/cifar100_classes.txt",
               "cub":"data/cub_classes.txt"}

# Custom Data
DATASET_ROOTS["treeoflife_train"] = "data/treeoflife/train"
DATASET_ROOTS["treeoflife_val"]   = "data/treeoflife/val"
LABEL_FILES["treeoflife"] = "data/treeoflife.txt"

DATASET_ROOTS["birds525_train"] = "data/birds525/train"
DATASET_ROOTS["birds525_val"]   = "data/birds525/val"
LABEL_FILES["birds525"] = "data/birds525.txt"

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess


def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar100_train":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, 
                                   transform=preprocess)
        
    elif dataset_name == "cifar10_train":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)
        
    elif dataset_name == "cifar10_val":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                   transform=preprocess)
        
    elif dataset_name == "places365_train":
        try:
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=True,
                                       transform=preprocess)
        except(RuntimeError):
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=False,
                                   transform=preprocess)
            
    elif dataset_name == "places365_val":
        try:
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=True,
                                   transform=preprocess)
        except(RuntimeError):
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=False,
                                   transform=preprocess)
        
    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)
               
    elif dataset_name == "imagenet_broden":
        data = torch.utils.data.ConcatDataset([datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess), 
                                                     datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess)])
    return data

def get_targets_only(dataset_name):
    pil_data = get_data(dataset_name)
    return pil_data.targets

class BioCLIPBackbone(torch.nn.Module):
    """BioCLIP's image tower, tapped at visual.ln_post.

    Returns the 768-d pre-projection features instead of the 512-d output of
    encode_image, so the concept projection sees the true penultimate layer.
    Kept as a Module (not a lambda) so that utils.save_target_activations can
    resolve the `visual.ln_post` attribute path when registering its own hook.
    """
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        self._feat = {}
        self.visual.ln_post.register_forward_hook(
            lambda module, inp, out: self._feat.__setitem__("f", out))

    def forward(self, x):
        self.visual(x)
        feat = self._feat["f"]
        # current open_clip applies ln_post after pooling (2-D); older versions
        # apply it to the full token sequence, hence the mean over tokens
        return feat.mean(dim=1).float() if feat.dim() == 3 else feat.float()


#  timm backbones used as baselines in the BioCLIP paper: same ViT-B/16 architecture
#  as BioCLIP, different pretraining objective, so they isolate pretraining from scale.
TIMM_BACKBONES = {
    "vit_in21k":   "vit_base_patch16_224.augreg_in21k",  # supervised ImageNet-21k
    "dino_vitb16": "vit_base_patch16_224.dino",          # DINO self-supervised
}


class TimmBackbone(torch.nn.Module):
    """A timm vision model exposed as a feature extractor.

    `self.out` is an Identity whose output is the final pooled feature, so hooking it
    (--feature_layer out) hands save_target_activations an already-2-D tensor. That
    keeps the pooling decision here instead of in utils.get_activation, which would
    otherwise mean-pool ViT tokens and silently discard the CLS token.
    """
    def __init__(self, model, pool="cls"):
        super().__init__()
        self.model = model
        self.pool = pool
        self.out = torch.nn.Identity()

    def forward(self, x):
        feats = self.model.forward_features(x)
        if feats.dim() == 3:                                  # (B, tokens, D)
            feats = feats[:, 0] if self.pool == "cls" else feats.mean(dim=1)
        elif feats.dim() == 4:                                # (B, D, H, W)
            feats = feats.mean(dim=[2, 3])
        return self.out(feats.float())


def load_timm_backbone(target_name, device, pool="cls"):
    """Load a timm backbone plus the preprocessing that model was trained with.

    num_classes=0 drops the classifier head (in21k's is 21843-wide). The transform
    comes from the model's own data config -- augreg_in21k and DINO use different
    normalisation, so a shared ImageNet transform would be wrong for one of them.
    """
    import timm
    model = timm.create_model(TIMM_BACKBONES[target_name], pretrained=True, num_classes=0)
    model = model.eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    preprocess = timm.data.create_transform(**cfg)
    return TimmBackbone(model, pool=pool).to(device).eval(), preprocess


def load_bioclip(device):
    """Load the BioCLIP ViT-B/16 checkpoint into an open_clip model."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16")
    checkpoint = torch.load(BIOCLIP_CKPT, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return model, preprocess


def get_target_model(target_name, device):

    if target_name == "bioclip":
        model, preprocess = load_bioclip(device)
        target_model = BioCLIPBackbone(model).to(device).eval()

    elif target_name in TIMM_BACKBONES:
        target_model, preprocess = load_timm_backbone(target_name, device)

    elif target_name.startswith("clip_"):
        target_name = target_name[5:]
        model, preprocess = clip.load(target_name, device=device)
        target_model = lambda x: model.encode_image(x).float()
    
    elif target_name == 'resnet18_places': 
        target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
        
    elif target_name == 'resnet18_cub':
        target_model = ptcv_get_model("resnet18_cub", pretrained=True).to(device)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    
    elif target_name.endswith("_v2"):
        target_name = target_name[:-3]
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
        
    else:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
    
    return target_model, preprocess