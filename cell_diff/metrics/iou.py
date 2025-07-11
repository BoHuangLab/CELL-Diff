import torch

def binarize_img(img, threshold_mode="median", quantile_q=0.5):
    if threshold_mode == "mean":
        return img > torch.mean(img, dim=(-2, -1), keepdim=True)
    elif threshold_mode == "median":
        B, C, H, W = img.shape
        img_flat = img.reshape(B, C, H * W)
        median_vals = torch.median(img_flat, dim=-1, keepdim=True)[0]
        binary_img = img_flat > median_vals
        return binary_img.reshape(B, C, H, W)
    elif threshold_mode == "quantile":
        B, C, H, W = img.shape
        img_flat = img.reshape(B, C, H * W)
        quantile_vals = torch.quantile(img_flat, q=quantile_q, dim=-1, keepdim=True)
        binary_img = img_flat > quantile_vals
        return binary_img.reshape(B, C, H, W)        
    else:
        raise ValueError(f"Unsupported threshold mode: {threshold_mode}")

def compute_iou(mask1, mask2):
    """
    Compute Intersection over Union (IoU) between two masks.
    
    Parameters:
    mask1 (torch.Tensor): Ground truth mask of shape (B, 1, H, W)
    mask2 (torch.Tensor): Predicted mask of shape (B, 1, H, W)
    
    Returns:
    float: IoU
    """
    mask1 = mask1.bool()
    mask2 = mask2.bool()

    # Compute intersection and union
    intersection = torch.sum((mask1 & mask2).float(), dim=(-1, -2, -3))
    union = torch.sum((mask1 | mask2).float(), dim=(-1, -2, -3))
    iou = intersection / union
    
    return iou
