from train import YoloLoss

def eval(model, device, imgs, labels, num_classes, minibatch_size):
    """
    Inference:
    
    Args:
        model: the model that gives us a prediction
        imgs: the images that it makes predictions on. shape: (minibatch_size, 3, 416, 416)
    """
    model.eval()

    loss_layer = YoloLoss(
        num_classes = num_classes,
        batch_size = minibatch_size,
        all_anchors = model.anchors,
        anchor_masks = model.anchor_masks,
        img_size = model.img_size,
        num_anchors = len(model.anchors) // 3,
        device = device
    )

    output = model(imgs)
    bboxes = loss_layer(output, labels, plot_conf_heatmap = False, eval = True)

    return bboxes