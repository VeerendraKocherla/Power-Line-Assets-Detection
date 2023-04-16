import numpy as np
import tensorflow as tf
import cv2


def img_to_tensor(path, input_shape):
    img = cv2.imread(f"/content/plad_project/plad/{path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape != input_shape:
        img = cv2.resize(img, (input_shape[1], input_shape[0]))
    return tf.convert_to_tensor(img)
    # return img


def scale_bbox(row, input_shape):
    bbox = row['bboxes']
    bbox[:, 0] = (bbox[:, 0] * input_shape[1]) // row['width']
    bbox[:, 1] = (bbox[:, 1] * input_shape[0]) // row['height']
    bbox[:, 2] = (bbox[:, 2] * input_shape[1]) // row['width']
    bbox[:, 3] = (bbox[:, 3] * input_shape[0]) // row['height']
    return bbox


def preprocess_true_boxes(true_boxes, anchors, anchor_mask,
                          input_shape=(3648, 5472), num_classes=5):
    # Preprocess true boxes to training input format

    # assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3  # default setting
    anchors = anchors * (input_shape[1], input_shape[0])
    true_boxes = np.array(true_boxes, dtype=np.float32)
    input_shape = np.array(input_shape, dtype=np.int32)
    # boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] /= input_shape[::-1]
    true_boxes[..., 2:4] /= input_shape[::-1]

    grid_shapes = np.array(
        [input_shape//{0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)])
    y_true = [np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]),
                        5+num_classes), dtype=np.float32) for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes

    # Discard zero rows.
    wh = boxes_wh[boxes_wh[..., 0] > 0]
    if len(wh) == 0:
        return
    # Expand dim to apply broadcasting.
    wh = np.expand_dims(wh, -2)
    box_maxes = wh / 2.
    box_mins = -box_maxes

    intersect_mins = np.maximum(box_mins, anchor_mins)
    intersect_maxes = np.minimum(box_maxes, anchor_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box_area = wh[..., 0] * wh[..., 1]
    anchor_area = anchors[..., 0] * anchors[..., 1]
    iou = intersect_area / (box_area + anchor_area - intersect_area)

    # Find best anchor for each true box
    best_anchor = np.argmax(iou, axis=-1)

    for t, n in enumerate(best_anchor):
        for l in range(num_layers):
            if n in anchor_mask[l]:
                i = np.floor(true_boxes[t, 0] *
                             grid_shapes[l, 1]).astype(np.int32)
                j = np.floor(true_boxes[t, 1] *
                             grid_shapes[l, 0]).astype(np.int32)
                k = int(np.where(anchor_mask[l] == n)[0])
                c = true_boxes[t, 4].astype(np.int32)
                y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                y_true[l][j, i, k, 4] = 1
                y_true[l][j, i, k, 5+c] = 1

    return tuple(y_true)


def draw_true_boxes(img_org, boxes, lookup):
    try:
        img = img_org.numpy().copy()
    except:
        img = img_org.copy()
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2])).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4])).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, str(lookup[int(boxes[i, -1])]),
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


def load_darknet_weights(model, weights_file):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    layers = [
        'yolo_darknet',
        'yolo_conv_0',
        'yolo_output_0',
        'yolo_conv_1',
        'yolo_output_1',
        'yolo_conv_2',
        'yolo_output_2',
    ]
    output_counter = 0
    params_output = [4984063, 1312511, 361471]
    for layer_name in layers:
        if layer_name.startswith('yolo_output'):
            __ = np.fromfile(wf, dtype=np.int32,
                             count=params_output[output_counter])
            output_counter += 1
            continue
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            # print("{}/{} {}".format(
            #     sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()
    print("weights loaded successfully.")


def broadcast_iou(box_1, box_2):
    # box_1: shape = (..., (x1, y1, x2, y2))
    # box_2: shape = (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)
