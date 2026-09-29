"""Architecture du modèle (3.2) — nécessite TensorFlow (pipeline offline uniquement).

Entrée : image (N, 128, 128, 3) float32 dans [0, 1] (sortie de `preprocessing.preprocess`).
Le passage à l'échelle [-1, 1] attendue par MobileNetV2 est une couche DU modèle :
il est donc embarqué dans l'ONNX et le contrat de prétraitement reste [0, 1].
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

FEATURES_LAYER = "out_relu"      # dernière couche convolutive de MobileNetV2 (Grad-CAM)
HEAD_DENSE = "head_dense"
OUTPUT_LAYER = "probability"
NON_BACKBONE = {"image", "to_mobilenet_range", "gap", HEAD_DENSE, "head_dropout", OUTPUT_LAYER}


def build_model(cfg: dict, weights: str | None = "imagenet") -> keras.Model:
    size = cfg["preprocessing"]["image_size"]
    m = cfg["model"]
    inputs = keras.Input((size, size, 3), name="image")
    x = layers.Rescaling(2.0, offset=-1.0, name="to_mobilenet_range")(inputs)
    # input_tensor => les couches du backbone sont "à plat" dans le modèle final,
    # ce qui rend la couche de features accessible pour Grad-CAM.
    base = keras.applications.MobileNetV2(input_tensor=x, include_top=False, weights=weights)
    features = base.output  # couche "out_relu"
    x = layers.GlobalAveragePooling2D(name="gap")(features)
    x = layers.Dense(m["dense_units"], activation="relu", name=HEAD_DENSE)(x)
    x = layers.Dropout(m["dropout"], name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name=OUTPUT_LAYER)(x)
    model = keras.Model(inputs, outputs, name="palunet")
    set_backbone_trainable(model, trainable_last=0)
    return model


def backbone_layers(model: keras.Model) -> list[layers.Layer]:
    return [l for l in model.layers if l.name not in NON_BACKBONE]


def set_backbone_trainable(model: keras.Model, trainable_last: int) -> None:
    """Gèle le backbone sauf ses `trainable_last` dernières couches (B2).

    Les BatchNormalization restent gelées en phase de fine-tuning (pratique
    standard : évite de détruire les statistiques ImageNet avec de petits batchs).
    """
    bb = backbone_layers(model)
    for i, layer in enumerate(bb):
        unfrozen = trainable_last > 0 and i >= len(bb) - trainable_last
        layer.trainable = unfrozen and not isinstance(layer, layers.BatchNormalization)


def build_inference_model(model: keras.Model) -> keras.Model:
    """Modèle d'export : sorties `probability` (N, 1) et `cam` (N, h, w).

    `cam` est la carte Grad-CAM brute (avant ReLU) vers la classe d'indice 1.
    La tête (GAP -> Dense ReLU -> Dense) permet un gradient analytique :
        d logit / d A[i,j,k] = sum_u w2[u] * 1[h_u > 0] * W1[k,u] / (H*W)
    constant sur (i, j) : c'est exactement le poids alpha_k de Grad-CAM.
    Aucun calcul de gradient n'est donc nécessaire à l'inférence (ONNX Runtime).
    Carte finale : ReLU(cam) pour la classe 1, ReLU(-cam) pour la classe 0.
    """
    features = model.get_layer(FEATURES_LAYER).output
    dense = model.get_layer(HEAD_DENSE)
    out = model.get_layer(OUTPUT_LAYER)

    def cam_fn(a):
        hw = tf.cast(tf.shape(a)[1] * tf.shape(a)[2], a.dtype)
        pooled = tf.reduce_mean(a, axis=[1, 2])
        h = tf.matmul(pooled, dense.kernel) + dense.bias
        mask = tf.cast(h > 0, a.dtype)
        alpha = tf.matmul(mask * tf.transpose(out.kernel), dense.kernel, transpose_b=True) / hw
        return tf.reduce_sum(a * alpha[:, None, None, :], axis=-1)

    cam = layers.Lambda(cam_fn, name="cam")(features)
    return keras.Model(model.input, [model.output, cam], name="palunet_inference")
