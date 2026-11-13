import io

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("tensorflow")
import keras

from automl_agents.catalog import CATALOG
from automl_agents.preprocessing import decode_and_preprocess
from automl_agents.trainer import build_model


def test_small_cnn_export_preserves_predictions_and_normalization(tmp_path):
    image = Image.new("L", (28, 28), 255)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    inputs = decode_and_preprocess(buf.getvalue(), 28, "small_cnn").numpy()[None]
    np.testing.assert_array_equal(inputs, np.ones((1, 28, 28, 3)))
    # pretrained=True must still train this local architecture without downloads.
    model = build_model(CATALOG["SmallCNN"], 10, 28, .2, True, .001, True)
    before = model(inputs, training=False).numpy()
    weights = [w.numpy().copy() for w in model.trainable_weights]
    model.train_on_batch(inputs, np.array([3]))
    assert any(not np.array_equal(w, v.numpy()) for w, v in zip(weights, model.trainable_weights))
    prediction = model(inputs, training=False).numpy()
    assert not np.array_equal(before, prediction)
    path = tmp_path / "model.keras"
    model.save(path)
    restored = keras.saving.load_model(path, compile=False)
    np.testing.assert_allclose(prediction, restored(inputs, training=False).numpy(), atol=1e-6)
    augmentation = model.get_layer("spatial_augmentation")
    assert not any(isinstance(layer, keras.layers.RandomFlip) for layer in augmentation.layers)
    keras.backend.clear_session()
