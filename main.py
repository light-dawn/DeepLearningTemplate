from utils import create_model, CC_CCII_DatasetUtils
import config


def main_loop():
    data_utils = CC_CCII_DatasetUtils(config.DATA_ROOT, image_size=(224, 224))
    dataset = data_utils.process_dataset(batch_size=12)
    model = create_model("resnet18", "adam", "sparse_categorical_crossentropy", ["accuracy"])
    model.fit(dataset, epochs=10, steps_per_epoch=100)


if __name__ == "__main__":
    main_loop()