import tensorflow as tf
import pathlib

"""
说明
1. 所有的工具类方法都可以通过反射的方式调用
"""

# 构造模型工具类，所有创建模型结构的函数都放在这里
class MyModels:
    @staticmethod
    def unet(output_channels):
        base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

        # 使用这些层的激活设置
        layer_names = [
            "block_1_expand_relu",  # 64x64
            "block_3_expand_relu",  # 32x32
            "block_6_expand_relu",  # 16x16
            "block_13_expand_relu",  # 8x8
            "block_16_project",  # 4x4
        ]
        layers = [base_model.get_layer(name).output for name in layer_names]

        # 创建特征提取模型
        down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
        down_stack.trainable = True

        up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),  # 32x32 -> 64x64
        ]

        inputs = tf.keras.layers.Input(shape=[128, 128, 3])
        x = inputs

        # 在模型中下采样
        skips = down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # 上采样然后建立跳跃连接
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # 这是模型的最后一层
        last = tf.keras.layers.Conv2DTranspose(
            output_channels, 3, strides=2, padding="same"
        )

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)


# 构造优化器工具类，所有创建优化器的函数都放在这里
class MyOptimizers:
    @staticmethod
    def adam(lr):
        return tf.keras.optimizers.Adam(learning_rate=lr)
    

# 构造损失函数工具类，所有创建模型结构的函数都放在这里
class MyLosses:
    @staticmethod
    def sparse_categorical_crossentropy(from_logits=True):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)


class DatasetUtils:
    # 获取文件夹下所有文件的路径列表
    @staticmethod
    def load_all_paths(data_root):
        return [str(path) for path in list(pathlib.Path(data_root).glob("*/*"))]

    def create_path_dataset(self, data_root):
        paths = self.load_all_paths(data_root)
        return tf.data.Dataset.from_tensor_slices(paths)

    @staticmethod
    def load_and_preprocess_image(path: str, resized_shape: list):
        image = tf.io.read_file(path)
        if path.endswith(".jpg"):
            image = tf.image.decode_jpeg(image)
        elif path.endswith(".png"):
            image = tf.image.decode_png(image)
        elif path.endswith(".bmp"):
            image = tf.image.decode_bmp(image)
        else:
            raise Exception("不支持此数据格式")
        image = tf.image.resize(image, resized_shape)
        image /= 255.0
        return image

    def create_dataset(self, data_root):
        path_dataset = self.create_path_dataset(data_root)
        dataset = path_dataset.map(self.load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset


# 反射调用所有组件，创建并编译模型，返回的model可以直接train
def create_model(model_name, optimizer_name, loss_name, metrics):
    """
    model_name: 模型结构 String 与模型工具类中函数名保持一致
    optimizer_name: 优化器 String 与优化器工具类中函数名保持一致
    loss_name: 损失函数 String 与损失函数工具类中函数名保持一致
    metrics: 评估指标 List 例 ["accuracy"]
    """
    model = getattr(MyModels, model_name)
    optimizer = getattr(MyOptimizers, optimizer_name)
    loss = getattr(MyLosses, loss_name)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
    

