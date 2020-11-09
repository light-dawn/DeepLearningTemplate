import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import config
from collections import Counter

"""
说明
1. 所有的工具类方法都可以通过反射的方式调用
"""

tf.compat.v1.enable_eager_execution()

class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filter_num,(3,3),strides=stride,padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')

        self.conv2 = tf.keras.layers.Conv2D(filter_num,(3,3),strides=1,padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        if stride!=1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filter_num,(1,1),strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self,input,training=None):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(input)
        output = tf.keras.layers.add([out,identity])
        output = tf.nn.relu(output)
        return output


class ResNet(tf.keras.Model):
    def __init__(self, layer_dims, num_classes=10):
        super(ResNet, self).__init__()
        # 预处理层
        self.stem = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), strides=(1,1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(1,1), padding='same')
        ])
        # resblock
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        # there are [b, 512, h, w]
        # 自适应
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, input, training=None):
        x = self.stem(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # [b,c]
        x = self.avgpool(x)
        x = self.fc(x)
        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = tf.keras.Sequential()
        # may down sample
        res_blocks.add(BasicBlock(filter_num, stride))
        # just down sample one time
        for pre in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks


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

    @staticmethod
    def resnet18(output_channels):
        return ResNet([2, 2, 2, 2], num_classes=output_channels)


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


# 提供通用的文件操作和数据集构建
class DatasetUtils:
    def __init__(self, data_root, image_size):
        self.data_root = data_root
        self.image_size = image_size

    @staticmethod
    def read_csv(csv):
        with open(csv, "r") as f:
            result = [line.strip() for line in f.readlines()][1:]
        return result

    # 获取文件夹下所有文件的路径列表
    def load_all_paths(self):
        return [str(path) for path in list(pathlib.Path(self.data_root).glob("*/*"))]

    # 递归地扫描文件夹下的所有目标类型数据
    def load_all_paths_recursive(self, data_types):
        result = []
        for data_type in data_types:
            print("数据类型: ", data_type)
            result.extend([str(path).strip("\n") for path in list(pathlib.Path(self.data_root).rglob("*."+data_type))])
        return sorted(result)

    def get_label_names(self):
        label_names = sorted(item.name for item in pathlib.Path(self.data_root).glob("*/") if item.is_dir())
        return label_names

    def create_path_dataset(self):
        paths = self.load_all_paths_recursive(data_types=["png"])
        print("路径: ", paths[:10])
        return tf.data.Dataset.from_tensor_slices(paths)

    def load_and_preprocess_image(self, path):
        # sess = tf.Session()
        # with sess.as_default():
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, self.image_size)
        image /= 255.0
        return image

    def create_dataset(self):
        path_dataset = self.create_path_dataset()
        dataset = path_dataset.map(self.load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    def create_labelset(self, data_root, desc_csv):
        label_names = sorted(item.name for item in data_root.glob("*/") if item.is_dir())
        print("标签名称: ", label_names)
        label_to_index = dict((name, index) for index, name in enumerate(label_names))
        print("标签索引: ", label_to_index)

    def create_image_dataset_from_directory(self, image_size, batch_size, mode="training", seed=0):
        return tf.keras.preprocessing.image_dataset_from_directory(
            self.data_root, validation_split=0.2, subset=mode, seed=seed, 
            image_size=image_size, batch_size=batch_size
        )


class CC_CCII_DatasetUtils(DatasetUtils):
    def __init__(self, data_root, image_size):
        super().__init__(data_root, image_size)
        self.data_nums = None
    
    def load_all_labels_specific(self, desc_csv):
        all_paths = self.load_all_paths_recursive(data_types=["png"])
        self.data_nums = len(all_paths)
        label_names = self.get_label_names()
        print("标签名称: ", label_names)
        label_to_index = dict((name, index) for index, name in enumerate(label_names))
        print("标签索引: ", label_to_index)

        # 控制NC图像的数量
        nc_limit = 3000
        nc_count = 0
        paths_in_csv = self.read_csv(desc_csv)
        # print("在CSV中的路径: ", repr(paths_in_csv[0]))
        all_labels = []
        for path in all_paths:
            if "/".join(path.split("/")[-4:]) in paths_in_csv:
                # print("在CSV中找到: ", repr("/".join(path.split("/")[-4:])))
                all_labels.append(path.split("/")[-4])
            else:
                if nc_count >= nc_limit:
                    continue
                nc_count += 1
                all_labels.append("NC")
            # return
        
        print(set(all_labels))
        return [label_to_index.get(label) for label in all_labels]

    def create_labelset(self):
        label_set = tf.data.Dataset.from_tensor_slices(tf.cast(self.load_all_labels_specific(config.DESC_CSV), tf.int64))
        return label_set

    def create_data_and_label_set(self):
        return tf.data.Dataset.zip((self.create_dataset(), self.create_labelset()))

    def process_dataset(self, batch_size):
        dataset = self.create_data_and_label_set()
        dataset = dataset.shuffle(buffer_size=self.data_nums)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    
# 反射调用所有组件，创建并编译模型，返回的model可以直接train
def create_model(model_name, optimizer_name, loss_name, metrics):
    """
    model_name: 模型结构 String 与模型工具类中函数名保持一致
    optimizer_name: 优化器 String 与优化器工具类中函数名保持一致
    loss_name: 损失函数 String 与损失函数工具类中函数名保持一致
    metrics: 评估指标 List 例 ["accuracy"]
    """
    model = getattr(MyModels, model_name)(output_channels=3)
    optimizer = getattr(MyOptimizers, optimizer_name)(lr=1e-2)
    loss = getattr(MyLosses, loss_name)()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


if __name__ == "__main__":
    data_utils = CC_CCII_DatasetUtils(config.DATA_ROOT, (224, 224))
    data_utils.create_path_dataset()
    

