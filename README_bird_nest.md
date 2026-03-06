# 鸟巢检测实例分割项目

本项目使用Mask R-CNN进行电网上鸟巢的实例分割检测。

## 数据集格式

数据集使用COCO格式的JSON标注文件，位于 `Datasets/my_coco_dataset/` 目录下：
- `dataset.json`: COCO格式的标注文件
- `*.jpg`: 图像文件

## 文件说明

1. **windows_utils.py**: 包含数据集类定义
   - `StudentIDDataset`: 原有的学生ID数据集类（保留）
   - `COCODataset`: 新的COCO格式数据集类，用于加载鸟巢数据集

2. **train_bird_nest.py**: 训练脚本
   - 加载COCO格式的鸟巢数据集
   - 训练Mask R-CNN模型
   - 自动保存最佳模型检查点

3. **visualize_bird_nest.py**: 可视化脚本
   - 加载训练好的模型
   - 在图像上可视化预测结果
   - 对比真实标注和预测结果

4. **可视化报告**
   - 可视化图表

## 使用方法

### 0. 转换COCO格式数据集

# 输出目录
Datasets/my_coco_dataset

# 运行转换
# 'path/to/your/labelme_jsons' 应同时包含图像和.json 文件
labelme2coco Datasets/original Datasets/my_coco_dataset

# 这将创建./my_coco_dataset/annotations.json
# 并将所有图像复制到./my_coco_dataset/images/
# 运行convert脚本增加负样本

### 1. 训练模型

运行训练脚本：

```bash
python train_bird_nest.py
```

训练参数可以在脚本中修改：
- `train_sz`: 训练图像尺寸（默认512）
- `bs`: 批次大小（默认4）
- `epochs`: 训练轮数（默认40）
- `lr`: 学习率（默认5e-4）

训练完成后，模型会保存在 `pytorch-mask-r-cnn-bird-nest/[timestamp]/` 目录下。

### 2. 可视化结果

运行可视化脚本：

```bash
python visualize_bird_nest.py
```

脚本会：
- 自动查找最新的模型检查点
- 在所有有标注的图像上运行预测
- 生成对比可视化结果（左侧真实标注，右侧预测结果）
- 保存可视化结果到 `bird_nest_visualization.png`

### 3. 在Jupyter Notebook中使用

你也可以在Jupyter Notebook中使用这些功能。主要步骤：

1. 导入必要的模块：
```python
from windows_utils import COCODataset, tuple_batch
from train_bird_nest import *  # 导入训练相关函数
from visualize_bird_nest import *  # 导入可视化函数
```

2. 加载数据集并训练（参考 `train_bird_nest.py`）

3. 可视化结果（参考 `visualize_bird_nest.py`）

## 数据集结构

COCO格式的JSON文件应包含以下字段：

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "1a.jpg",
      "width": 8688,
      "height": 5792
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [x, y, width, height],
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 447473
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "bird_nest"
    }
  ]
}
```

## 注意事项

1. **图像路径**: 如果JSON中的图像路径不正确，代码会自动在指定目录中查找同名图像文件。

2. **类别映射**: 确保 `class_to_idx` 字典正确映射了类别名称到索引（0是背景类）。

3. **GPU内存**: 如果遇到GPU内存不足，可以：
   - 减小 `train_sz`（如改为448或416）
   - 减小 `bs`（批次大小）
   - 减少数据增强的复杂度

4. **训练时间**: 根据数据集大小和硬件配置，训练可能需要较长时间。建议先用少量epoch测试。

## 输出文件

训练完成后会生成：
- `maskrcnn_resnet50_fpn_v2.pth`: 模型权重文件
- `training_metadata.json`: 训练元数据（损失、学习率等）
- `bird_nest-colormap.json`: 类别颜色映射

可视化会生成：
- `bird_nest_visualization.png`: 所有图像的可视化结果

## 依赖库

确保安装了以下库：
- torch, torchvision
- PIL (Pillow)
- numpy
- matplotlib
- tqdm
- distinctipy
- cjm_pytorch_utils, cjm_pil_utils, cjm_torchvision_tfms

