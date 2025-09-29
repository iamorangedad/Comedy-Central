# 图标设置说明

## 需要创建的图标文件

扩展需要以下三个PNG格式的图标文件：

- `icon16.png` (16x16像素)
- `icon48.png` (48x48像素) 
- `icon128.png` (128x128像素)

## 转换方法

### 方法1: 使用在线工具

1. 访问 [Convertio](https://convertio.co/svg-png/) 或 [CloudConvert](https://cloudconvert.com/svg-to-png)
2. 上传 `icon.svg` 文件
3. 设置输出尺寸为 16x16, 48x48, 128x128
4. 下载转换后的PNG文件
5. 重命名为对应的文件名

### 方法2: 使用命令行工具

如果你安装了 ImageMagick：

```bash
# 转换16x16图标
convert icon.svg -resize 16x16 icon16.png

# 转换48x48图标
convert icon.svg -resize 48x48 icon48.png

# 转换128x128图标
convert icon.svg -resize 128x128 icon128.png
```

### 方法3: 使用设计软件

1. 使用 Photoshop, GIMP, 或 Figma 打开 `icon.svg`
2. 导出为PNG格式，分别设置不同的尺寸
3. 保存为对应的文件名

## 临时解决方案

如果暂时无法创建图标文件，你可以：

1. 从网上下载任意16x16, 48x48, 128x128的PNG图标
2. 重命名为 `icon16.png`, `icon48.png`, `icon128.png`
3. 放在 `tts` 目录下

扩展仍然可以正常工作，只是图标可能不是自定义的。

## 验证图标

创建图标后，确保：
- 文件格式为PNG
- 文件尺寸正确
- 文件在 `tts` 目录下
- 文件名完全匹配（区分大小写）
