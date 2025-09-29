# AI Text Summarizer Chrome Extension

一个Chrome浏览器扩展，可以选中网页上的文本并使用AI进行智能总结。

## 功能特点

- 🎯 选中任意网页文本，右键即可总结
- 🤖 使用Google Gemini AI模型进行智能总结
- 💬 友好的模态框界面显示结果
- ⚡ 快速响应，支持长文本处理
- 🛡️ 完善的错误处理机制

## 安装和使用

### 1. 加载扩展

1. 打开 Chrome 浏览器，在地址栏输入 `chrome://extensions` 并回车
2. 打开右上角的 "开发者模式" 开关
3. 点击 "加载已解压的扩展程序" 按钮
4. 选择 `summarizeText` 项目文件夹

### 2. 使用方法

1. 在任意网页上选中一段文本
2. 右键点击选中的文本
3. 选择 "使用AI总结选中文本" 菜单项
4. 等待AI处理，结果会在模态框中显示

## 技术实现

- **Manifest V3**: 使用最新的Chrome扩展API
- **Context Menus**: 右键菜单集成
- **Content Scripts**: 页面内容交互
- **Background Service Worker**: 后台API调用
- **Google Gemini API**: AI文本总结服务

## 文件结构

```
summarizeText/
├── manifest.json      # 扩展配置文件
├── background.js      # 后台服务脚本
├── content.js         # 内容脚本
├── icon.png          # 扩展图标
└── README.md         # 说明文档
```

## 注意事项

- 需要有效的Google Gemini API密钥
- 建议在测试环境中使用
- 长文本会自动截断到8000字符以内
- 支持中文和英文文本总结

## 故障排除

如果遇到问题，请检查：

1. Chrome开发者工具的控制台是否有错误信息
2. API密钥是否有效
3. 网络连接是否正常
4. 扩展是否正确加载

## 开发说明

- 使用 `console.log()` 进行调试
- 可以在 `chrome://extensions` 页面点击扩展的"检查视图"查看日志
- 修改代码后需要重新加载扩展