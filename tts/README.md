# AI Text-to-Speech Chrome Extension

一个Chrome浏览器扩展，可以选中网页上的文本并使用Gemini AI生成语音进行播放。

## 功能特点

- 🎯 **选中文本播放**: 选中任意网页文本，右键即可生成语音
- 🤖 **AI语音生成**: 使用Google Gemini 2.5 TTS模型
- 🎵 **丰富语音选项**: 支持30种不同的语音风格
- 🌍 **多语言支持**: 自动检测语言，支持24种语言
- 🎮 **播放控制**: 完整的音频播放控制界面
- ⚙️ **个性化设置**: 可配置API密钥和语音偏好

## 安装方法

### 1. 准备图标文件

由于扩展需要PNG格式的图标，请将 `icon.svg` 转换为以下尺寸的PNG文件：
- `icon16.png` (16x16像素)
- `icon48.png` (48x48像素) 
- `icon128.png` (128x128像素)

你可以使用在线工具如 [Convertio](https://convertio.co/svg-png/) 或 [CloudConvert](https://cloudconvert.com/svg-to-png) 进行转换。

### 2. 加载扩展

1. 打开Chrome浏览器，在地址栏输入 `chrome://extensions`
2. 打开右上角的"开发者模式"开关
3. 点击"加载已解压的扩展程序"按钮
4. 选择 `tts` 文件夹

### 3. 配置API密钥

1. 点击扩展图标，打开设置页面
2. 输入你的Gemini API密钥
3. 选择喜欢的语音风格
4. 点击"保存设置"

## 使用方法

1. 在任意网页上选中一段文本
2. 右键点击选中的文本
3. 选择"AI语音播放选中文本"菜单项
4. 等待AI生成语音
5. 使用播放控制界面控制音频播放

## 支持的语音选项

| 语音名称 | 风格 | 语音名称 | 风格 |
|---------|------|---------|------|
| Kore | Firm | Puck | Upbeat |
| Zephyr | Bright | Fenrir | Excitable |
| Leda | Youthful | Charon | Informative |
| Orus | Firm | Aoede | Breezy |
| Callirrhoe | Easy-going | Autonoe | Bright |
| Enceladus | Breathy | Iapetus | Clear |
| Umbriel | Easy-going | Algieba | Smooth |
| Despina | Smooth | Erinome | Clear |
| Algenib | Gravelly | Rasalgethi | Informative |
| Laomedeia | Upbeat | Achernar | Soft |
| Alnilam | Firm | Schedar | Even |
| Gacrux | Mature | Pulcherrima | Forward |
| Achird | Friendly | Zubenelgenubi | Casual |
| Vindemiatrix | Gentle | Sadachbia | Lively |
| Sadaltager | Knowledgeable | Sulafat | Warm |

## 支持的语言

支持24种语言，包括：
- 中文 (简体/繁体)
- English (US/India)
- 日本語
- 한국어
- Español
- Français
- Deutsch
- 等等...

## 技术实现

- **Manifest V3**: 使用最新的Chrome扩展API
- **Gemini TTS API**: 调用Google Gemini 2.5 Flash Preview TTS模型
- **音频处理**: 支持PCM格式音频播放
- **用户界面**: 现代化的播放控制界面

## 文件结构

```
tts/
├── manifest.json      # 扩展配置文件
├── background.js      # 后台服务脚本
├── content.js         # 内容脚本
├── popup.html         # 设置页面
├── popup.js           # 设置页面脚本
├── icon.svg           # 图标源文件
└── README.md          # 说明文档
```

## 获取API密钥

1. 访问 [Google AI Studio](https://aistudio.google.com/)
2. 登录你的Google账户
3. 点击"Get API key"
4. 创建新的API密钥
5. 复制密钥到扩展设置中

## 注意事项

- 需要有效的Gemini API密钥
- 文本长度限制为8000字符
- 音频格式为24kHz, 16位, 单声道PCM
- 建议在测试环境中使用

## 故障排除

如果遇到问题：

1. **API密钥错误**: 检查密钥是否正确配置
2. **音频无法播放**: 检查浏览器音频权限
3. **扩展无法加载**: 确认所有文件都在正确位置
4. **网络错误**: 检查网络连接和API服务状态

## 开发说明

- 使用 `console.log()` 进行调试
- 可以在 `chrome://extensions` 页面查看扩展日志
- 修改代码后需要重新加载扩展

## 参考文档

- [Gemini API 语音生成文档](https://ai.google.dev/gemini-api/docs/speech-generation)
- [Chrome扩展开发文档](https://developer.chrome.com/docs/extensions/)
- [Google AI Studio](https://aistudio.google.com/)
