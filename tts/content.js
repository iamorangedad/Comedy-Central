// 内容脚本初始化
console.log("🎬 内容脚本已加载，准备接收消息");

// 内容脚本状态管理
let contentScriptReady = false;
let messageQueue = [];

// 标记内容脚本为准备状态
function markAsReady() {
    contentScriptReady = true;
    console.log("✅ 内容脚本已准备就绪");

    // 处理队列中的消息
    if (messageQueue.length > 0) {
        console.log(`📦 处理队列中的 ${messageQueue.length} 条消息`);
        messageQueue.forEach(msg => {
            processMessage(msg.message, msg.sender, msg.sendResponse);
        });
        messageQueue = [];
    }
}

// 处理消息的核心函数
function processMessage(message, sender, sendResponse) {
    console.log("📨 处理消息:", message.action, "发送者:", sender);

    try {
        // 处理ping消息，确认内容脚本已准备好
        if (message.action === "ping") {
            console.log("🏓 收到ping消息，内容脚本已准备好");
            sendResponse({ status: "ready", timestamp: Date.now() });
            markAsReady();
            return true;
        }

        // 播放音频
        if (message.action === "playAudio") {
            console.log("🎵 收到音频数据，开始播放");
            console.log("📊 音频数据信息:", {
                hasAudioData: !!message.audioData,
                dataLength: message.audioData ? message.audioData.length : 0,
                textLength: message.text ? message.text.length : 0,
                autoSave: message.autoSave,
                saveFormat: message.saveFormat
            });

            // 验证音频数据
            if (!message.audioData) {
                console.error("❌ 音频数据为空");
                sendResponse({ status: "error", error: "音频数据为空" });
                return;
            }

            // 如果启用自动保存，先保存文件
            if (message.autoSave) {
                console.log("💾 开始自动保存音频文件...");
                saveAudioFile(message.audioData, message.text, message.saveFormat);
            }

            playAudioFromBase64(message.audioData, message.text);
            sendResponse({ status: "success", message: "音频播放已开始" });
            return true;
        }

        // 显示错误信息
        if (message.action === "displayError") {
            console.error("❌ 收到错误信息：", message.error);
            showErrorModal(message.error);
            sendResponse({ status: "success", message: "错误信息已显示" });
            return true;
        }

        // 显示加载状态
        if (message.action === "showLoading") {
            console.log("⏳ 显示加载状态：", message.message);
            showLoadingModal(message.message);
            sendResponse({ status: "success", message: "加载状态已显示" });
            return true;
        }

        // 未知消息类型
        console.warn("⚠️ 未知消息类型:", message.action);
        sendResponse({ status: "unknown", message: "未知消息类型" });

    } catch (error) {
        console.error("💥 处理消息时发生错误:", error);
        sendResponse({ status: "error", error: error.message });
    }
}

// 监听来自后台脚本的消息
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    // 如果内容脚本还未准备好，将消息加入队列
    if (!contentScriptReady && message.action !== "ping") {
        console.log("⏳ 内容脚本未准备就绪，消息加入队列:", message.action);
        messageQueue.push({ message, sender, sendResponse });
        return true; // 保持消息通道开放
    }

    return processMessage(message, sender, sendResponse);
});

// 页面加载完成后标记为准备状态
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', markAsReady);
} else {
    // 页面已经加载完成
    setTimeout(markAsReady, 100);
}
// 将PCM数据转换为WAV格式
function convertPCMToWAV(pcmData, sampleRate = 24000, numChannels = 1, bitsPerSample = 16) {
    console.log("🔄 开始PCM到WAV转换:", {
        pcmLength: pcmData.length,
        sampleRate,
        numChannels,
        bitsPerSample
    });

    const dataLength = pcmData.length;
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);

    // WAV文件头 - RIFF chunk
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    let pos = 0;

    // RIFF header
    writeString(pos, 'RIFF'); pos += 4;
    view.setUint32(pos, 36 + dataLength, true); pos += 4; // File size - 8
    writeString(pos, 'WAVE'); pos += 4;

    // fmt chunk
    writeString(pos, 'fmt '); pos += 4;
    view.setUint32(pos, 16, true); pos += 4; // fmt chunk size
    view.setUint16(pos, 1, true); pos += 2; // Audio format (PCM = 1)
    view.setUint16(pos, numChannels, true); pos += 2; // Number of channels
    view.setUint32(pos, sampleRate, true); pos += 4; // Sample rate
    view.setUint32(pos, sampleRate * numChannels * bitsPerSample / 8, true); pos += 4; // Byte rate
    view.setUint16(pos, numChannels * bitsPerSample / 8, true); pos += 2; // Block align
    view.setUint16(pos, bitsPerSample, true); pos += 2; // Bits per sample

    // data chunk
    writeString(pos, 'data'); pos += 4;
    view.setUint32(pos, dataLength, true); pos += 4; // Data size

    // 复制PCM数据
    const uint8View = new Uint8Array(buffer);
    uint8View.set(pcmData, pos);

    console.log("✅ PCM到WAV转换完成，WAV大小:", buffer.byteLength, "bytes");
    return buffer;
}
// 获取音频错误消息
function getAudioErrorMessage(errorCode) {
    const errorMessages = {
        1: "MEDIA_ERR_ABORTED - 音频播放被中止",
        2: "MEDIA_ERR_NETWORK - 网络错误",
        3: "MEDIA_ERR_DECODE - 音频解码错误",
        4: "MEDIA_ERR_SRC_NOT_SUPPORTED - 不支持的音频格式"
    };
    return errorMessages[errorCode] || `未知错误 (代码: ${errorCode})`;
}
// 改进的保存音频文件函数
function saveAudioFile(base64Data, text, format = 'wav') {
    try {
        console.log("💾 开始保存音频文件，格式:", format);

        // 生成文件名
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const textPreview = text.substring(0, 30).replace(/[^\w\s]/g, '').replace(/\s+/g, '_');
        const filename = `ai_tts_${textPreview}_${timestamp}.${format}`;

        console.log("📁 文件名:", filename);

        // 将Base64数据转换为PCM
        const binaryString = atob(base64Data);
        const pcmData = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            pcmData[i] = binaryString.charCodeAt(i);
        }

        let blob;
        let mimeType;

        if (format === 'wav') {
            // 转换为WAV格式
            const wavBuffer = convertPCMToWAV(pcmData, 24000, 1, 16);
            blob = new Blob([wavBuffer], { type: 'audio/wav' });
            mimeType = 'audio/wav';
        } else {
            // 其他格式暂时不支持，默认保存为PCM
            blob = new Blob([pcmData], { type: 'application/octet-stream' });
            mimeType = 'application/octet-stream';
        }

        console.log("📊 音频文件信息:", {
            filename: filename,
            size: blob.size,
            mimeType: mimeType,
            format: format
        });

        // 创建下载链接
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.style.display = 'none';

        // 触发下载
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        // 清理URL
        setTimeout(() => {
            URL.revokeObjectURL(url);
        }, 1000);

        console.log("✅ 音频文件保存成功:", filename);
        showSaveSuccessNotification(filename);

    } catch (error) {
        console.error("❌ 音频文件保存失败:", error);
        showErrorModal("音频文件保存失败: " + error.message);
    }
}
// 创建WAV文件头
function createWavHeader(dataLength, sampleRate, channels, bitsPerSample) {
    const header = new ArrayBuffer(44);
    const view = new DataView(header);

    // RIFF header
    view.setUint32(0, 0x46464952, false); // "RIFF"
    view.setUint32(4, 36 + dataLength, true); // File size - 8
    view.setUint32(8, 0x45564157, false); // "WAVE"

    // fmt chunk
    view.setUint32(12, 0x20746d66, false); // "fmt "
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, 1, true); // Audio format (PCM)
    view.setUint16(22, channels, true); // Number of channels
    view.setUint32(24, sampleRate, true); // Sample rate
    view.setUint32(28, sampleRate * channels * bitsPerSample / 8, true); // Byte rate
    view.setUint16(32, channels * bitsPerSample / 8, true); // Block align
    view.setUint16(34, bitsPerSample, true); // Bits per sample

    // data chunk
    view.setUint32(36, 0x61746164, false); // "data"
    view.setUint32(40, dataLength, true); // Data size

    return new Uint8Array(header);
}

// 显示保存成功通知
function showSaveSuccessNotification(filename) {
    const notification = document.createElement('div');
    notification.id = 'ai-tts-save-notification';
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        left: 20px;
        background: #28a745;
        color: white;
        padding: 12px 16px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        z-index: 10001;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 14px;
        max-width: 300px;
        animation: slideIn 0.3s ease-out;
    `;

    notification.innerHTML = `
        <div style="display: flex; align-items: center; gap: 8px;">
            <span>💾</span>
            <div>
                <div style="font-weight: 600;">音频文件已保存</div>
                <div style="font-size: 12px; opacity: 0.9;">${filename}</div>
            </div>
        </div>
    `;

    // 添加动画样式
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(-100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
    `;
    document.head.appendChild(style);

    document.body.appendChild(notification);

    // 3秒后自动隐藏
    setTimeout(() => {
        if (notification.parentNode) {
            notification.style.animation = 'slideIn 0.3s ease-out reverse';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.remove();
                }
            }, 300);
        }
    }, 3000);
}

// 播放Base64编码的音频数据 - 修复版本
function playAudioFromBase64(base64Data, text) {
    console.log("🎵 开始处理音频数据");
    console.log("📊 输入参数:", {
        hasBase64Data: !!base64Data,
        dataLength: base64Data ? base64Data.length : 0,
        textLength: text ? text.length : 0
    });

    // 存储Base64数据供手动保存使用
    window.currentAudioBase64 = base64Data;
    window.currentAudioText = text;

    try {
        // 隐藏加载状态
        console.log("🔄 隐藏加载状态");
        hideLoadingModal();

        // 将Base64数据转换为Blob
        console.log("🔄 开始Base64解码...");
        const binaryString = atob(base64Data);
        console.log("✅ Base64解码完成，二进制字符串长度:", binaryString.length);

        const pcmData = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            pcmData[i] = binaryString.charCodeAt(i);
        }
        console.log("✅ PCM字节数组创建完成，长度:", pcmData.length);

        // 转换PCM为WAV格式
        console.log("🔄 转换PCM为WAV格式...");
        const wavData = convertPCMToWAV(pcmData, 24000, 1, 16);
        console.log("✅ WAV转换完成，数据长度:", wavData.length);

        // 创建WAV音频Blob
        console.log("🔄 创建WAV音频Blob...");
        const audioBlob = new Blob([wavData], { type: 'audio/wav' });
        console.log("✅ WAV Blob创建完成，大小:", audioBlob.size, "bytes");

        // 创建音频URL
        console.log("🔄 创建音频URL...");
        const audioUrl = URL.createObjectURL(audioBlob);
        console.log("✅ 音频URL创建完成:", audioUrl);

        // 创建音频元素并播放
        console.log("🔄 创建音频元素...");
        const audio = new Audio();
        audio.preload = 'auto';
        audio.crossOrigin = 'anonymous';

        // 添加详细的事件监听器
        audio.addEventListener('loadstart', () => console.log("🔄 音频开始加载"));
        audio.addEventListener('loadedmetadata', () => {
            console.log("✅ 音频元数据加载完成，时长:", audio.duration, "秒");
        });
        audio.addEventListener('loadeddata', () => console.log("✅ 音频数据加载完成"));
        audio.addEventListener('canplay', () => console.log("✅ 音频可以播放"));
        audio.addEventListener('canplaythrough', () => console.log("✅ 音频可以流畅播放"));
        audio.addEventListener('error', (e) => {
            console.error("❌ 音频错误:", e);
            console.error("❌ 音频错误代码:", audio.error?.code);
            console.error("❌ 音频错误消息:", audio.error?.message);
            showErrorModal(`音频播放失败: ${getAudioErrorMessage(audio.error?.code)}`);
        });

        audio.src = audioUrl;
        console.log("✅ 音频元素创建完成");

        // 显示播放控制界面
        console.log("🔄 显示播放控制界面...");
        showAudioPlayer(audio, text);
        console.log("✅ 播放控制界面显示完成");

        // 播放音频
        console.log("🔄 开始播放音频...");
        audio.play().then(() => {
            console.log("✅ 音频开始播放成功");
        }).catch((error) => {
            console.error("❌ 音频播放失败:", error);
            showErrorModal("音频播放失败: " + error.message + "\n请检查音频数据格式是否正确");
        });

        // 清理URL对象
        audio.addEventListener('ended', () => {
            console.log("🏁 音频播放结束，清理资源");
            URL.revokeObjectURL(audioUrl);
            hideAudioPlayer();
        });

        // 添加播放失败的备用处理
        setTimeout(() => {
            if (audio.readyState === 0) {
                console.warn("⚠️ 音频5秒后仍未开始加载，可能存在问题");
                showErrorModal("音频加载超时，请检查音频数据是否正确");
            }
        }, 5000);

    } catch (error) {
        console.error("💥 音频处理失败:", error);
        console.error("📋 错误详情:", {
            name: error.name,
            message: error.message,
            stack: error.stack
        });
        showErrorModal("音频处理失败: " + error.message);
    }
}

// 显示音频播放器界面
function showAudioPlayer(audio, text) {
    // 移除已存在的播放器
    const existingPlayer = document.getElementById('ai-tts-player');
    if (existingPlayer) {
        existingPlayer.remove();
    }

    // 创建播放器容器
    const player = document.createElement('div');
    player.id = 'ai-tts-player';
    player.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: white;
        border: 2px solid #007bff;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        z-index: 10000;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        min-width: 300px;
        max-width: 400px;
    `;

    // 创建标题
    const title = document.createElement('h4');
    title.textContent = '🎵 AI语音播放';
    title.style.cssText = `
        margin: 0 0 12px 0;
        color: #007bff;
        font-size: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    `;

    // 创建文本预览
    const textPreview = document.createElement('div');
    textPreview.textContent = text.length > 100 ? text.substring(0, 100) + '...' : text;
    textPreview.style.cssText = `
        background: #f8f9fa;
        padding: 8px;
        border-radius: 6px;
        font-size: 14px;
        color: #555;
        margin-bottom: 12px;
        line-height: 1.4;
        max-height: 60px;
        overflow-y: auto;
    `;

    // 创建控制按钮容器
    const controls = document.createElement('div');
    controls.style.cssText = `
        display: flex;
        gap: 8px;
        align-items: center;
    `;

    // 播放/暂停按钮
    const playPauseBtn = document.createElement('button');
    playPauseBtn.innerHTML = '⏸️';
    playPauseBtn.style.cssText = `
        background: #007bff;
        color: white;
        border: none;
        padding: 8px 12px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 14px;
    `;

    // 停止按钮
    const stopBtn = document.createElement('button');
    stopBtn.innerHTML = '⏹️';
    stopBtn.style.cssText = `
        background: #6c757d;
        color: white;
        border: none;
        padding: 8px 12px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 14px;
    `;

    // 保存按钮
    const saveBtn = document.createElement('button');
    saveBtn.innerHTML = '💾';
    saveBtn.title = '保存音频文件';
    saveBtn.style.cssText = `
        background: #28a745;
        color: white;
        border: none;
        padding: 8px 12px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 14px;
    `;

    // 关闭按钮
    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '✕';
    closeBtn.style.cssText = `
        background: #dc3545;
        color: white;
        border: none;
        padding: 8px 12px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 14px;
        margin-left: auto;
    `;

    // 进度条
    const progressContainer = document.createElement('div');
    progressContainer.style.cssText = `
        margin: 8px 0;
    `;

    const progressBar = document.createElement('div');
    progressBar.style.cssText = `
        width: 100%;
        height: 4px;
        background: #e9ecef;
        border-radius: 2px;
        overflow: hidden;
    `;

    const progressFill = document.createElement('div');
    progressFill.style.cssText = `
        height: 100%;
        background: #007bff;
        width: 0%;
        transition: width 0.1s ease;
    `;

    progressBar.appendChild(progressFill);
    progressContainer.appendChild(progressBar);

    // 时间显示
    const timeDisplay = document.createElement('div');
    timeDisplay.style.cssText = `
        font-size: 12px;
        color: #666;
        text-align: center;
        margin-top: 4px;
    `;
    timeDisplay.textContent = '0:00 / 0:00';

    // 组装播放器
    controls.appendChild(playPauseBtn);
    controls.appendChild(stopBtn);
    controls.appendChild(saveBtn);
    controls.appendChild(closeBtn);

    player.appendChild(title);
    player.appendChild(textPreview);
    player.appendChild(progressContainer);
    player.appendChild(timeDisplay);
    player.appendChild(controls);

    document.body.appendChild(player);

    // 事件监听
    playPauseBtn.addEventListener('click', () => {
        if (audio.paused) {
            audio.play();
            playPauseBtn.innerHTML = '⏸️';
        } else {
            audio.pause();
            playPauseBtn.innerHTML = '▶️';
        }
    });

    stopBtn.addEventListener('click', () => {
        audio.pause();
        audio.currentTime = 0;
        playPauseBtn.innerHTML = '▶️';
        progressFill.style.width = '0%';
        updateTimeDisplay(0, audio.duration);
    });

    saveBtn.addEventListener('click', () => {
        console.log("💾 用户手动保存音频文件");
        // 获取当前音频的Base64数据
        if (window.currentAudioBase64 && window.currentAudioText) {
            saveAudioFile(window.currentAudioBase64, window.currentAudioText, 'wav');
        } else {
            showErrorModal("无法获取音频数据，请重新生成");
        }
    });

    closeBtn.addEventListener('click', () => {
        audio.pause();
        hideAudioPlayer();
    });

    // 更新进度条
    audio.addEventListener('timeupdate', () => {
        if (audio.duration) {
            const progress = (audio.currentTime / audio.duration) * 100;
            progressFill.style.width = progress + '%';
            updateTimeDisplay(audio.currentTime, audio.duration);
        }
    });

    // 音频结束
    audio.addEventListener('ended', () => {
        playPauseBtn.innerHTML = '▶️';
        progressFill.style.width = '100%';
    });

    // 更新时间显示
    function updateTimeDisplay(current, duration) {
        const formatTime = (time) => {
            const minutes = Math.floor(time / 60);
            const seconds = Math.floor(time % 60);
            return `${minutes}:${seconds.toString().padStart(2, '0')}`;
        };
        timeDisplay.textContent = `${formatTime(current)} / ${formatTime(duration)}`;
    }
}

// 隐藏音频播放器
function hideAudioPlayer() {
    const player = document.getElementById('ai-tts-player');
    if (player) {
        player.remove();
    }
}

// 显示加载状态
function showLoadingModal(message) {
    console.log("⏳ 显示加载状态:", message);

    // 移除已存在的加载状态
    const existingLoading = document.getElementById('ai-tts-loading');
    if (existingLoading) {
        console.log("🔄 移除已存在的加载状态");
        existingLoading.remove();
    }

    const loading = document.createElement('div');
    loading.id = 'ai-tts-loading';
    loading.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: white;
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        z-index: 10000;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        min-width: 250px;
    `;

    const content = document.createElement('div');
    content.style.cssText = `
        display: flex;
        align-items: center;
        gap: 12px;
        color: #28a745;
        font-size: 14px;
    `;

    const spinner = document.createElement('div');
    spinner.style.cssText = `
        width: 20px;
        height: 20px;
        border: 2px solid #e9ecef;
        border-top: 2px solid #28a745;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    `;

    // 添加旋转动画
    const style = document.createElement('style');
    style.textContent = `
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    `;
    document.head.appendChild(style);

    content.appendChild(spinner);
    content.appendChild(document.createTextNode(message));
    loading.appendChild(content);

    document.body.appendChild(loading);
    console.log("✅ 加载状态显示完成");
}

// 隐藏加载状态
function hideLoadingModal() {
    console.log("🔄 隐藏加载状态");
    const loading = document.getElementById('ai-tts-loading');
    if (loading) {
        loading.remove();
        console.log("✅ 加载状态已隐藏");
    } else {
        console.log("ℹ️ 没有找到加载状态元素");
    }
}

// 显示错误信息
function showErrorModal(error) {
    console.log("❌ 显示错误信息:", error);

    // 隐藏加载状态
    hideLoadingModal();

    const modal = document.createElement('div');
    modal.id = 'ai-tts-error-modal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        z-index: 10000;
        display: flex;
        justify-content: center;
        align-items: center;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    `;

    const modalContent = document.createElement('div');
    modalContent.style.cssText = `
        background: white;
        padding: 20px;
        border-radius: 8px;
        max-width: 400px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        position: relative;
    `;

    const title = document.createElement('h3');
    title.textContent = '❌ 错误';
    title.style.cssText = `
        margin: 0 0 15px 0;
        color: #dc3545;
        font-size: 18px;
    `;

    const content = document.createElement('div');
    content.textContent = error;
    content.style.cssText = `
        line-height: 1.6;
        color: #555;
        margin-bottom: 20px;
    `;

    const closeBtn = document.createElement('button');
    closeBtn.textContent = '关闭';
    closeBtn.style.cssText = `
        background: #dc3545;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 14px;
    `;

    closeBtn.onclick = () => modal.remove();

    modalContent.appendChild(title);
    modalContent.appendChild(content);
    modalContent.appendChild(closeBtn);
    modal.appendChild(modalContent);

    modal.onclick = (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    };

    document.body.appendChild(modal);
}

// 音频兼容性检测
function checkAudioSupport() {
    const audio = new Audio();
    const formats = {
        wav: audio.canPlayType('audio/wav'),
        mp3: audio.canPlayType('audio/mpeg'),
        ogg: audio.canPlayType('audio/ogg'),
        webm: audio.canPlayType('audio/webm')
    };

    console.log("🔊 浏览器音频格式支持:", formats);
    return formats;
}

// 在页面加载时检测音频支持
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', checkAudioSupport);
} else {
    checkAudioSupport();
}