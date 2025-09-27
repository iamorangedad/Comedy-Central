// 内容脚本初始化
console.log("🎬 内容脚本已加载，准备接收消息");

// 监听来自后台脚本的消息
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log("📨 收到消息:", message.action, "发送者:", sender);
    
    // 处理ping消息，确认内容脚本已准备好
    if (message.action === "ping") {
        console.log("🏓 收到ping消息，内容脚本已准备好");
        sendResponse({ status: "ready" });
        return true;
    }

    // 播放音频
    if (message.action === "playAudio") {
        console.log("🎵 收到音频数据，开始播放");
        console.log("📊 音频数据信息:", {
            hasAudioData: !!message.audioData,
            dataLength: message.audioData ? message.audioData.length : 0,
            textLength: message.text ? message.text.length : 0
        });
        playAudioFromBase64(message.audioData, message.text);
    }

    // 显示错误信息
    if (message.action === "displayError") {
        console.error("❌ 收到错误信息：", message.error);
        showErrorModal(message.error);
    }

    // 显示加载状态
    if (message.action === "showLoading") {
        console.log("⏳ 显示加载状态：", message.message);
        showLoadingModal(message.message);
    }
});

// 播放Base64编码的音频数据
function playAudioFromBase64(base64Data, text) {
    console.log("🎵 开始处理音频数据");
    console.log("📊 输入参数:", {
        hasBase64Data: !!base64Data,
        dataLength: base64Data ? base64Data.length : 0,
        textLength: text ? text.length : 0
    });
    
    try {
        // 隐藏加载状态
        console.log("🔄 隐藏加载状态");
        hideLoadingModal();
        
        // 将Base64数据转换为Blob
        console.log("🔄 开始Base64解码...");
        const binaryString = atob(base64Data);
        console.log("✅ Base64解码完成，二进制字符串长度:", binaryString.length);
        
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        console.log("✅ 字节数组创建完成，长度:", bytes.length);
        
        // 创建PCM音频Blob (24kHz, 16-bit, mono)
        console.log("🔄 创建音频Blob...");
        const audioBlob = new Blob([bytes], { type: 'audio/pcm' });
        console.log("✅ 音频Blob创建完成，大小:", audioBlob.size, "bytes");
        
        // 创建音频URL
        console.log("🔄 创建音频URL...");
        const audioUrl = URL.createObjectURL(audioBlob);
        console.log("✅ 音频URL创建完成:", audioUrl);
        
        // 创建音频元素并播放
        console.log("🔄 创建音频元素...");
        const audio = new Audio();
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
            showErrorModal("音频播放失败: " + error.message);
        });
        
        // 清理URL对象
        audio.addEventListener('ended', () => {
            console.log("🏁 音频播放结束，清理资源");
            URL.revokeObjectURL(audioUrl);
            hideAudioPlayer();
        });
        
        // 添加其他事件监听器用于调试
        audio.addEventListener('loadstart', () => console.log("🔄 音频开始加载"));
        audio.addEventListener('canplay', () => console.log("✅ 音频可以播放"));
        audio.addEventListener('error', (e) => console.error("❌ 音频错误:", e));
        
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
