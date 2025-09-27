// 在插件安装时创建右键菜单项
chrome.runtime.onInstalled.addListener(() => {
    console.log("🚀 TTS扩展安装/更新完成，创建右键菜单");
    chrome.contextMenus.create({
        id: "ttsText",
        title: "AI语音播放选中文本",
        contexts: ["selection"] // 只在用户选中文字时显示
    });
    console.log("✅ 右键菜单创建成功");
});

// 监听右键菜单点击事件
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "ttsText") {
        console.log("🎯 TTS菜单被点击");
        console.log("📋 Tab信息:", { id: tab.id, url: tab.url, title: tab.title });
        
        // 检查是否有选中的文本
        if (!info.selectionText || info.selectionText.trim().length === 0) {
            console.log("❌ 没有选中文本");
            return;
        }
        
        console.log("📝 选中的文本长度:", info.selectionText.length);
        console.log("📝 选中的文本内容:", info.selectionText.substring(0, 100) + (info.selectionText.length > 100 ? "..." : ""));
        
        // 确保内容脚本已加载，然后调用TTS API
        console.log("🔧 开始确保内容脚本准备就绪...");
        ensureContentScriptReady(tab.id, () => {
            console.log("✅ 内容脚本准备完成，开始调用TTS API");
            callTTSAPI(tab.id, info.selectionText);
        });
    }
});

// 确保内容脚本已准备好接收消息
function ensureContentScriptReady(tabId, callback) {
    console.log("🔍 检查内容脚本状态，Tab ID:", tabId);
    
    // 先尝试发送一个ping消息
    chrome.tabs.sendMessage(tabId, { action: "ping" }, (response) => {
        if (chrome.runtime.lastError) {
            console.log("⚠️ 内容脚本未响应，错误:", chrome.runtime.lastError.message);
            console.log("🔄 开始重新注入内容脚本...");
            
            chrome.scripting.executeScript({
                target: { tabId: tabId },
                files: ['content.js']
            }, () => {
                if (chrome.runtime.lastError) {
                    console.error("❌ 内容脚本注入失败:", chrome.runtime.lastError.message);
                    callback();
                    return;
                }
                
                console.log("✅ 内容脚本注入成功，等待初始化...");
                // 等待一下让内容脚本初始化，然后重试
                setTimeout(() => {
                    console.log("🔄 重试ping消息...");
                    // 重试ping消息
                    chrome.tabs.sendMessage(tabId, { action: "ping" }, (retryResponse) => {
                        if (chrome.runtime.lastError) {
                            console.log("⚠️ 内容脚本注入后仍然无法连接:", chrome.runtime.lastError.message);
                        } else {
                            console.log("✅ 内容脚本重新注入成功，已响应ping");
                        }
                        callback();
                    });
                }, 200);
            });
        } else {
            // 内容脚本已准备好
            console.log("✅ 内容脚本已准备好，响应:", response);
            callback();
        }
    });
}

// 可靠的消息发送函数
function sendMessageToContentScript(tabId, message, retryCount = 0) {
    const maxRetries = 3;
    
    console.log(`📤 发送消息到内容脚本 (尝试 ${retryCount + 1}/${maxRetries + 1}):`, {
        action: message.action,
        tabId: tabId,
        hasData: !!message.audioData,
        dataSize: message.audioData ? message.audioData.length : 0
    });
    
    chrome.tabs.sendMessage(tabId, message, (response) => {
        if (chrome.runtime.lastError) {
            console.error(`❌ 发送消息到内容脚本失败 (尝试 ${retryCount + 1}/${maxRetries + 1}):`, chrome.runtime.lastError.message);
            
            // 如果还有重试次数，等待后重试
            if (retryCount < maxRetries) {
                const delay = 500 * (retryCount + 1);
                console.log(`⏳ 等待 ${delay}ms 后重试...`);
                setTimeout(() => {
                    sendMessageToContentScript(tabId, message, retryCount + 1);
                }, delay);
            } else {
                console.error("💥 消息发送最终失败，已用尽所有重试次数");
            }
        } else {
            console.log("✅ 消息发送成功:", message.action, "响应:", response);
        }
    });
}

// 调用Gemini TTS API的函数
async function callTTSAPI(tabId, text) {
    console.log("🎤 开始TTS API调用流程");
    console.log("📋 输入参数:", { tabId, textLength: text.length });
    
    // 从storage获取API密钥，如果没有则使用默认值
    console.log("🔑 获取存储设置...");
    const result = await chrome.storage.sync.get(['geminiApiKey', 'selectedVoice']);
    console.log("💾 存储设置:", result);
    
    // const apiKey = result.geminiApiKey || "AIzaSyCtRQ_1cl3sfPxcj91rZESm7rOFRq5RsFg";
    const apiKey = "AIzaSyCtRQ_1cl3sfPxcj91rZESm7rOFRq5RsFg";
    const selectedVoice = result.selectedVoice || "Kore";
    
    // 使用固定的API密钥进行测试
    console.log("🔑 使用固定API密钥进行TTS测试");
    console.log("🎵 选择的语音:", selectedVoice);
    
    // 检查文本长度
    if (!text || text.trim().length === 0) {
        console.log("❌ 文本内容无效");
        sendMessageToContentScript(tabId, {
            action: "displayError",
            error: "没有有效的文本内容"
        });
        return;
    }

    // 限制文本长度，避免API调用过长
    const maxLength = 8000;
    const textToSpeak = text.length > maxLength ? text.substring(0, maxLength) + "..." : text;
    
    console.log("📝 处理后的文本长度:", textToSpeak.length);
    if (text.length > maxLength) {
        console.log("⚠️ 文本被截断，原长度:", text.length, "截断后长度:", textToSpeak.length);
    }
    
    const apiUrl = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent";
    console.log("🌐 API URL:", apiUrl);

    try {
        console.log("🚀 开始调用TTS API...");
        
        // 发送加载状态
        console.log("📤 发送加载状态到内容脚本");
        sendMessageToContentScript(tabId, {
            action: "showLoading",
            message: "正在生成语音..."
        });
        
        const requestBody = {
            contents: [
                {
                    parts: [
                        {
                            text: textToSpeak
                        }
                    ]
                }
            ],
            generationConfig: {
                responseModalities: ["AUDIO"],
                speechConfig: {
                    voiceConfig: {
                        prebuiltVoiceConfig: {
                            voiceName: selectedVoice
                        }
                    }
                }
            }
        };
        
        console.log("📦 请求体:", {
            textLength: textToSpeak.length,
            voice: selectedVoice,
            responseModalities: requestBody.generationConfig.responseModalities
        });
        
        const response = await fetch(apiUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "x-goog-api-key": apiKey
            },
            body: JSON.stringify(requestBody)
        });

        console.log("📡 HTTP响应状态:", response.status, response.statusText);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error("❌ API请求失败，响应内容:", errorText);
            throw new Error(`API请求失败: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        console.log("📥 TTS API 返回数据:", {
            hasCandidates: !!data.candidates,
            candidatesCount: data.candidates ? data.candidates.length : 0,
            modelVersion: data.modelVersion,
            responseId: data.responseId,
            usageMetadata: data.usageMetadata
        });
        
        // 检查API返回的数据结构
        console.log("🔍 验证API返回数据结构...");
        if (!data.candidates || !data.candidates[0] || !data.candidates[0].content || !data.candidates[0].content.parts) {
            console.error("❌ API返回数据格式不正确:", data);
            throw new Error("API返回数据格式不正确");
        }
        
        const candidate = data.candidates[0];
        const part = candidate.content.parts[0];
        
        console.log("📊 候选数据详情:", {
            hasContent: !!candidate.content,
            partsCount: candidate.content.parts.length,
            hasInlineData: !!part.inlineData,
            dataMimeType: part.inlineData ? part.inlineData.mimeType : 'none'
        });
        
        const audioData = part.inlineData.data;
        
        if (!audioData) {
            console.error("❌ API返回的音频数据为空");
            throw new Error("API返回的音频数据为空");
        }

        console.log("🎵 音频数据获取成功，数据长度:", audioData.length);
        console.log("📤 准备发送音频数据到内容脚本...");

        // 将音频数据发送到内容脚本进行播放
        sendMessageToContentScript(tabId, {
            action: "playAudio",
            audioData: audioData,
            text: textToSpeak
        });

    } catch (error) {
        console.error("💥 TTS API调用失败:", error);
        console.error("📋 错误详情:", {
            name: error.name,
            message: error.message,
            stack: error.stack
        });
        
        // 发送错误信息到内容脚本
        console.log("📤 发送错误信息到内容脚本");
        sendMessageToContentScript(tabId, {
            action: "displayError",
            error: `语音生成失败: ${error.message}`
        });
    }
}

// 监听来自popup的消息
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "saveSettings") {
        // 保存设置到storage
        chrome.storage.sync.set({
            selectedVoice: request.voice
        }, () => {
            sendResponse({ success: true });
        });
        return true; // 保持消息通道开放
    }
    
    if (request.action === "getSettings") {
        // 获取设置
        chrome.storage.sync.get(['selectedVoice'], (result) => {
            sendResponse({
                voice: result.selectedVoice || "Kore"
            });
        });
        return true; // 保持消息通道开放
    }
});
