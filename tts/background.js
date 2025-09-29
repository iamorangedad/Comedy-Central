// 在插件安装时创建右键菜单项
chrome.runtime.onInstalled.addListener(() => {
    console.log("🚀 TTS扩展安装/更新完成，创建右键菜单");
    chrome.contextMenus.create({
        id: "ttsText",
        title: "AI语音播放选中文本",
        contexts: ["selection"] // 只在用户选中文字时显示
    });
    // 新增：AI总结并播放菜单
    chrome.contextMenus.create({
        id: "summarizeAndTTS",
        title: "AI总结并语音播放",
        contexts: ["selection"]
    });
    console.log("✅ 右键菜单创建成功");
});

// 监听右键菜单点击事件
chrome.contextMenus.onClicked.addListener((info, tab) => {
    // 检查是否有选中的文本
    if (!info.selectionText || info.selectionText.trim().length === 0) {
        console.log("❌ 没有选中文本");
        return;
    }

    if (info.menuItemId === "ttsText") {
        console.log("🎯 TTS菜单被点击");
        console.log("📋 Tab信息:", { id: tab.id, url: tab.url, title: tab.title });
        console.log("📝 选中的文本长度:", info.selectionText.length);
        console.log("📝 选中的文本内容:", info.selectionText.substring(0, 100) + (info.selectionText.length > 100 ? "..." : ""));

        // 确保内容脚本已加载，然后调用TTS API
        console.log("🔧 开始确保内容脚本准备就绪...");
        ensureContentScriptReady(tab.id, () => {
            console.log("✅ 内容脚本准备完成，开始调用TTS API");
            callTTSAPI(tab.id, info.selectionText);
        });
    } else if (info.menuItemId === "summarizeAndTTS") {
        console.log("🎯 AI总结并播放菜单被点击");
        console.log("📋 Tab信息:", { id: tab.id, url: tab.url, title: tab.title });
        console.log("📝 选中的文本长度:", info.selectionText.length);

        ensureContentScriptReady(tab.id, () => {
            console.log("✅ 内容脚本准备完成，开始AI总结流程");
            summarizeAndSpeak(tab.id, info.selectionText);
        });
    }
});

// 确保内容脚本已准备好接收消息
function ensureContentScriptReady(tabId, callback, retryCount = 0) {
    const maxRetries = 5;
    const baseDelay = 300;

    console.log(`🔍 检查内容脚本状态，Tab ID: ${tabId}, 尝试 ${retryCount + 1}/${maxRetries + 1}`);

    // 先尝试发送一个ping消息
    chrome.tabs.sendMessage(tabId, { action: "ping" }, (response) => {
        if (chrome.runtime.lastError) {
            console.log("⚠️ 内容脚本未响应，错误:", chrome.runtime.lastError.message);

            // 如果还有重试次数，尝试重新注入内容脚本
            if (retryCount < maxRetries) {
                console.log("🔄 开始重新注入内容脚本...");

                chrome.scripting.executeScript({
                    target: { tabId: tabId },
                    files: ['content.js']
                }, () => {
                    if (chrome.runtime.lastError) {
                        console.error("❌ 内容脚本注入失败:", chrome.runtime.lastError.message);
                        // 继续重试
                        const delay = baseDelay * Math.pow(1.5, retryCount);
                        console.log(`⏳ 等待 ${delay}ms 后重试注入...`);
                        setTimeout(() => {
                            ensureContentScriptReady(tabId, callback, retryCount + 1);
                        }, delay);
                        return;
                    }

                    console.log("✅ 内容脚本注入成功，等待初始化...");
                    // 等待内容脚本初始化，然后重试ping
                    const delay = baseDelay * Math.pow(1.2, retryCount);
                    setTimeout(() => {
                        console.log("🔄 重试ping消息...");
                        ensureContentScriptReady(tabId, callback, retryCount + 1);
                    }, delay);
                });
            } else {
                console.error("💥 内容脚本准备最终失败，已用尽所有重试次数");
                callback();
            }
        } else {
            // 内容脚本已准备好
            console.log("✅ 内容脚本已准备好，响应:", response);
            callback();
        }
    });
}

// 可靠的消息发送函数
function sendMessageToContentScript(tabId, message, retryCount = 0) {
    const maxRetries = 5;
    const baseDelay = 200;

    console.log(`📤 发送消息到内容脚本 (尝试 ${retryCount + 1}/${maxRetries + 1}):`, {
        action: message.action,
        tabId: tabId,
        hasData: !!message.audioData,
        dataSize: message.audioData ? message.audioData.length : 0
    });

    // 检查tab是否仍然有效
    chrome.tabs.get(tabId, (tab) => {
        if (chrome.runtime.lastError) {
            console.error("❌ Tab不存在或已关闭:", chrome.runtime.lastError.message);
            return;
        }

        const startTime = Date.now();
        chrome.tabs.sendMessage(tabId, message, (response) => {
            const responseTime = Date.now() - startTime;

            if (chrome.runtime.lastError) {
                console.error(`❌ 发送消息到内容脚本失败 (尝试 ${retryCount + 1}/${maxRetries + 1}):`, chrome.runtime.lastError.message);
                console.error("📋 错误详情:", {
                    error: chrome.runtime.lastError.message,
                    action: message.action,
                    tabId: tabId,
                    retryCount: retryCount,
                    responseTime: `${responseTime}ms`
                });

                // 更新统计
                updateMessageStats(false, responseTime);

                // 如果还有重试次数，使用指数退避策略
                if (retryCount < maxRetries) {
                    messageStats.retries++;
                    const delay = baseDelay * Math.pow(2, retryCount) + Math.random() * 100; // 添加随机抖动
                    console.log(`⏳ 等待 ${Math.round(delay)}ms 后重试...`);

                    // 对于音频数据，如果重试次数较多，尝试重新确保内容脚本准备
                    if (retryCount >= 2 && message.action === "playAudio") {
                        console.log("🔄 音频消息发送失败，重新确保内容脚本准备...");
                        ensureContentScriptReady(tabId, () => {
                            setTimeout(() => {
                                sendMessageToContentScript(tabId, message, retryCount + 1);
                            }, delay);
                        });
                    } else {
                        setTimeout(() => {
                            sendMessageToContentScript(tabId, message, retryCount + 1);
                        }, delay);
                    }
                } else {
                    console.error("💥 消息发送最终失败，已用尽所有重试次数");
                    // 如果是音频播放失败，显示错误信息
                    if (message.action === "playAudio") {
                        chrome.tabs.sendMessage(tabId, {
                            action: "displayError",
                            error: "音频播放失败：无法与内容脚本通信"
                        });
                    }
                }
            } else {
                console.log("✅ 消息发送成功:", message.action, "响应:", response, `响应时间: ${responseTime}ms`);
                updateMessageStats(true, responseTime);
            }
        });
    });
}

// 连接健康检查
const connectionHealth = new Map();

// 消息发送统计
const messageStats = {
    totalSent: 0,
    successful: 0,
    failed: 0,
    retries: 0,
    averageResponseTime: 0
};

function updateMessageStats(success, responseTime = 0) {
    messageStats.totalSent++;
    if (success) {
        messageStats.successful++;
        messageStats.averageResponseTime =
            (messageStats.averageResponseTime * (messageStats.successful - 1) + responseTime) / messageStats.successful;
    } else {
        messageStats.failed++;
    }

    console.log("📊 消息发送统计:", {
        totalSent: messageStats.totalSent,
        successful: messageStats.successful,
        failed: messageStats.failed,
        successRate: `${((messageStats.successful / messageStats.totalSent) * 100).toFixed(1)}%`,
        averageResponseTime: `${messageStats.averageResponseTime.toFixed(0)}ms`
    });
}

function checkConnectionHealth(tabId) {
    return new Promise((resolve) => {
        const startTime = Date.now();
        chrome.tabs.sendMessage(tabId, { action: "ping" }, (response) => {
            const responseTime = Date.now() - startTime;
            const isHealthy = !chrome.runtime.lastError && response && response.status === "ready";

            connectionHealth.set(tabId, {
                isHealthy,
                lastCheck: Date.now(),
                responseTime,
                error: chrome.runtime.lastError ? chrome.runtime.lastError.message : null
            });

            console.log(`🏥 连接健康检查 - Tab ${tabId}:`, {
                isHealthy,
                responseTime: `${responseTime}ms`,
                error: chrome.runtime.lastError ? chrome.runtime.lastError.message : null
            });

            resolve(isHealthy);
        });
    });
}

// 调用Gemini API 进行相声剧本制作
async function scriptText(text) {
    console.log("🤖 开始调用Gemini API进行剧本制作");
    console.log("📝 输入文本长度:", text.length);

    const apiKey = "TODO"; // 使用相同的API Key
    const apiUrl = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent";

    // 限制输入文本长度
    const maxInputLength = 15000;
    const textToSummarize = text.length > maxInputLength ? text.substring(0, maxInputLength) : text;

    const prompt = `请将以下内容改写为相声的形式，格式为张三和李四的对话，且只保留对话内容，去除开头和结尾的介绍：\n\n${textToSummarize}`;

    try {
        const requestBody = {
            contents: [
                {
                    parts: [
                        {
                            text: prompt
                        }
                    ]
                }
            ],
            generationConfig: {
                temperature: 0.7,
                topK: 40,
                topP: 0.95,
                maxOutputTokens: 500,
            }
        };

        console.log("📦 发送剧本制作请求到Gemini API...");

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
        console.log("📥 Gemini API 返回数据结构:", {
            hasCandidates: !!data.candidates,
            candidatesCount: data.candidates ? data.candidates.length : 0
        });

        if (!data.candidates || !data.candidates[0] || !data.candidates[0].content || !data.candidates[0].content.parts) {
            console.error("❌ API返回数据格式不正确:", data);
            throw new Error("API返回数据格式不正确");
        }

        const summaryText = data.candidates[0].content.parts[0].text;
        console.log("✅ 剧本制作完成，剧本长度:", summaryText.length);
        console.log("📄 剧本内容:", summaryText);

        return summaryText;

    } catch (error) {
        console.error("💥 剧本制作失败:", error);
        // 发送错误信息到内容脚本
        chrome.tabs.sendMessage(tabId, {
            action: "displayError",
            error: `剧本制作失败: ${error.message}`
        });
    }
}

// 调用Gemini TTS API的函数
async function callTTSAPI(tabId, text) {
    console.log("🎤 开始TTS API调用流程");
    console.log("📋 输入参数:", { tabId, textLength: text.length });

    // 先进行连接健康检查
    console.log("🏥 进行连接健康检查...");
    const isHealthy = await checkConnectionHealth(tabId);
    if (!isHealthy) {
        console.log("⚠️ 连接不健康，重新确保内容脚本准备...");
        await new Promise((resolve) => {
            ensureContentScriptReady(tabId, resolve);
        });
    }

    // 从storage获取API密钥，如果没有则使用默认值
    console.log("🔑 获取存储设置...");
    const result = await chrome.storage.sync.get(['geminiApiKey', 'selectedVoice']);
    console.log("💾 存储设置:", result);

    // const apiKey = result.geminiApiKey || "todo";
    const apiKey = "TODO";
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

        // 检查是否启用自动保存
        const saveSettings = await chrome.storage.sync.get(['autoSaveAudio', 'saveFormat']);
        const autoSave = saveSettings.autoSaveAudio !== false; // 默认启用
        const saveFormat = saveSettings.saveFormat || 'wav'; // 默认WAV格式

        console.log("💾 自动保存设置:", { autoSave, saveFormat });

        // 将音频数据发送到内容脚本进行播放
        sendMessageToContentScript(tabId, {
            action: "playAudio",
            audioData: audioData,
            text: textToSpeak,
            autoSave: false,
            saveFormat: saveFormat
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

// 调用Gemini TTS API的双人对话函数
async function callDualVoiceTTSAPI(tabId, text) {
    console.log("🎭 开始双人对话TTS API调用流程");
    console.log("📋 输入参数:", { tabId, textLength: text.length });

    // 先进行连接健康检查
    console.log("🏥 进行连接健康检查...");
    const isHealthy = await checkConnectionHealth(tabId);
    if (!isHealthy) {
        console.log("⚠️ 连接不健康，重新确保内容脚本准备...");
        await new Promise((resolve) => {
            ensureContentScriptReady(tabId, resolve);
        });
    }

    // 从storage获取API密钥和语音配置
    console.log("🔑 获取存储设置...");
    const result = await chrome.storage.sync.get([
        'geminiApiKey',
        'voice1',
        'voice2',
        'voiceName1',
        'voiceName2'
    ]);
    console.log("💾 存储设置:", result);

    const apiKey = "TODO";
    // 默认使用两个不同的预设语音
    const voice1 = result.voice1 || "Puck";  // 第一个说话者（男声）
    const voice2 = result.voice2 || "Charon"; // 第二个说话者（女声）
    const voiceName1 = result.voiceName1 || "张三";
    const voiceName2 = result.voiceName2 || "李四";

    console.log("🎵 选择的语音配置:", {
        voice1,
        voice2,
        voiceName1,
        voiceName2
    });

    // 检查文本长度
    if (!text || text.trim().length === 0) {
        console.log("❌ 文本内容无效");
        sendMessageToContentScript(tabId, {
            action: "displayError",
            error: "没有有效的文本内容"
        });
        return;
    }

    // 限制文本长度
    const maxLength = 8000;
    const textToSpeak = text.length > maxLength ? text.substring(0, maxLength) + "..." : text;
    const prompt = `请生成以下对话的音频：\n\n${textToSpeak}`;

    console.log("📝 处理后的文本长度:", textToSpeak.length);
    if (text.length > maxLength) {
        console.log("⚠️ 文本被截断，原长度:", text.length, "截断后长度:", textToSpeak.length);
    }

    const apiUrl = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent";
    console.log("🌐 API URL:", apiUrl);

    try {
        console.log("🚀 开始调用双人对话TTS API...");

        // 发送加载状态
        console.log("📤 发送加载状态到内容脚本");
        sendMessageToContentScript(tabId, {
            action: "showLoading",
            message: "正在生成双人对话语音..."
        });

        // 构建双人对话的请求体
        // 使用speaker标记来指定不同的说话者
        const requestBody = {
            contents: [
                {
                    parts: [
                        {
                            text: prompt
                        }
                    ]
                }
            ],
            generationConfig: {
                responseModalities: ["AUDIO"],
                speechConfig: {
                    // 使用官方的 multiSpeakerVoiceConfig 格式
                    multiSpeakerVoiceConfig: {
                        speakerVoiceConfigs: [
                            {
                                speaker: voiceName1,
                                voiceConfig: {
                                    prebuiltVoiceConfig: {
                                        voiceName: voice1
                                    }
                                }
                            },
                            {
                                speaker: voiceName2,
                                voiceConfig: {
                                    prebuiltVoiceConfig: {
                                        voiceName: voice2
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        };

        console.log("📦 请求体:", {
            textLength: textToSpeak.length,
            voice1: voice1,
            voice2: voice2,
            voiceName1: voiceName1,
            voiceName2: voiceName2,
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
        console.log("📥 双人对话TTS API 返回数据:", {
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

        console.log("🎵 双人对话音频数据获取成功，数据长度:", audioData.length);
        console.log("📤 准备发送音频数据到内容脚本...");

        // 检查是否启用自动保存
        const saveSettings = await chrome.storage.sync.get(['autoSaveAudio', 'saveFormat']);
        const autoSave = saveSettings.autoSaveAudio !== false;
        const saveFormat = saveSettings.saveFormat || 'wav';

        console.log("💾 自动保存设置:", { autoSave, saveFormat });

        // 将音频数据发送到内容脚本进行播放
        sendMessageToContentScript(tabId, {
            action: "playAudio",
            audioData: audioData,
            text: textToSpeak,
            autoSave: false,
            saveFormat: saveFormat,
            isDualVoice: true  // 标记为双人对话音频
        });

    } catch (error) {
        console.error("💥 双人对话TTS API调用失败:", error);
        console.error("📋 错误详情:", {
            name: error.name,
            message: error.message,
            stack: error.stack
        });

        // 发送错误信息到内容脚本
        console.log("📤 发送错误信息到内容脚本");
        sendMessageToContentScript(tabId, {
            action: "displayError",
            error: `双人对话语音生成失败: ${error.message}`
        });
    }
}

// 总结并播放功能
async function summarizeAndSpeak(tabId, text) {
    console.log("🎯 开始AI总结并播放流程");

    try {
        // 先进行连接健康检查
        console.log("🏥 进行连接健康检查...");
        const isHealthy = await checkConnectionHealth(tabId);
        if (!isHealthy) {
            console.log("⚠️ 连接不健康，重新确保内容脚本准备...");
            await new Promise((resolve) => {
                ensureContentScriptReady(tabId, resolve);
            });
        }

        // 显示"正在总结"状态
        sendMessageToContentScript(tabId, {
            action: "showLoading",
            message: "正在AI总结内容..."
        });

        // 调用Gemini进行剧本制作
        console.log("🤖 步骤1: 调用Gemini进行剧本制作");
        const summary = await scriptText(text);

        // 显示剧本内容（可选）
        console.log("📢 显示剧本内容给用户");
        sendMessageToContentScript(tabId, {
            action: "displaySummary",
            summary: summary
        });

        // 更新状态为"正在生成语音"
        sendMessageToContentScript(tabId, {
            action: "showLoading",
            message: "正在生成语音..."
        });

        // 调用TTS API将总结转为语音
        console.log("🎤 步骤2: 将总结内容转为语音");
        // await callTTSAPI(tabId, summary);
        await callDualVoiceTTSAPI(tabId, summary);

        console.log("✅ AI总结并播放流程完成");

    } catch (error) {
        console.error("💥 AI总结并播放失败:", error);
        sendMessageToContentScript(tabId, {
            action: "displayError",
            error: `AI总结失败: ${error.message}`
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

    if (request.action === "getDebugInfo") {
        // 获取调试信息
        sendResponse({
            messageStats: messageStats,
            connectionHealth: Object.fromEntries(connectionHealth),
            timestamp: Date.now()
        });
        return true;
    }
});
