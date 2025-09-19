// 在插件安装时创建右键菜单项
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "summarizeText",
        title: "使用AI总结选中文本",
        contexts: ["selection"] // 只在用户选中文字时显示
    });
});

// 监听右键菜单点击事件
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "summarizeText") {
        // 菜单项被点击后，向内容脚本发送消息，请求选中的文本
        chrome.tabs.sendMessage(tab.id, { action: "summarize" }, (response) => {
            if (chrome.runtime.lastError) {
                console.error("消息发送失败:", chrome.runtime.lastError.message);
            } else {
                console.log("消息成功发送到内容脚本");
            }
        });
        console.log("点击事件被触发");
    }
});


// 监听内容脚本发来的消息（包含选中的文本）
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "getSelectedText") {
        const selectedText = request.text;
        // 获取发送消息的 tab ID
        const tabId = sender.tab.id;
        // 在这里调用大模型 API
        callLLMAPI(tabId, selectedText);
    }
});

// 调用大模型API的函数
// API文档: https://ai.google.dev/gemini-api/docs/text-generation
async function callLLMAPI(tabId, text) {
    // 1. 从 storage 中获取 API 密钥，或者直接在代码中 hardcode (不推荐)
    const apiKey = "AIzaSyCtRQ_1cl3sfPxcj91rZESm7rOFRq5RsFg";
    const prompt = `Please summarize the following text: ${text}`;
    const apiUrl = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"; // 替换为你的大模型API端点

    try {
        const response = await fetch(apiUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "x-goog-api-key": apiKey // 不同的API授权方式可能不同
            },
            body: JSON.stringify({
                contents: [
                    {
                        parts: [
                            {
                                text: prompt // 将您的提示词放在这里
                            }
                        ]
                    }
                ]
            })
        });

        const data = await response.json();
        console.log("API 返回数据:", data);
        const summary = data.candidates[0].content.parts[0].text; // 根据API返回结构修改

        // 2. 将总结结果发送回内容脚本或在弹出窗口中显示
        chrome.tabs.sendMessage(tabId, {
            action: "displaySummary",
            summary: summary
        });

    } catch (error) {
        console.error("API调用失败:", error);
    }
}