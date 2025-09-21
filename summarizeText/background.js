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
        console.log("右键菜单被点击");
        
        // 检查是否有选中的文本
        if (!info.selectionText || info.selectionText.trim().length === 0) {
            console.log("没有选中文本");
            return;
        }
        
        console.log("选中的文本:", info.selectionText);
        
        // 直接使用选中的文本调用API，避免消息传递的复杂性
        callLLMAPI(tab.id, info.selectionText);
    }
});


// 注意：现在直接通过右键菜单的 info.selectionText 获取选中文本，
// 不再需要通过消息传递，简化了代码逻辑

// 调用大模型API的函数
// API文档: https://ai.google.dev/gemini-api/docs/text-generation
async function callLLMAPI(tabId, text) {
    // 1. 从 storage 中获取 API 密钥，或者直接在代码中 hardcode (不推荐)
    const apiKey = "AIzaSyCtRQ_1cl3sfPxcj91rZESm7rOFRq5RsFg";
    
    // 检查文本长度
    if (!text || text.trim().length === 0) {
        chrome.tabs.sendMessage(tabId, {
            action: "displayError",
            error: "没有有效的文本内容"
        });
        return;
    }

    // 限制文本长度，避免API调用过长
    const maxLength = 8000;
    const textToSummarize = text.length > maxLength ? text.substring(0, maxLength) + "..." : text;
    
    const prompt = `请用中文总结以下文本，要求简洁明了，突出重点：\n\n${textToSummarize}`;
    const apiUrl = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent";

    try {
        console.log("开始调用API，文本长度:", textToSummarize.length);
        
        const response = await fetch(apiUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "x-goog-api-key": apiKey
            },
            body: JSON.stringify({
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
                    maxOutputTokens: 1024,
                }
            })
        });

        if (!response.ok) {
            throw new Error(`API请求失败: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        console.log("API 返回数据:", data);
        
        // 检查API返回的数据结构
        if (!data.candidates || !data.candidates[0] || !data.candidates[0].content || !data.candidates[0].content.parts) {
            throw new Error("API返回数据格式不正确");
        }
        
        const summary = data.candidates[0].content.parts[0].text;
        
        if (!summary || summary.trim().length === 0) {
            throw new Error("API返回的总结内容为空");
        }

        // 将总结结果发送回内容脚本
        chrome.tabs.sendMessage(tabId, {
            action: "displaySummary",
            summary: summary.trim()
        });

    } catch (error) {
        console.error("API调用失败:", error);
        
        // 发送错误信息到内容脚本
        chrome.tabs.sendMessage(tabId, {
            action: "displayError",
            error: `总结失败: ${error.message}`
        });
    }
}