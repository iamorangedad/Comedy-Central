// 监听来自后台脚本的消息
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "summarize") {
        console.log("收到 summarize 指令");
        // 获取用户选中的文本
        const selectedText = window.getSelection().toString();
        console.log("被选中的内容:\n" + selectedText);
        if (selectedText.length > 0) {
            // 将选中的文本发送回后台脚本
            // chrome.runtime.sendMessage({
            //     action: "getSelectedText",
            //     text: selectedText
            // });
            sendResponse({ action: "getSelectedText", text: selectedText });
        } else {
            alert("请先选中一段文本！");
            sendResponse({ error: "没有选中文本" });
        }
        return true;// 如果有异步逻辑，这行很重要
    }

    // 接收后台脚本发来的总结结果
    if (message.action === "displaySummary") {
        console.log("总结结果：\n" + message.summary);
        alert("总结结果：\n" + message.summary);
    }
});