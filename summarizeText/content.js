// 监听来自后台脚本的消息
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    // 接收后台脚本发来的总结结果
    if (message.action === "displaySummary") {
        console.log("总结结果：\n" + message.summary);
        displaySummaryModal(message.summary);
    }

    // 接收错误信息
    if (message.action === "displayError") {
        console.error("错误信息：\n" + message.error);
        displayErrorModal(message.error);
    }
});

// 显示总结结果的模态框
function displaySummaryModal(summary) {
    // 移除已存在的模态框
    const existingModal = document.getElementById('ai-summarizer-modal');
    if (existingModal) {
        existingModal.remove();
    }

    // 创建模态框
    const modal = document.createElement('div');
    modal.id = 'ai-summarizer-modal';
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
        max-width: 500px;
        max-height: 400px;
        overflow-y: auto;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        position: relative;
    `;

    const title = document.createElement('h3');
    title.textContent = 'AI 文本总结';
    title.style.cssText = `
        margin: 0 0 15px 0;
        color: #333;
        font-size: 18px;
    `;

    const content = document.createElement('div');
    content.textContent = summary;
    content.style.cssText = `
        line-height: 1.6;
        color: #555;
        margin-bottom: 20px;
    `;

    const closeBtn = document.createElement('button');
    closeBtn.textContent = '关闭';
    closeBtn.style.cssText = `
        background: #007bff;
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

    // 点击背景关闭模态框
    modal.onclick = (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    };

    document.body.appendChild(modal);
}

// 显示错误信息的模态框
function displayErrorModal(error) {
    const modal = document.createElement('div');
    modal.id = 'ai-summarizer-error-modal';
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
    title.textContent = '错误';
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