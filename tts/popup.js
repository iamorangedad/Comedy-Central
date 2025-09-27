// 页面加载时获取当前设置
document.addEventListener('DOMContentLoaded', function() {
    loadSettings();
});

// 加载设置
function loadSettings() {
    chrome.runtime.sendMessage({ action: "getSettings" }, function(response) {
        if (response) {
            document.getElementById('voice').value = response.voice || 'Kore';
        }
    });
}

// 保存设置
document.getElementById('settingsForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const voice = document.getElementById('voice').value;
    
    // 保存设置
    chrome.runtime.sendMessage({
        action: "saveSettings",
        voice: voice
    }, function(response) {
        if (response && response.success) {
            showStatus('语音设置保存成功！', 'success');
        } else {
            showStatus('设置保存失败', 'error');
        }
    });
});

// 显示状态信息
function showStatus(message, type) {
    const status = document.getElementById('status');
    status.textContent = message;
    status.className = `status ${type}`;
    status.style.display = 'block';
    
    // 3秒后隐藏状态信息
    setTimeout(() => {
        status.style.display = 'none';
    }, 3000);
}
