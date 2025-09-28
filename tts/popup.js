// 页面加载时获取当前设置
document.addEventListener('DOMContentLoaded', function() {
    loadSettings();
    loadSaveSettings();
    loadDebugInfo();
});

// 加载设置
function loadSettings() {
    chrome.runtime.sendMessage({ action: "getSettings" }, function(response) {
        if (response) {
            document.getElementById('voice').value = response.voice || 'Kore';
        }
    });
}

// 加载保存设置
function loadSaveSettings() {
    chrome.storage.sync.get(['autoSaveAudio', 'saveFormat'], function(result) {
        const autoSave = result.autoSaveAudio !== false; // 默认启用
        const saveFormat = result.saveFormat || 'wav';
        
        document.getElementById('autoSave').checked = autoSave;
        document.getElementById('saveFormat').value = saveFormat;
        
        // 根据自动保存状态显示/隐藏格式选择
        toggleSaveFormatVisibility(autoSave);
    });
}

// 切换保存格式可见性
function toggleSaveFormatVisibility(autoSave) {
    const saveFormatGroup = document.getElementById('saveFormatGroup');
    if (saveFormatGroup) {
        saveFormatGroup.style.display = autoSave ? 'block' : 'none';
    }
}

// 保存设置
document.getElementById('settingsForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const voice = document.getElementById('voice').value;
    const autoSave = document.getElementById('autoSave').checked;
    const saveFormat = document.getElementById('saveFormat').value;
    
    // 保存语音设置
    chrome.runtime.sendMessage({
        action: "saveSettings",
        voice: voice
    }, function(response) {
        if (response && response.success) {
            // 保存自动保存设置
            chrome.storage.sync.set({
                autoSaveAudio: autoSave,
                saveFormat: saveFormat
            }, function() {
                showStatus('所有设置保存成功！', 'success');
            });
        } else {
            showStatus('设置保存失败', 'error');
        }
    });
});

// 监听自动保存复选框变化
document.getElementById('autoSave').addEventListener('change', function() {
    const autoSave = this.checked;
    toggleSaveFormatVisibility(autoSave);
});

// 加载调试信息
function loadDebugInfo() {
    chrome.runtime.sendMessage({ action: "getDebugInfo" }, function(response) {
        if (response) {
            updateDebugDisplay(response);
        }
    });
}

// 更新调试信息显示
function updateDebugDisplay(debugInfo) {
    const debugContainer = document.getElementById('debugInfo');
    if (!debugContainer) return;
    
    const stats = debugInfo.messageStats;
    const health = debugInfo.connectionHealth;
    
    const successRate = stats.totalSent > 0 ? ((stats.successful / stats.totalSent) * 100).toFixed(1) : 0;
    
    debugContainer.innerHTML = `
        <div class="debug-section">
            <h4>📊 消息统计</h4>
            <div class="debug-item">
                <span>总发送:</span> <strong>${stats.totalSent}</strong>
            </div>
            <div class="debug-item">
                <span>成功:</span> <strong style="color: #28a745;">${stats.successful}</strong>
            </div>
            <div class="debug-item">
                <span>失败:</span> <strong style="color: #dc3545;">${stats.failed}</strong>
            </div>
            <div class="debug-item">
                <span>成功率:</span> <strong>${successRate}%</strong>
            </div>
            <div class="debug-item">
                <span>平均响应时间:</span> <strong>${stats.averageResponseTime.toFixed(0)}ms</strong>
            </div>
            <div class="debug-item">
                <span>重试次数:</span> <strong>${stats.retries}</strong>
            </div>
        </div>
        
        <div class="debug-section">
            <h4>🏥 连接健康</h4>
            ${Object.keys(health).length > 0 ? 
                Object.entries(health).map(([tabId, info]) => `
                    <div class="debug-item">
                        <span>Tab ${tabId}:</span> 
                        <strong style="color: ${info.isHealthy ? '#28a745' : '#dc3545'}">
                            ${info.isHealthy ? '健康' : '不健康'}
                        </strong>
                        <span style="font-size: 12px; color: #666;">
                            (${info.responseTime}ms)
                        </span>
                    </div>
                `).join('') : 
                '<div class="debug-item">暂无连接信息</div>'
            }
        </div>
    `;
}

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
