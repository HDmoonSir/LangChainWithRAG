/**
 * RAG Engine Chat UI - Client Side Logic
 * Vanilla JS with Fetch API for Streaming
 */

const dom_chatHistory = document.getElementById('chat-history');
const dom_queryInput = document.getElementById('query-input');
const dom_sendBtn = document.getElementById('send-btn');
const dom_statusIndicator = document.getElementById('status-indicator');

// Marked setup for markdown rendering
marked.setOptions({
    highlight: function(code, lang) {
        const str_language = hljs.getLanguage(lang) ? lang : 'plaintext';
        return hljs.highlight(code, { language: str_language }).value;
    },
    langPrefix: 'hljs language-'
});

/**
 * 새로운 메시지 버블을 생성하여 화면에 추가한다.
 */
function createMessageBubble(str_role, str_content = '') {
    const dom_wrapper = document.createElement('div');
    dom_wrapper.className = `message-wrapper ${str_role}`;
    
    const dom_bubble = document.createElement('div');
    dom_bubble.className = 'message';
    
    if (str_role === 'ai') {
        dom_bubble.innerHTML = marked.parse(str_content || '...');
    } else {
        dom_bubble.textContent = str_content;
    }
    
    dom_wrapper.appendChild(dom_bubble);
    dom_chatHistory.appendChild(dom_wrapper);
    scrollToBottom();
    
    return dom_bubble;
}

function scrollToBottom() {
    dom_chatHistory.scrollTop = dom_chatHistory.scrollHeight;
}

/**
 * 메인 채팅 처리 로직
 */
async function handleChatSubmission() {
    const str_query = dom_queryInput.value.trim();
    if (!str_query) return;

    // UI 상태 업데이트
    dom_queryInput.value = '';
    dom_queryInput.style.height = 'auto';
    dom_sendBtn.disabled = true;
    updateStatus('Thinking...', '#e3b341');

    // 사용자 질의 추가
    createMessageBubble('user', str_query);

    // AI 스트리밍용 버블 생성
    const dom_aiBubble = createMessageBubble('ai');
    let str_accumulatedResponse = '';

    try {
        const obj_response = await fetch('/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: str_query })
        });

        if (!obj_response.ok) throw new Error(`HTTP Error: ${obj_response.status}`);

        const obj_reader = obj_response.body.getReader();
        const obj_decoder = new TextDecoder();
        let str_buffer = '';

        while (true) {
            const { value, done } = await obj_reader.read();
            if (done) break;

            str_buffer += obj_decoder.decode(value, { stream: true });
            const list_lines = str_buffer.split('\n');
            
            // 마지막 줄이 미완성일 수 있으므로 버퍼에 보관
            str_buffer = list_lines.pop();

            for (const str_line of list_lines) {
                if (str_line.startsWith('data: ')) {
                    try {
                        const obj_payload = JSON.parse(str_line.substring(6));
                        
                        if (obj_payload.str_event === 'token') {
                            str_accumulatedResponse += obj_payload.dict_data.token;
                            dom_aiBubble.innerHTML = marked.parse(str_accumulatedResponse);
                            scrollToBottom();
                        } else if (obj_payload.str_event === 'finish') {
                            updateStatus('Ready', '#3fb950');
                        } else if (obj_payload.str_event === 'error') {
                            throw new Error(obj_payload.dict_data.message);
                        }
                    } catch (e) {
                        console.warn('Failed to parse SSE frame', e);
                    }
                }
            }
        }
    } catch (obj_error) {
        console.error('Chat error:', obj_error);
        dom_aiBubble.innerHTML = `<span style="color: var(--error);">Error: ${obj_error.message}</span>`;
        updateStatus('Error', '#f85149');
    } finally {
        dom_sendBtn.disabled = false;
        hljs.highlightAll();
    }
}

function updateStatus(str_text, str_color) {
    dom_statusIndicator.textContent = str_text;
    dom_statusIndicator.style.color = str_color;
}

// Event Listeners
dom_sendBtn.addEventListener('click', handleChatSubmission);

dom_queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleChatSubmission();
    }
});

dom_queryInput.addEventListener('input', () => {
    dom_queryInput.style.height = 'auto';
    dom_queryInput.style.height = dom_queryInput.scrollHeight + 'px';
});
