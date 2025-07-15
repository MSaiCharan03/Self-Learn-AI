// Chat Management
class ChatManager {
    constructor() {
        this.currentConversationId = null;
        this.conversations = new Map();
        this.isTyping = false;
        this.messageHistory = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadConversations();
    }

    setupEventListeners() {
        // Message form submission
        const chatForm = document.getElementById('chat-form');
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');

        chatForm.addEventListener('submit', this.handleMessageSubmit.bind(this));
        
        // Auto-resize textarea
        messageInput.addEventListener('input', this.handleInputChange.bind(this));
        
        // Handle Enter key
        messageInput.addEventListener('keydown', this.handleKeyDown.bind(this));
        
        // Sidebar toggle
        document.getElementById('sidebar-toggle').addEventListener('click', this.toggleSidebar.bind(this));
        document.getElementById('close-sidebar').addEventListener('click', this.closeSidebar.bind(this));
        
        // New chat button
        document.getElementById('new-chat-btn').addEventListener('click', this.startNewChat.bind(this));
        
        // Conversation selection
        document.addEventListener('click', this.handleConversationClick.bind(this));
        
        // Enable input when authenticated
        document.addEventListener('user-authenticated', this.enableChat.bind(this));
    }

    // Handle message form submission
    async handleMessageSubmit(event) {
        event.preventDefault();
        
        const messageInput = document.getElementById('message-input');
        const message = messageInput.value.trim();
        
        if (!message || this.isTyping) return;
        
        // Clear input
        messageInput.value = '';
        this.adjustTextareaHeight(messageInput);
        
        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Send message to API
        await this.sendMessage(message);
    }

    // Handle input changes
    handleInputChange(event) {
        const input = event.target;
        const sendButton = document.getElementById('send-button');
        
        // Adjust textarea height
        this.adjustTextareaHeight(input);
        
        // Enable/disable send button
        sendButton.disabled = !input.value.trim() || this.isTyping;
    }

    // Handle keyboard shortcuts
    handleKeyDown(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            if (!this.isTyping) {
                document.getElementById('chat-form').dispatchEvent(new Event('submit'));
            }
        }
    }

    // Adjust textarea height based on content
    adjustTextareaHeight(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    }

    // Send message to API
    async sendMessage(message) {
        this.setTyping(true);
        
        try {
            const response = await api.sendMessage(message, this.currentConversationId);
            
            // Update current conversation ID
            this.currentConversationId = response.conversation_id;
            
            // Add assistant message
            this.addMessage(
                response.model_responses[0].response_text,
                'assistant',
                response.model_responses[0].id
            );
            
            // Update conversation list
            this.updateConversationList();
            
        } catch (error) {
            console.error('Send message error:', error);
            showNotification('Failed to send message: ' + error.message, 'error');
            
            // Add error message
            this.addMessage(
                'Sorry, I encountered an error processing your message. Please try again.',
                'assistant',
                null,
                true
            );
        } finally {
            this.setTyping(false);
        }
    }

    // Add message to chat display
    addMessage(content, type, responseId = null, isError = false) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageElement = document.createElement('div');
        messageElement.className = `message ${type}`;
        messageElement.dataset.responseId = responseId;
        
        const timestamp = new Date().toLocaleTimeString();
        const avatar = type === 'user' ? 'U' : 'AI';
        
        messageElement.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-bubble ${isError ? 'error' : ''}">
                    ${this.formatMessage(content)}
                </div>
                <div class="message-meta">
                    <span class="message-time">${timestamp}</span>
                    ${type === 'assistant' && responseId ? 
                        feedbackManager.createFeedbackButtons(responseId, true) : ''
                    }
                </div>
            </div>
        `;
        
        messagesContainer.appendChild(messageElement);
        
        // Apply existing feedback if available
        if (type === 'assistant' && responseId) {
            feedbackManager.applyExistingFeedback(messageElement, responseId);
        }
        
        // Scroll to bottom
        this.scrollToBottom();
        
        // Hide welcome message
        const welcomeMessage = messagesContainer.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.style.display = 'none';
        }
    }

    // Format message content
    formatMessage(content) {
        // Basic formatting - could be expanded
        return content
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>');
    }

    // Set typing state
    setTyping(isTyping) {
        this.isTyping = isTyping;
        
        const sendButton = document.getElementById('send-button');
        const messageInput = document.getElementById('message-input');
        
        sendButton.disabled = isTyping || !messageInput.value.trim();
        messageInput.disabled = isTyping;
        
        if (isTyping) {
            this.showTypingIndicator();
        } else {
            this.hideTypingIndicator();
        }
    }

    // Show typing indicator
    showTypingIndicator() {
        const messagesContainer = document.getElementById('chat-messages');
        
        // Remove existing indicator
        const existingIndicator = messagesContainer.querySelector('.typing-indicator');
        if (existingIndicator) {
            existingIndicator.remove();
        }
        
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = `
            <div class="message-avatar">AI</div>
            <div class="message-content">
                <div class="typing-dots">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
        `;
        
        messagesContainer.appendChild(indicator);
        this.scrollToBottom();
    }

    // Hide typing indicator
    hideTypingIndicator() {
        const indicator = document.querySelector('.typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    // Scroll to bottom of messages
    scrollToBottom() {
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Toggle sidebar
    toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        sidebar.classList.toggle('open');
    }

    // Close sidebar
    closeSidebar() {
        const sidebar = document.getElementById('sidebar');
        sidebar.classList.remove('open');
    }

    // Start new chat
    startNewChat() {
        this.currentConversationId = null;
        this.clearMessages();
        this.closeSidebar();
        
        // Focus on input
        document.getElementById('message-input').focus();
    }

    // Clear messages
    clearMessages() {
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.innerHTML = `
            <div class="welcome-message">
                <h2>Welcome to the Self-Learning LLM Platform</h2>
                <p>Start a conversation to begin. Your feedback helps improve the AI responses.</p>
            </div>
        `;
    }

    // Handle conversation click
    handleConversationClick(event) {
        const conversationItem = event.target.closest('.conversation-item');
        if (conversationItem) {
            const conversationId = conversationItem.dataset.conversationId;
            this.loadConversation(conversationId);
        }
    }

    // Load conversation
    async loadConversation(conversationId) {
        try {
            const conversation = await api.getConversation(conversationId);
            
            this.currentConversationId = conversationId;
            this.clearMessages();
            
            // Load messages
            for (const message of conversation.messages) {
                this.addMessage(
                    message.content,
                    message.message_type,
                    message.message_type === 'assistant' ? message.id : null
                );
            }
            
            // Update active conversation in sidebar
            this.updateActiveConversation(conversationId);
            
            // Close sidebar on mobile
            if (window.innerWidth <= 768) {
                this.closeSidebar();
            }
            
        } catch (error) {
            console.error('Load conversation error:', error);
            showNotification('Failed to load conversation', 'error');
        }
    }

    // Load conversations list
    async loadConversations() {
        try {
            const conversations = await api.getConversations();
            this.displayConversations(conversations);
        } catch (error) {
            console.error('Load conversations error:', error);
            // Don't show error notification for this as it might be due to no auth
        }
    }

    // Display conversations in sidebar
    displayConversations(conversations) {
        const conversationList = document.getElementById('conversation-list');
        
        if (conversations.length === 0) {
            conversationList.innerHTML = `
                <div class="empty-conversations">
                    <p>No conversations yet</p>
                    <p>Start a new chat to begin</p>
                </div>
            `;
            return;
        }
        
        conversationList.innerHTML = conversations.map(conv => {
            const lastMessage = conv.last_message_at ? 
                new Date(conv.last_message_at).toLocaleDateString() : 
                'No messages';
            
            return `
                <div class="conversation-item" data-conversation-id="${conv.id}">
                    <div class="conversation-title">${conv.title || 'Untitled'}</div>
                    <div class="conversation-meta">
                        ${conv.message_count} messages • ${lastMessage}
                    </div>
                </div>
            `;
        }).join('');
    }

    // Update conversations list
    updateConversationList() {
        // Reload conversations to get the latest
        this.loadConversations();
    }

    // Update active conversation
    updateActiveConversation(conversationId) {
        const conversationItems = document.querySelectorAll('.conversation-item');
        conversationItems.forEach(item => {
            item.classList.toggle('active', item.dataset.conversationId === conversationId);
        });
    }

    // Enable chat interface
    enableChat() {
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');
        
        messageInput.disabled = false;
        messageInput.placeholder = 'Type your message...';
        sendButton.disabled = !messageInput.value.trim();
        
        // Load conversations
        this.loadConversations();
        
        // Focus on input
        messageInput.focus();
    }

    // Disable chat interface
    disableChat() {
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');
        
        messageInput.disabled = true;
        messageInput.placeholder = 'Please login to start chatting...';
        sendButton.disabled = true;
        
        // Clear conversations
        document.getElementById('conversation-list').innerHTML = '';
        
        // Clear current conversation
        this.currentConversationId = null;
        this.clearMessages();
    }
}

// Create global chat manager
const chatManager = new ChatManager();

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChatManager;
}