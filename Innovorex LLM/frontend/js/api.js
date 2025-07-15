// API Communication Layer
class API {
    constructor() {
        this.baseURL = 'http://localhost:8000/api/v1';
        this.token = localStorage.getItem('authToken');
        this.refreshToken = localStorage.getItem('refreshToken');
    }

    // Helper method to make authenticated requests
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        // Add authorization header if token exists
        if (this.token) {
            config.headers.Authorization = `Bearer ${this.token}`;
        }

        try {
            const response = await fetch(url, config);
            
            // Handle 401 - Unauthorized
            if (response.status === 401) {
                this.handleAuthError();
                throw new Error('Authentication required');
            }

            // Handle other HTTP errors
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }

            // Handle empty responses
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            }
            
            return await response.text();
        } catch (error) {
            console.error('API Request Error:', error);
            throw error;
        }
    }

    // Handle authentication errors
    handleAuthError() {
        this.token = null;
        this.refreshToken = null;
        localStorage.removeItem('authToken');
        localStorage.removeItem('refreshToken');
        localStorage.removeItem('user');
        
        // Redirect to login if not already there
        if (window.location.pathname !== '/login') {
            window.dispatchEvent(new CustomEvent('auth-required'));
        }
    }

    // Set authentication token
    setToken(token) {
        this.token = token;
        localStorage.setItem('authToken', token);
    }

    // Clear authentication
    clearAuth() {
        this.token = null;
        this.refreshToken = null;
        localStorage.removeItem('authToken');
        localStorage.removeItem('refreshToken');
        localStorage.removeItem('user');
    }

    // Authentication Methods
    async login(username, password) {
        const formData = new FormData();
        formData.append('username', username);
        formData.append('password', password);

        const response = await this.request('/auth/login', {
            method: 'POST',
            headers: {}, // Remove Content-Type for FormData
            body: formData
        });

        if (response.access_token) {
            this.setToken(response.access_token);
            localStorage.setItem('user', JSON.stringify(response.user));
        }

        return response;
    }

    async register(username, email, password) {
        return await this.request('/auth/register', {
            method: 'POST',
            body: JSON.stringify({ username, email, password })
        });
    }

    async logout() {
        try {
            await this.request('/auth/logout', { method: 'POST' });
        } catch (error) {
            console.error('Logout error:', error);
        } finally {
            this.clearAuth();
        }
    }

    async getCurrentUser() {
        return await this.request('/auth/me');
    }

    // Chat Methods
    async sendMessage(content, conversationId = null) {
        return await this.request('/chat/send', {
            method: 'POST',
            body: JSON.stringify({
                content,
                conversation_id: conversationId
            })
        });
    }

    async getConversations(limit = 50, offset = 0) {
        return await this.request(`/chat/conversations?limit=${limit}&offset=${offset}`);
    }

    async getConversation(conversationId) {
        return await this.request(`/chat/conversations/${conversationId}`);
    }

    async deleteConversation(conversationId) {
        return await this.request(`/chat/conversations/${conversationId}`, {
            method: 'DELETE'
        });
    }

    async archiveConversation(conversationId) {
        return await this.request(`/chat/conversations/${conversationId}/archive`, {
            method: 'PUT'
        });
    }

    // Feedback Methods
    async submitFeedback(responseId, rating = null, thumbsUp = null, comment = null) {
        return await this.request('/feedback/submit', {
            method: 'POST',
            body: JSON.stringify({
                response_id: responseId,
                rating,
                thumbs_up: thumbsUp,
                comment,
                feedback_type: 'rating'
            })
        });
    }

    async getResponseFeedback(responseId) {
        return await this.request(`/feedback/response/${responseId}`);
    }

    async getMyFeedback(limit = 50, offset = 0) {
        return await this.request(`/feedback/my-feedback?limit=${limit}&offset=${offset}`);
    }

    async getFeedbackStats() {
        return await this.request('/feedback/stats');
    }

    // Search Methods
    async semanticSearch(query, limit = 10, contentTypes = null, minSimilarity = 0.7) {
        return await this.request('/search/semantic', {
            method: 'POST',
            body: JSON.stringify({
                query,
                limit,
                content_types: contentTypes,
                min_similarity: minSimilarity
            })
        });
    }

    async getKnowledgeBase(limit = 50, offset = 0, contentType = null) {
        let url = `/search/knowledge?limit=${limit}&offset=${offset}`;
        if (contentType) {
            url += `&content_type=${contentType}`;
        }
        return await this.request(url);
    }

    async createKnowledgeEntry(title, content, source = null, contentType = 'text') {
        return await this.request('/search/knowledge', {
            method: 'POST',
            body: JSON.stringify({
                title,
                content,
                source,
                content_type: contentType,
                metadata: {}
            })
        });
    }

    async getSimilarMessages(messageId, topK = 5, minSimilarity = 0.8) {
        return await this.request(`/search/similar-messages/${messageId}?top_k=${topK}&min_similarity=${minSimilarity}`);
    }

    async getRelevantKnowledge(query, topK = 3, minSimilarity = 0.7) {
        return await this.request(`/search/relevant-knowledge?query=${encodeURIComponent(query)}&top_k=${topK}&min_similarity=${minSimilarity}`);
    }

    async getSearchStats() {
        return await this.request('/search/stats');
    }

    // Model Methods
    async getAvailableModels() {
        return await this.request('/models/available');
    }

    async getModelStatus(modelName) {
        return await this.request(`/models/status/${modelName}`);
    }

    async generateResponse(prompt, modelName = 'phi-2', parameters = {}) {
        return await this.request('/models/generate', {
            method: 'POST',
            body: JSON.stringify({
                prompt,
                model_name: modelName,
                ...parameters
            })
        });
    }

    async startTraining(modelName, trainingType, parameters = {}) {
        return await this.request('/models/train', {
            method: 'POST',
            body: JSON.stringify({
                model_name: modelName,
                training_type: trainingType,
                parameters
            })
        });
    }

    async getTrainingSessions(limit = 50, offset = 0) {
        return await this.request(`/models/training?limit=${limit}&offset=${offset}`);
    }

    async getTrainingSession(sessionId) {
        return await this.request(`/models/training/${sessionId}`);
    }

    // Health Check
    async healthCheck() {
        return await this.request('/health', { method: 'GET' });
    }
}

// Create global API instance
const api = new API();

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = API;
}