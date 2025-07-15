// Main Application Logic
class App {
    constructor() {
        this.currentUser = null;
        this.isAuthenticated = false;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkAuthStatus();
        this.setupNotifications();
    }

    setupEventListeners() {
        // Auth form submissions
        document.getElementById('login-form').addEventListener('submit', this.handleLogin.bind(this));
        document.getElementById('register-form').addEventListener('submit', this.handleRegister.bind(this));
        
        // Tab switching
        document.addEventListener('click', this.handleTabClick.bind(this));
        
        // User menu
        document.getElementById('user-menu-btn').addEventListener('click', this.toggleUserMenu.bind(this));
        document.getElementById('logout-btn').addEventListener('click', this.handleLogout.bind(this));
        
        // Close dropdowns on outside click
        document.addEventListener('click', this.handleOutsideClick.bind(this));
        
        // Auth events
        window.addEventListener('auth-required', this.showLoginScreen.bind(this));
        
        // Responsive sidebar
        window.addEventListener('resize', this.handleResize.bind(this));
    }

    // Check authentication status on load
    async checkAuthStatus() {
        const token = localStorage.getItem('authToken');
        const userData = localStorage.getItem('user');
        
        if (token && userData) {
            try {
                // Verify token is still valid
                const user = await api.getCurrentUser();
                this.setUser(user);
                this.showChatScreen();
            } catch (error) {
                console.error('Auth check failed:', error);
                this.clearAuth();
                this.showLoginScreen();
            }
        } else {
            this.showLoginScreen();
        }
    }

    // Handle login form submission
    async handleLogin(event) {
        event.preventDefault();
        
        const form = event.target;
        const formData = new FormData(form);
        const username = formData.get('username') || document.getElementById('login-username').value;
        const password = formData.get('password') || document.getElementById('login-password').value;
        
        const submitButton = form.querySelector('button[type="submit"]');
        const originalText = submitButton.textContent;
        
        try {
            // Show loading state
            submitButton.disabled = true;
            submitButton.textContent = 'Logging in...';
            
            // Clear any previous errors
            this.clearFormErrors(form);
            
            // Attempt login
            const response = await api.login(username, password);
            
            // Set user and show chat
            this.setUser(response.user);
            this.showChatScreen();
            
            // Show success notification
            showNotification('Login successful!', 'success');
            
        } catch (error) {
            console.error('Login error:', error);
            this.showFormError(form, error.message);
            showNotification('Login failed: ' + error.message, 'error');
        } finally {
            submitButton.disabled = false;
            submitButton.textContent = originalText;
        }
    }

    // Handle register form submission
    async handleRegister(event) {
        event.preventDefault();
        
        const form = event.target;
        const formData = new FormData(form);
        const username = formData.get('username') || document.getElementById('register-username').value;
        const email = formData.get('email') || document.getElementById('register-email').value;
        const password = formData.get('password') || document.getElementById('register-password').value;
        
        const submitButton = form.querySelector('button[type="submit"]');
        const originalText = submitButton.textContent;
        
        try {
            // Show loading state
            submitButton.disabled = true;
            submitButton.textContent = 'Creating account...';
            
            // Clear any previous errors
            this.clearFormErrors(form);
            
            // Attempt registration
            await api.register(username, email, password);
            
            // Show success and switch to login
            showNotification('Account created successfully! Please login.', 'success');
            this.switchToLoginTab();
            
        } catch (error) {
            console.error('Registration error:', error);
            this.showFormError(form, error.message);
            showNotification('Registration failed: ' + error.message, 'error');
        } finally {
            submitButton.disabled = false;
            submitButton.textContent = originalText;
        }
    }

    // Handle logout
    async handleLogout() {
        try {
            await api.logout();
        } catch (error) {
            console.error('Logout error:', error);
        }
        
        this.clearAuth();
        this.showLoginScreen();
        showNotification('Logged out successfully', 'info');
    }

    // Handle tab clicks
    handleTabClick(event) {
        const tabButton = event.target.closest('.tab-button');
        if (tabButton) {
            const tab = tabButton.dataset.tab;
            this.switchTab(tab);
        }
    }

    // Switch between login and register tabs
    switchTab(tab) {
        // Update tab buttons
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tab);
        });
        
        // Update forms
        document.querySelectorAll('.auth-form').forEach(form => {
            form.classList.toggle('active', form.id === `${tab}-form`);
        });
        
        // Clear any errors
        document.querySelectorAll('.auth-form').forEach(form => {
            this.clearFormErrors(form);
        });
        
        // Focus on first input
        setTimeout(() => {
            const activeForm = document.querySelector('.auth-form.active');
            const firstInput = activeForm.querySelector('input');
            if (firstInput) firstInput.focus();
        }, 100);
    }

    // Switch to login tab
    switchToLoginTab() {
        this.switchTab('login');
    }

    // Toggle user menu
    toggleUserMenu() {
        const dropdown = document.getElementById('user-dropdown');
        dropdown.classList.toggle('hidden');
    }

    // Handle outside clicks
    handleOutsideClick(event) {
        const userMenu = document.getElementById('user-dropdown');
        const userMenuBtn = document.getElementById('user-menu-btn');
        
        if (!userMenuBtn.contains(event.target) && !userMenu.contains(event.target)) {
            userMenu.classList.add('hidden');
        }
    }

    // Handle window resize
    handleResize() {
        const sidebar = document.getElementById('sidebar');
        if (window.innerWidth > 768) {
            // Close sidebar on larger screens
            sidebar.classList.remove('open');
        }
    }

    // Set current user
    setUser(user) {
        this.currentUser = user;
        this.isAuthenticated = true;
        
        // Update user display
        document.getElementById('username-display').textContent = user.username;
        
        // Store in localStorage
        localStorage.setItem('user', JSON.stringify(user));
        
        // Dispatch auth event
        document.dispatchEvent(new CustomEvent('user-authenticated', { detail: user }));
    }

    // Clear authentication
    clearAuth() {
        this.currentUser = null;
        this.isAuthenticated = false;
        
        // Clear API auth
        api.clearAuth();
        
        // Disable chat
        if (chatManager) {
            chatManager.disableChat();
        }
        
        // Clear feedback cache
        if (feedbackManager) {
            feedbackManager.clearCache();
        }
    }

    // Show login screen
    showLoginScreen() {
        document.getElementById('login-screen').classList.remove('hidden');
        document.getElementById('chat-screen').classList.add('hidden');
        
        // Focus on first input
        setTimeout(() => {
            const firstInput = document.querySelector('.auth-form.active input');
            if (firstInput) firstInput.focus();
        }, 100);
    }

    // Show chat screen
    showChatScreen() {
        document.getElementById('login-screen').classList.add('hidden');
        document.getElementById('chat-screen').classList.remove('hidden');
        
        // Enable chat
        if (chatManager) {
            chatManager.enableChat();
        }
    }

    // Show form error
    showFormError(form, message) {
        // Remove existing errors
        this.clearFormErrors(form);
        
        // Create error element
        const errorElement = document.createElement('div');
        errorElement.className = 'form-error';
        errorElement.textContent = message;
        
        // Add to form
        form.insertBefore(errorElement, form.firstChild);
    }

    // Clear form errors
    clearFormErrors(form) {
        const errors = form.querySelectorAll('.form-error');
        errors.forEach(error => error.remove());
    }

    // Setup notifications
    setupNotifications() {
        // Create notification styles if not exists
        if (!document.getElementById('notification-styles')) {
            const style = document.createElement('style');
            style.id = 'notification-styles';
            style.textContent = `
                .form-error {
                    background: #f56565;
                    color: white;
                    padding: 0.75rem;
                    border-radius: 5px;
                    margin-bottom: 1rem;
                    font-size: 0.9rem;
                }
                
                .comment-dialog-overlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0,0,0,0.5);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 1000;
                }
                
                .comment-dialog {
                    background: white;
                    border-radius: 10px;
                    width: 90%;
                    max-width: 500px;
                    max-height: 80vh;
                    overflow: hidden;
                }
                
                .dialog-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 1rem;
                    border-bottom: 1px solid #e0e0e0;
                }
                
                .dialog-header h3 {
                    margin: 0;
                    color: #333;
                }
                
                .close-dialog {
                    background: none;
                    border: none;
                    font-size: 1.5rem;
                    cursor: pointer;
                    color: #666;
                }
                
                .dialog-content {
                    padding: 1rem;
                }
                
                .dialog-content textarea {
                    width: 100%;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 0.75rem;
                    font-family: inherit;
                    font-size: 1rem;
                    resize: vertical;
                }
                
                .dialog-actions {
                    display: flex;
                    justify-content: flex-end;
                    gap: 0.5rem;
                    padding: 1rem;
                    border-top: 1px solid #e0e0e0;
                }
                
                .dialog-actions button {
                    padding: 0.5rem 1rem;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 0.9rem;
                }
                
                .cancel-btn {
                    background: #f0f0f0;
                    color: #333;
                }
                
                .submit-btn {
                    background: #667eea;
                    color: white;
                }
                
                .cancel-btn:hover {
                    background: #e0e0e0;
                }
                
                .submit-btn:hover {
                    background: #5a67d8;
                }
                
                .comment-tooltip {
                    position: absolute;
                    background: #333;
                    color: white;
                    padding: 0.5rem;
                    border-radius: 5px;
                    max-width: 200px;
                    z-index: 1000;
                    font-size: 0.85rem;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                }
                
                .empty-conversations {
                    text-align: center;
                    padding: 2rem;
                    color: #666;
                }
                
                .empty-conversations p {
                    margin: 0.5rem 0;
                }
            `;
            document.head.appendChild(style);
        }
    }
}

// Global notification function
function showNotification(message, type = 'info') {
    const container = document.getElementById('notifications');
    
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    container.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
    
    // Remove on click
    notification.addEventListener('click', () => {
        notification.remove();
    });
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = App;
}