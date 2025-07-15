// Feedback Management
class FeedbackManager {
    constructor() {
        this.feedbackCache = new Map();
        this.init();
    }

    init() {
        // Listen for feedback events
        document.addEventListener('click', this.handleFeedbackClick.bind(this));
        document.addEventListener('feedback-submitted', this.handleFeedbackSubmitted.bind(this));
    }

    // Handle feedback button clicks
    async handleFeedbackClick(event) {
        const button = event.target.closest('.feedback-button');
        if (!button) return;

        event.preventDefault();
        
        const messageElement = button.closest('.message');
        const responseId = messageElement.dataset.responseId;
        
        if (!responseId) {
            console.error('No response ID found for feedback');
            return;
        }

        const action = button.dataset.action;
        
        try {
            await this.submitFeedback(responseId, action, button);
        } catch (error) {
            console.error('Feedback submission error:', error);
            showNotification('Failed to submit feedback', 'error');
        }
    }

    // Submit feedback to API
    async submitFeedback(responseId, action, buttonElement) {
        const messageElement = buttonElement.closest('.message');
        
        let rating = null;
        let thumbsUp = null;
        let comment = null;

        // Handle different feedback types
        switch (action) {
            case 'thumbs-up':
                thumbsUp = true;
                break;
            case 'thumbs-down':
                thumbsUp = false;
                break;
            case 'rating':
                rating = parseInt(buttonElement.dataset.rating);
                break;
            case 'comment':
                comment = await this.showCommentDialog();
                if (comment === null) return; // User cancelled
                break;
        }

        // Show loading state
        buttonElement.disabled = true;
        const originalText = buttonElement.textContent;
        buttonElement.textContent = '...';

        try {
            const response = await api.submitFeedback(responseId, rating, thumbsUp, comment);
            
            // Update UI
            this.updateFeedbackUI(messageElement, response);
            
            // Cache feedback
            this.feedbackCache.set(responseId, response);
            
            // Show success notification
            showNotification('Feedback submitted successfully', 'success');
            
            // Dispatch custom event
            document.dispatchEvent(new CustomEvent('feedback-submitted', {
                detail: { responseId, feedback: response }
            }));
            
        } catch (error) {
            console.error('Feedback submission failed:', error);
            showNotification('Failed to submit feedback', 'error');
        } finally {
            buttonElement.disabled = false;
            buttonElement.textContent = originalText;
        }
    }

    // Update feedback UI based on response
    updateFeedbackUI(messageElement, feedbackData) {
        const feedbackContainer = messageElement.querySelector('.message-feedback');
        if (!feedbackContainer) return;

        // Update thumbs up/down buttons
        const thumbsUpBtn = feedbackContainer.querySelector('.thumbs-up');
        const thumbsDownBtn = feedbackContainer.querySelector('.thumbs-down');
        
        if (thumbsUpBtn && thumbsDownBtn) {
            thumbsUpBtn.classList.toggle('active', feedbackData.thumbs_up === true);
            thumbsDownBtn.classList.toggle('active', feedbackData.thumbs_up === false);
        }

        // Update rating if present
        if (feedbackData.rating) {
            const ratingButtons = feedbackContainer.querySelectorAll('.rating-button');
            ratingButtons.forEach((btn, index) => {
                btn.classList.toggle('active', index + 1 <= feedbackData.rating);
            });
        }

        // Show comment indicator if present
        if (feedbackData.comment) {
            this.showCommentIndicator(feedbackContainer, feedbackData.comment);
        }
    }

    // Show comment dialog
    async showCommentDialog() {
        return new Promise((resolve) => {
            const dialog = document.createElement('div');
            dialog.className = 'comment-dialog-overlay';
            dialog.innerHTML = `
                <div class="comment-dialog">
                    <div class="dialog-header">
                        <h3>Add Comment</h3>
                        <button class="close-dialog">×</button>
                    </div>
                    <div class="dialog-content">
                        <textarea 
                            id="comment-input" 
                            placeholder="Share your feedback..."
                            rows="4"
                        ></textarea>
                    </div>
                    <div class="dialog-actions">
                        <button class="cancel-btn">Cancel</button>
                        <button class="submit-btn">Submit</button>
                    </div>
                </div>
            `;

            document.body.appendChild(dialog);

            const commentInput = dialog.querySelector('#comment-input');
            const submitBtn = dialog.querySelector('.submit-btn');
            const cancelBtn = dialog.querySelector('.cancel-btn');
            const closeBtn = dialog.querySelector('.close-dialog');

            // Focus on input
            commentInput.focus();

            // Handle submit
            const handleSubmit = () => {
                const comment = commentInput.value.trim();
                document.body.removeChild(dialog);
                resolve(comment || null);
            };

            // Handle cancel
            const handleCancel = () => {
                document.body.removeChild(dialog);
                resolve(null);
            };

            // Event listeners
            submitBtn.addEventListener('click', handleSubmit);
            cancelBtn.addEventListener('click', handleCancel);
            closeBtn.addEventListener('click', handleCancel);
            
            // Submit on Ctrl+Enter
            commentInput.addEventListener('keydown', (e) => {
                if (e.ctrlKey && e.key === 'Enter') {
                    handleSubmit();
                }
                if (e.key === 'Escape') {
                    handleCancel();
                }
            });

            // Close on overlay click
            dialog.addEventListener('click', (e) => {
                if (e.target === dialog) {
                    handleCancel();
                }
            });
        });
    }

    // Show comment indicator
    showCommentIndicator(container, comment) {
        // Remove existing indicator
        const existingIndicator = container.querySelector('.comment-indicator');
        if (existingIndicator) {
            existingIndicator.remove();
        }

        // Create new indicator
        const indicator = document.createElement('button');
        indicator.className = 'feedback-button comment-indicator';
        indicator.innerHTML = '💬';
        indicator.title = 'Comment: ' + comment;
        indicator.addEventListener('click', () => {
            this.showCommentTooltip(indicator, comment);
        });

        container.appendChild(indicator);
    }

    // Show comment tooltip
    showCommentTooltip(element, comment) {
        // Remove existing tooltip
        const existingTooltip = document.querySelector('.comment-tooltip');
        if (existingTooltip) {
            existingTooltip.remove();
        }

        const tooltip = document.createElement('div');
        tooltip.className = 'comment-tooltip';
        tooltip.textContent = comment;

        document.body.appendChild(tooltip);

        // Position tooltip
        const rect = element.getBoundingClientRect();
        tooltip.style.left = rect.left + 'px';
        tooltip.style.top = (rect.top - tooltip.offsetHeight - 5) + 'px';

        // Auto-hide after 3 seconds
        setTimeout(() => {
            if (tooltip.parentNode) {
                tooltip.remove();
            }
        }, 3000);

        // Hide on click anywhere
        const hideTooltip = (e) => {
            if (!tooltip.contains(e.target)) {
                tooltip.remove();
                document.removeEventListener('click', hideTooltip);
            }
        };
        document.addEventListener('click', hideTooltip);
    }

    // Create feedback buttons for a message
    createFeedbackButtons(responseId, isAssistant = true) {
        if (!isAssistant) return '';

        return `
            <div class="message-feedback">
                <button class="feedback-button thumbs-up" data-action="thumbs-up" title="Good response">
                    👍
                </button>
                <button class="feedback-button thumbs-down" data-action="thumbs-down" title="Poor response">
                    👎
                </button>
                <button class="feedback-button comment-btn" data-action="comment" title="Add comment">
                    💬
                </button>
            </div>
        `;
    }

    // Load existing feedback for a response
    async loadFeedback(responseId) {
        // Check cache first
        if (this.feedbackCache.has(responseId)) {
            return this.feedbackCache.get(responseId);
        }

        try {
            const feedback = await api.getResponseFeedback(responseId);
            
            // Cache the result
            if (feedback.length > 0) {
                this.feedbackCache.set(responseId, feedback[0]);
                return feedback[0];
            }
            
            return null;
        } catch (error) {
            console.error('Failed to load feedback:', error);
            return null;
        }
    }

    // Apply existing feedback to UI
    async applyExistingFeedback(messageElement, responseId) {
        const feedback = await this.loadFeedback(responseId);
        if (feedback) {
            this.updateFeedbackUI(messageElement, feedback);
        }
    }

    // Handle feedback submitted event
    handleFeedbackSubmitted(event) {
        const { responseId, feedback } = event.detail;
        
        // Update any other UI elements that might need to reflect this feedback
        console.log('Feedback submitted:', responseId, feedback);
        
        // Could trigger analytics, update stats, etc.
    }

    // Get feedback statistics
    async getFeedbackStats() {
        try {
            return await api.getFeedbackStats();
        } catch (error) {
            console.error('Failed to get feedback stats:', error);
            return null;
        }
    }

    // Clear feedback cache
    clearCache() {
        this.feedbackCache.clear();
    }
}

// Create global feedback manager
const feedbackManager = new FeedbackManager();

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FeedbackManager;
}