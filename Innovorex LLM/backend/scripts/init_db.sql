-- PostgreSQL Database Schema for Self-Learning LLM Platform
-- Version: 1.0
-- Description: Complete schema for users, conversations, messages, feedback, and vector mappings

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT valid_email CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    CONSTRAINT valid_username CHECK (username ~* '^[A-Za-z0-9_-]+$')
);

-- Conversations table
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_archived BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Indexes
    INDEX idx_conversations_user_id ON conversations(user_id),
    INDEX idx_conversations_created_at ON conversations(created_at DESC),
    INDEX idx_conversations_updated_at ON conversations(updated_at DESC)
);

-- Messages table
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    message_type VARCHAR(20) NOT NULL CHECK (message_type IN ('user', 'assistant')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    embedding_id VARCHAR(255), -- Links to Faiss vector index
    token_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Indexes
    INDEX idx_messages_conversation_id ON messages(conversation_id),
    INDEX idx_messages_created_at ON messages(created_at DESC),
    INDEX idx_messages_embedding_id ON messages(embedding_id),
    INDEX idx_messages_type ON messages(message_type)
);

-- Model responses table (stores all model outputs for comparison)
CREATE TABLE model_responses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL, -- 'phi-2', 'gemini', 'claude', etc.
    response_text TEXT NOT NULL,
    confidence_score FLOAT,
    generation_time_ms INTEGER,
    token_count INTEGER DEFAULT 0,
    embedding_id VARCHAR(255), -- Links to Faiss vector index
    model_version VARCHAR(50),
    parameters JSONB DEFAULT '{}'::jsonb, -- Model parameters used
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_primary BOOLEAN DEFAULT FALSE, -- Which response was shown to user
    
    -- Indexes
    INDEX idx_model_responses_message_id ON model_responses(message_id),
    INDEX idx_model_responses_model_name ON model_responses(model_name),
    INDEX idx_model_responses_created_at ON model_responses(created_at DESC),
    INDEX idx_model_responses_embedding_id ON model_responses(embedding_id),
    INDEX idx_model_responses_is_primary ON model_responses(is_primary)
);

-- Feedback table
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    response_id UUID NOT NULL REFERENCES model_responses(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    thumbs_up BOOLEAN,
    comment TEXT,
    feedback_type VARCHAR(50) DEFAULT 'rating', -- 'rating', 'correction', 'preference'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Indexes
    INDEX idx_feedback_response_id ON feedback(response_id),
    INDEX idx_feedback_user_id ON feedback(user_id),
    INDEX idx_feedback_rating ON feedback(rating),
    INDEX idx_feedback_created_at ON feedback(created_at DESC),
    
    -- Constraints
    CONSTRAINT feedback_rating_or_thumbs CHECK (
        (rating IS NOT NULL) OR (thumbs_up IS NOT NULL)
    )
);

-- Vector embeddings metadata table
CREATE TABLE vector_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    embedding_id VARCHAR(255) UNIQUE NOT NULL, -- References Faiss index
    content_type VARCHAR(50) NOT NULL, -- 'message', 'response', 'knowledge'
    content_id UUID NOT NULL, -- References messages.id or model_responses.id
    content_text TEXT NOT NULL,
    model_name VARCHAR(100) NOT NULL, -- Embedding model used
    dimensions INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Indexes
    INDEX idx_vector_embeddings_embedding_id ON vector_embeddings(embedding_id),
    INDEX idx_vector_embeddings_content_type ON vector_embeddings(content_type),
    INDEX idx_vector_embeddings_content_id ON vector_embeddings(content_id),
    INDEX idx_vector_embeddings_created_at ON vector_embeddings(created_at DESC)
);

-- Knowledge base table (for RAG context)
CREATE TABLE knowledge_base (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    source VARCHAR(255), -- URL, filename, etc.
    content_type VARCHAR(50) DEFAULT 'text', -- 'text', 'code', 'markdown'
    embedding_id VARCHAR(255), -- Links to Faiss vector index
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Indexes
    INDEX idx_knowledge_base_embedding_id ON knowledge_base(embedding_id),
    INDEX idx_knowledge_base_content_type ON knowledge_base(content_type),
    INDEX idx_knowledge_base_created_at ON knowledge_base(created_at DESC),
    INDEX idx_knowledge_base_is_active ON knowledge_base(is_active)
);

-- Model training sessions table
CREATE TABLE training_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    training_type VARCHAR(50) NOT NULL, -- 'fine_tune', 'lora', 'feedback_update'
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'running', -- 'running', 'completed', 'failed'
    data_size INTEGER, -- Number of training examples
    loss_metrics JSONB DEFAULT '{}'::jsonb,
    model_checkpoint_path VARCHAR(500),
    parameters JSONB DEFAULT '{}'::jsonb,
    error_log TEXT,
    created_by UUID REFERENCES users(id),
    
    -- Indexes
    INDEX idx_training_sessions_model_name ON training_sessions(model_name),
    INDEX idx_training_sessions_status ON training_sessions(status),
    INDEX idx_training_sessions_start_time ON training_sessions(start_time DESC)
);

-- Model comparison results table
CREATE TABLE model_comparisons (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prompt_text TEXT NOT NULL,
    phi2_response_id UUID REFERENCES model_responses(id),
    external_response_id UUID REFERENCES model_responses(id),
    winner VARCHAR(50), -- 'phi2', 'external', 'tie'
    comparison_criteria JSONB DEFAULT '{}'::jsonb,
    human_preference VARCHAR(50), -- From user feedback
    automated_score FLOAT, -- Automated evaluation score
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Indexes
    INDEX idx_model_comparisons_winner ON model_comparisons(winner),
    INDEX idx_model_comparisons_created_at ON model_comparisons(created_at DESC)
);

-- User sessions table
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    ip_address INET,
    user_agent TEXT,
    
    -- Indexes
    INDEX idx_user_sessions_user_id ON user_sessions(user_id),
    INDEX idx_user_sessions_token ON user_sessions(session_token),
    INDEX idx_user_sessions_expires_at ON user_sessions(expires_at)
);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_messages_updated_at BEFORE UPDATE ON messages
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_feedback_updated_at BEFORE UPDATE ON feedback
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_base_updated_at BEFORE UPDATE ON knowledge_base
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create indexes for performance
CREATE INDEX idx_messages_conversation_created ON messages(conversation_id, created_at DESC);
CREATE INDEX idx_model_responses_message_model ON model_responses(message_id, model_name);
CREATE INDEX idx_feedback_response_rating ON feedback(response_id, rating);

-- Create views for common queries
CREATE VIEW conversation_summary AS
SELECT 
    c.id,
    c.title,
    c.user_id,
    c.created_at,
    c.updated_at,
    COUNT(m.id) as message_count,
    MAX(m.created_at) as last_message_at
FROM conversations c
LEFT JOIN messages m ON c.id = m.conversation_id
GROUP BY c.id, c.title, c.user_id, c.created_at, c.updated_at;

CREATE VIEW model_performance_stats AS
SELECT 
    mr.model_name,
    COUNT(*) as total_responses,
    AVG(f.rating) as avg_rating,
    COUNT(CASE WHEN f.thumbs_up = true THEN 1 END) as thumbs_up_count,
    COUNT(CASE WHEN f.thumbs_up = false THEN 1 END) as thumbs_down_count,
    AVG(mr.generation_time_ms) as avg_generation_time_ms
FROM model_responses mr
LEFT JOIN feedback f ON mr.id = f.response_id
GROUP BY mr.model_name;

-- Sample data for testing
INSERT INTO users (username, email, password_hash) VALUES
('admin', 'admin@example.com', '$2b$12$example_hash_here'),
('testuser', 'test@example.com', '$2b$12$example_hash_here');

-- Grant permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO llm_platform_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO llm_platform_user;