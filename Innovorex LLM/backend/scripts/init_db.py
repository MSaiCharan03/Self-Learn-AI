#!/usr/bin/env python3
"""
Database initialization script for the Self-Learning LLM Platform
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.database import engine, Base, create_tables
from models.user import User
from models.conversation import Conversation, Message
from models.feedback import Feedback, UserSession
from models.training import TrainingSession, TrainingData
from models.model_response import ModelResponse, ModelComparison
from models.search import SearchQuery, SearchResult
from utils.security import get_password_hash
from datetime import datetime

def init_database():
    """Initialize the database with all tables"""
    print("Initializing database...")
    
    try:
        # Create all tables
        create_tables()
        print("✅ Database tables created successfully!")
        
        # List all created tables
        from sqlalchemy import inspect
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        print(f"\nCreated tables ({len(table_names)}):")
        for table_name in sorted(table_names):
            print(f"  - {table_name}")
            
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        sys.exit(1)

def create_default_admin():
    """Create default admin user"""
    print("\nCreating default admin user...")
    
    from sqlalchemy.orm import Session
    from core.database import get_db
    
    db = next(get_db())
    
    try:
        # Check if admin user already exists
        existing_admin = db.query(User).filter(User.username == "admin").first()
        
        if existing_admin:
            print("ℹ️  Admin user already exists, skipping creation")
            return
        
        # Create admin user
        from models.user import UserRole
        
        admin_user = User(
            username="admin",
            email="admin@innovorex.com",
            password_hash=get_password_hash("Innovorex@1"),
            role=UserRole.ADMIN,
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        
        print("✅ Default admin user created!")
        print(f"   Username: admin")
        print(f"   Password: Innovorex@1")
        print(f"   Email: admin@innovorex.com")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error creating admin user: {e}")
    finally:
        db.close()

def verify_database():
    """Verify database setup"""
    print("\nVerifying database setup...")
    
    try:
        from sqlalchemy import text
        
        with engine.connect() as conn:
            # Test database connection
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connection successful")
            
            # Check table counts
            from sqlalchemy import inspect
            inspector = inspect(engine)
            table_names = inspector.get_table_names()
            
            print(f"✅ Found {len(table_names)} tables")
            
            # Check if admin user exists
            from sqlalchemy.orm import Session
            from core.database import get_db
            
            db = next(get_db())
            
            try:
                admin_count = db.query(User).filter(User.username == "admin").count()
                if admin_count > 0:
                    print("✅ Admin user exists")
                else:
                    print("❌ Admin user not found")
            finally:
                db.close()
                
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False
    
    return True

def main():
    """Main initialization function"""
    print("Database Initialization")
    print("=" * 50)
    
    # Initialize database tables
    init_database()
    
    # Create default admin user
    create_default_admin()
    
    # Verify setup
    if verify_database():
        print("\n" + "=" * 50)
        print("✅ Database initialization completed successfully!")
        print("\nDefault admin credentials:")
        print("Username: admin")
        print("Password: Innovorex@1")
        print("\nYou can now start the application with:")
        print("python -m uvicorn main:app --host 0.0.0.0 --port 8000")
    else:
        print("\n❌ Database initialization failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()