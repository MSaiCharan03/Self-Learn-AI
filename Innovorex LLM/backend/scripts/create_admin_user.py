#!/usr/bin/env python3
"""
Script to create the admin user for the Self-Learning LLM Platform
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy.orm import Session
from core.database import get_db, engine
from models.user import User
from utils.security import get_password_hash
from datetime import datetime

def create_admin_user():
    """Create the admin user with predefined credentials"""
    print("Creating admin user...")
    
    # Create database session
    db = next(get_db())
    
    try:
        # Check if admin user already exists
        existing_admin = db.query(User).filter(User.username == "admin").first()
        
        if existing_admin:
            print("❌ Admin user already exists!")
            print(f"   Username: {existing_admin.username}")
            print(f"   Email: {existing_admin.email}")
            print(f"   Created: {existing_admin.created_at}")
            print(f"   Is Active: {existing_admin.is_active}")
            
            # Ask if user wants to update password
            response = input("\nDo you want to update the admin password? (y/N): ").lower()
            if response == 'y':
                existing_admin.password_hash = get_password_hash("Innovorex@1")
                existing_admin.updated_at = datetime.utcnow()
                db.commit()
                print("✅ Admin password updated successfully!")
            else:
                print("Admin user creation cancelled.")
            return
        
        # Create new admin user
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
        
        print("✅ Admin user created successfully!")
        print(f"   Username: {admin_user.username}")
        print(f"   Email: {admin_user.email}")
        print(f"   Password: Innovorex@1")
        print(f"   User ID: {admin_user.id}")
        print(f"   Created: {admin_user.created_at}")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error creating admin user: {e}")
        sys.exit(1)
    finally:
        db.close()

def verify_admin_user():
    """Verify admin user credentials"""
    print("\nVerifying admin user credentials...")
    
    db = next(get_db())
    
    try:
        from utils.security import verify_password
        
        admin_user = db.query(User).filter(User.username == "admin").first()
        
        if not admin_user:
            print("❌ Admin user not found!")
            return False
            
        # Verify password
        if verify_password("Innovorex@1", admin_user.password_hash):
            print("✅ Admin credentials verified successfully!")
            print(f"   Username: {admin_user.username}")
            print(f"   Email: {admin_user.email}")
            print(f"   Active: {admin_user.is_active}")
            return True
        else:
            print("❌ Admin password verification failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error verifying admin user: {e}")
        return False
    finally:
        db.close()

def main():
    """Main function"""
    print("Admin User Setup")
    print("=" * 50)
    
    # Create admin user
    create_admin_user()
    
    # Verify admin user
    verify_admin_user()
    
    print("\n" + "=" * 50)
    print("Admin user setup completed!")
    print("\nYou can now login with:")
    print("Username: admin")
    print("Password: Innovorex@1")

if __name__ == "__main__":
    main()