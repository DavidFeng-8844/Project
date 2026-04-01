from __future__ import annotations

from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Tenant(db.Model):
    """
    SaaS Tenant (Organization/Company).
    Implements multi-tenant architecture with billing plans and quota limits.
    """
    __tablename__ = "tenants"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), unique=True, nullable=False)
    billing_plan = db.Column(db.String(32), nullable=False, default="Free") # e.g., Free, Pro, Enterprise
    api_quota_limit = db.Column(db.Integer, nullable=False, default=100) # Max inferences allowed for tenant
    api_quota_used = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    # A tenant owns multiple users, tasks, and reports. 
    # If a tenant is deleted, all their data is cascaded (deleted).
    users = db.relationship("User", back_populates="tenant", lazy=True, cascade="all, delete-orphan")
    inference_tasks = db.relationship("InferenceTask", back_populates="tenant", lazy=True, cascade="all, delete-orphan")
    inference_reports = db.relationship("InferenceReport", back_populates="tenant", lazy=True, cascade="all, delete-orphan")


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    # Role System Upgrade for SaaS: System_Admin, Tenant_Admin, Tenant_User
    role = db.Column(db.String(32), nullable=False, default="Tenant_User")

    tenant = db.relationship("Tenant", back_populates="users")
    inference_tasks = db.relationship("InferenceTask", back_populates="user", lazy=True)
    inference_reports = db.relationship("InferenceReport", back_populates="user", lazy=True)


class AIModel(db.Model):
    """
    System-wide AI Models available for Inference.
    These are global across the platform (not bound to a single tenant).
    """
    __tablename__ = "ai_models"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), unique=True, nullable=False)
    type = db.Column(db.String(64), nullable=False)
    status = db.Column(db.String(16), nullable=False, default="Active")
    weights_path = db.Column(db.String(255), nullable=False)

    inference_tasks = db.relationship("InferenceTask", back_populates="model", lazy=True)


class InferenceTask(db.Model):
    __tablename__ = "inference_tasks"

    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey("ai_models.id"), nullable=False)
    
    # Advanced Concurrency Support: Async Task Status
    status = db.Column(db.String(32), nullable=False, default="PENDING") # PENDING, RUNNING, SUCCESS, FAILED
    result_summary = db.Column(db.Text, nullable=True) # Nullable at first since task is PENDING
    
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    tenant = db.relationship("Tenant", back_populates="inference_tasks")
    user = db.relationship("User", back_populates="inference_tasks")
    model = db.relationship("AIModel", back_populates="inference_tasks")


class InferenceReport(db.Model):
    __tablename__ = "inference_reports"

    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    model_name = db.Column(db.String(128), nullable=False)
    summary = db.Column(db.Text, nullable=False)
    detail_json = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    tenant = db.relationship("Tenant", back_populates="inference_reports")
    user = db.relationship("User", back_populates="inference_reports")
