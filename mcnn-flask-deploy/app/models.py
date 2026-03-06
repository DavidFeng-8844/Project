from __future__ import annotations

from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(16), nullable=False, default="User")

    inference_tasks = db.relationship("InferenceTask", back_populates="user", lazy=True)


class AIModel(db.Model):
    __tablename__ = "ai_models"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), unique=True, nullable=False)
    type = db.Column(db.String(64), nullable=False)
    status = db.Column(db.String(16), nullable=False, default="Active")
    weights_path = db.Column(db.String(255), nullable=False)

    inference_tasks = db.relationship(
        "InferenceTask", back_populates="model", lazy=True
    )


class InferenceTask(db.Model):
    __tablename__ = "inference_tasks"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey("ai_models.id"), nullable=False)
    result_summary = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    user = db.relationship("User", back_populates="inference_tasks")
    model = db.relationship("AIModel", back_populates="inference_tasks")
