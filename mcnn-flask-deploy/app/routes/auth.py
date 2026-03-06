from __future__ import annotations

from flask import Blueprint, jsonify, request, session
from werkzeug.security import check_password_hash, generate_password_hash

from app.models import User, db

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


@auth_bp.post("/register")
def register():
    payload = request.get_json(silent=True) or {}
    username = payload.get("username", "").strip()
    password = payload.get("password", "")

    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400
    if len(username) < 3:
        return jsonify({"error": "username must be at least 3 characters"}), 400
    if len(password) < 6:
        return jsonify({"error": "password must be at least 6 characters"}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "username already exists"}), 409

    user = User(
        username=username,
        password_hash=generate_password_hash(password),
        role="User",
    )
    db.session.add(user)
    db.session.commit()

    return jsonify(
        {
            "message": "registration successful",
            "user": {"id": user.id, "username": user.username, "role": user.role},
        }
    ), 201


@auth_bp.post("/login")
def login():
    payload = request.get_json(silent=True) or {}
    username = payload.get("username", "").strip()
    password = payload.get("password", "")

    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400

    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "invalid username or password"}), 401

    session["user_id"] = user.id
    session["username"] = user.username
    session["role"] = user.role

    return jsonify(
        {
            "message": "login successful",
            "user": {"id": user.id, "username": user.username, "role": user.role},
        }
    )


@auth_bp.post("/logout")
def logout():
    session.clear()
    return jsonify({"message": "logout successful"})


@auth_bp.get("/me")
def me():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"authenticated": False, "user": None}), 401

    return jsonify(
        {
            "authenticated": True,
            "user": {
                "id": user_id,
                "username": session.get("username"),
                "role": session.get("role"),
            },
        }
    )
