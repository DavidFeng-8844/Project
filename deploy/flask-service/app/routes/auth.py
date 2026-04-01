from __future__ import annotations

from flask import Blueprint, jsonify, request, session
from werkzeug.security import check_password_hash, generate_password_hash

from app.models import User, Tenant, db

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


@auth_bp.post("/register")
def register():
    payload = request.get_json(silent=True) or {}
    username = payload.get("username", "").strip()
    password = payload.get("password", "")
    tenant_name = payload.get("tenant_name", "").strip()

    if not username or not password or not tenant_name:
        return jsonify({"error": "username, password, and tenant_name are required"}), 400
    if len(username) < 3:
        return jsonify({"error": "username must be at least 3 characters"}), 400
    if len(password) < 6:
        return jsonify({"error": "password must be at least 6 characters"}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "username already exists"}), 409

    # Multitenant Logic: Find or Create Tenant
    tenant = Tenant.query.filter_by(name=tenant_name).first()
    role = "Tenant_User"

    if not tenant:
        # If tenant doesn't exist, create it auto-magically and make this first user the Admin
        tenant = Tenant(name=tenant_name)
        db.session.add(tenant)
        db.session.commit() # Commit right away to generate tenant.id
        role = "Tenant_Admin"

    # Create the user and bind to the tenant
    user = User(
        username=username,
        password_hash=generate_password_hash(password),
        tenant_id=tenant.id,
        role=role,
    )
    db.session.add(user)
    db.session.commit()

    return jsonify(
        {
            "message": "registration successful",
            "user": {
                "id": user.id, 
                "username": user.username, 
                "role": user.role,
                "tenant_id": tenant.id,
                "tenant_name": tenant.name
            },
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

    # Save to session (Flask secure cookie)
    session["user_id"] = user.id
    session["username"] = user.username
    session["role"] = user.role
    session["tenant_id"] = user.tenant_id

    return jsonify(
        {
            "message": "login successful",
            "user": {
                "id": user.id, 
                "username": user.username, 
                "role": user.role,
                "tenant_id": user.tenant.id,
                "tenant_name": user.tenant.name
            },
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

    # Fetch fresh to grab the associated tenant name
    user = User.query.get(user_id)
    if not user:
        session.clear()
        return jsonify({"authenticated": False, "user": None}), 401

    return jsonify(
        {
            "authenticated": True,
            "user": {
                "id": user.id,
                "username": user.username,
                "role": user.role,
                "tenant_id": user.tenant.id,
                "tenant_name": user.tenant.name
            },
        }
    )
