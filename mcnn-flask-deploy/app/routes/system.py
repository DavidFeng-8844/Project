from __future__ import annotations

from flask import Blueprint, jsonify, render_template, session

system_bp = Blueprint("system", __name__)


@system_bp.get("/")
def index():
    return render_template(
        "index.html",
        current_user=session.get("username", "Guest"),
        current_role=session.get("role", "Anonymous"),
    )


@system_bp.get("/history")
def history_page():
    return render_template(
        "history.html",
        current_user=session.get("username", "Guest"),
        current_role=session.get("role", "Anonymous"),
    )


@system_bp.get("/health")
def health():
    return jsonify({"status": "ok"})
