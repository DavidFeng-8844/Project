import re

with open("deploy/flask-service/app/routes/inference.py", "r") as f:
    code = f.read()

# 1. Imports
code = code.replace("from PIL import Image", "from PIL import Image\nimport concurrent.futures\nimport io")

# 2. Add executor
code = code.replace("inference_bp = Blueprint(\"inference\", __name__, url_prefix=\"/inference\")\n", "inference_bp = Blueprint(\"inference\", __name__, url_prefix=\"/inference\")\n\nexecutor = concurrent.futures.ThreadPoolExecutor(max_workers=4)\n")

# 3. Add tenant_id to create_batch_inference_report
code = code.replace(
    "user_id = session.get(\"user_id\")\n    if not user_id:\n        return jsonify({\"error\": \"authentication required\"}), 401\n",
    "user_id = session.get(\"user_id\")\n    tenant_id = session.get(\"tenant_id\")\n    if not user_id or not tenant_id:\n        return jsonify({\"error\": \"authentication required\"}), 401\n"
)

# 4. Add tenant_id to InferenceTask in create_batch_inference_report
code = code.replace(
    "InferenceTask(user_id=user_id, model_id=model.id, result_summary=task_summary)",
    "InferenceTask(tenant_id=tenant_id, user_id=user_id, model_id=model.id, result_summary=task_summary)"
)

# 5. Add tenant_id to InferenceReport in create_batch_inference_report
code = code.replace(
    "InferenceReport(\n        user_id=user_id,\n        model_name=model_name,\n        summary=summary,\n        detail_json=json.dumps(detail_payload),\n    )",
    "InferenceReport(\n        tenant_id=tenant_id,\n        user_id=user_id,\n        model_name=model_name,\n        summary=summary,\n        detail_json=json.dumps(detail_payload),\n    )"
)

# 6. Add tenant_id to InferenceReport in create_inference_report_from_payload
code = code.replace(
    "InferenceReport(\n        user_id=user_id,\n        model_name=model.name,\n        summary=summary,\n        detail_json=json.dumps(detail_payload),\n    )",
    "InferenceReport(\n        tenant_id=tenant_id,\n        user_id=user_id,\n        model_name=model.name,\n        summary=summary,\n        detail_json=json.dumps(detail_payload),\n    )"
)

# 7. Rewrite create_inference_task and add get_task_status
new_task_code = """@inference_bp.post("/tasks")
def create_inference_task():
    user_id = session.get("user_id")
    tenant_id = session.get("tenant_id")
    if not user_id or not tenant_id:
        return jsonify({"error": "authentication required"}), 401

    model_id = request.form.get("model_id", type=int)
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400
    if "image" not in request.files:
        return jsonify({"error": "image file is required"}), 400

    model = AIModel.query.get(model_id)
    if not model:
        return jsonify({"error": "model not found"}), 404
    if model.status != "Active":
        return jsonify({"error": "selected model is inactive"}), 400
    
    image_file = request.files["image"]
    image_bytes = image_file.read()
    filename = image_file.filename or "image.jpg"
    
    temperature_threshold = (
        request.form.get("temperature_threshold", type=float)
        or request.args.get("temperature_threshold", type=float)
        or 50.0
    )
    confidence = (
        request.form.get("confidence", type=float)
        or request.args.get("confidence", type=float)
    )
    threshold = confidence if confidence is not None else _default_confidence_for_model(model)

    task = InferenceTask(
        tenant_id=tenant_id,
        user_id=user_id,
        model_id=model.id,
        status="PENDING"
    )
    db.session.add(task)
    db.session.commit()

    # Dispatch to background thread
    app = current_app._get_current_object()
    executor.submit(
        _background_inference_worker,
        app, task.id, model.id, image_bytes, filename, threshold, temperature_threshold
    )

    return jsonify({
        "task": {
            "id": task.id,
            "status": task.status
        }
    }), 202

def _background_inference_worker(app, task_id, model_id, image_bytes, filename, threshold, temperature_threshold):
    with app.app_context():
        try:
            task = InferenceTask.query.get(task_id)
            if not task:
                return
            model = AIModel.query.get(model_id)
            
            if _is_infrared_model(model):
                weights_path = Path("hybrid://infrared")
                suffix = Path(filename).suffix or ".jpg"
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(image_bytes)
                        tmp_path = tmp_file.name
                    result = run_infrared_hybrid_detection(
                        image_path=tmp_path, alarm_temp=float(temperature_threshold)
                    )
                finally:
                    if tmp_path:
                        try:
                            Path(tmp_path).unlink(missing_ok=True)
                        except Exception:
                            pass
            else:
                weights_path = _resolve_weights_path(model)
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                result = _predict_with_fallback(model, pil_image, threshold)

            result_payload = {
                "predictions": result["predictions"],
                "annotated_image": result["annotated_image"],
                "annotated_image_url": result["annotated_image_url"],
                "confidence_threshold": result["confidence_threshold"],
                "summary": result["summary"],
                "weights_path": str(weights_path),
                "fallback": result["fallback"],
            }
            task.result_summary = json.dumps(result_payload)
            task.status = "SUCCESS"
            db.session.commit()
            
        except Exception as exc:
            app.logger.exception("Background task failed")
            task = InferenceTask.query.get(task_id)
            if task:
                task.status = "FAILED"
                task.result_summary = str(exc)
                db.session.commit()

@inference_bp.get("/tasks/<int:task_id>/status")
def get_task_status(task_id):
    user_id = session.get("user_id")
    tenant_id = session.get("tenant_id")
    if not user_id or not tenant_id:
        return jsonify({"error": "authentication required"}), 401
        
    task = InferenceTask.query.filter_by(id=task_id, tenant_id=tenant_id).first()
    if not task:
        return jsonify({"error": "task not found"}), 404
        
    response_data = {
        "task": {
            "id": task.id,
            "status": task.status
        }
    }
    
    if task.status == "SUCCESS":
        try:
            result_payload = json.loads(task.result_summary)
            response_data.update(result_payload)
        except Exception:
            pass
    elif task.status == "FAILED":
        response_data["error"] = task.result_summary
        
    return jsonify(response_data)
"""

old_tasks_start = code.find('@inference_bp.post("/tasks")')
if old_tasks_start != -1:
    code = code[:old_tasks_start] + new_task_code

with open("deploy/flask-service/app/routes/inference.py", "w") as f:
    f.write(code)

print("FILE UPDATED SUCCESSFULLY")
