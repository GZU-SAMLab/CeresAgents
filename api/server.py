import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TOOLS_SERVICE_DIR = os.path.join(PROJECT_ROOT, "tools_service")
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
SEGMENTATION_OUTPUT_DIR = os.path.join(UPLOAD_FOLDER, "segmentation")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
MAX_FILE_SIZE = 16 * 1024 * 1024

from ceres_agents import CeresAgentsGraph
from tools_service.disease_class_service import DiseaseClassService
from tools_service.segmentation_service import SegmentationSeverityService

app = Flask(__name__)
CORS(app)
app.config["JSON_AS_ASCII"] = False
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

try:
    app.json.ensure_ascii = False
except Exception:
    pass

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEGMENTATION_OUTPUT_DIR, exist_ok=True)

_agent: Optional[CeresAgentsGraph] = None
_segmentation_service: Optional[SegmentationSeverityService] = None
_disease_class_service: Optional[DiseaseClassService] = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def error_response(status_code: int, message: str):
    return jsonify({"code": status_code, "error": message}), status_code


def debug_enabled() -> bool:
    return os.getenv("CERES_SERVER_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _save_uploaded_file(file_storage) -> str:
    original_filename = secure_filename(file_storage.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{original_filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file_storage.save(filepath)
    return filepath


def _resolve_request_image_path() -> tuple[Optional[str], Optional[Response]]:
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return None, error_response(400, "empty filename")
        if not allowed_file(file.filename):
            return None, error_response(400, f"unsupported file type; allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
        return _save_uploaded_file(file), None

    data = request.get_json(silent=True) or {}
    image_path = data.get("image_path")
    if not image_path:
        return None, error_response(400, "missing image_path or file")
    return image_path, None


def _model_base_url() -> str:
    return (
        os.getenv("MODEL_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("DASHSCOPE_BASE_URL")
    )


def _resolve_model_path(env_name: str, *default_parts: str) -> str:
    configured = (os.getenv(env_name) or "").strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return os.path.join(TOOLS_SERVICE_DIR, *default_parts)


def get_agent() -> CeresAgentsGraph:
    global _agent
    if _agent is None:
        model = os.getenv("MODEL_NAME") or os.getenv("LLM_MODEL", "qwen-plus")
        _agent = CeresAgentsGraph(model=model, base_url=_model_base_url())
    return _agent


def get_segmentation_service() -> SegmentationSeverityService:
    global _segmentation_service
    if _segmentation_service is None:
        leaf_model_path = _resolve_model_path(
            "CERES_SEG_LEAF_MODEL_PATH",
            "segformer_para", "checkpoints", "SegFormer", "seg_leaf", "best_model.pth",
        )
        lesion_model_path = _resolve_model_path(
            "CERES_SEG_LESION_MODEL_PATH",
            "segformer_para", "checkpoints", "SegFormer", "seg_lesion", "best_model.pth",
        )
        missing = [path for path in [leaf_model_path, lesion_model_path] if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                "Missing segmentation model weights. Set CERES_SEG_LEAF_MODEL_PATH and "
                f"CERES_SEG_LESION_MODEL_PATH or place weights at the default paths: {missing}"
            )
        _segmentation_service = SegmentationSeverityService(
            leaf_model_path=leaf_model_path,
            lesion_model_path=lesion_model_path,
        )
    return _segmentation_service


def get_disease_class_service() -> DiseaseClassService:
    global _disease_class_service
    if _disease_class_service is None:
        model_path = _resolve_model_path(
            "CERES_CLASSIFIER_MODEL_PATH",
            "EfficientNet", "checkpoints", "best_model.pth",
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Missing classifier model weights. Set CERES_CLASSIFIER_MODEL_PATH "
                f"or place the weight file at the default path: {model_path}"
            )
        _disease_class_service = DiseaseClassService(model_path=model_path)
    return _disease_class_service


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.route("/health", methods=["GET"])
def health_check():
    try:
        agent = get_agent()
        return jsonify(
            {
                "status": "healthy",
                "service": "CeresAgents",
                "version": "1.0.0",
                "model": agent.llm.model_name,
            }
        )
    except Exception as exc:
        return jsonify({"status": "unhealthy", "error": str(exc)}), 500


@app.route("/status", methods=["GET"])
def status_check():
    try:
        agent = get_agent()
        return jsonify(
            {
                "code": 200,
                "msg": "ok",
                "status": "running",
                "model": agent.llm.model_name,
            }
        )
    except Exception as exc:
        return error_response(500, str(exc))


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return error_response(400, "no file in request")
        file = request.files["file"]
        if file.filename == "":
            return error_response(400, "empty filename")
        if not allowed_file(file.filename):
            return error_response(400, f"unsupported file type; allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

        filepath = _save_uploaded_file(file)
        filename = os.path.basename(filepath)
        file_size = os.path.getsize(filepath)
        return jsonify(
            {
                "code": 200,
                "msg": "ok",
                "data": {
                    "filename": filename,
                    "path": filepath,
                    "relative_path": f"uploads/{filename}",
                    "size": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "url": f"{request.host_url}uploads/{filename}",
                },
            }
        )
    except Exception as exc:
        return error_response(500, str(exc))


@app.route("/segmentation/severity", methods=["POST"])
def segmentation_severity():
    try:
        save_masks = True
        if "file" not in request.files:
            data = request.get_json(silent=True) or {}
            save_masks = bool(data.get("save_masks", True))
        image_path, error = _resolve_request_image_path()
        if error is not None:
            return error
        if not os.path.exists(image_path):
            return error_response(404, "image file not found")

        service = get_segmentation_service()
        result = service.infer_from_path(image_path)

        mask_paths = {}
        mask_urls = {}
        if save_masks:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            mask_paths = service.save_masks(
                leaf_mask=result["leaf_mask"],
                lesion_mask=result["lesion_mask"],
                overlay=result["overlay"],
                output_dir=SEGMENTATION_OUTPUT_DIR,
                base_name=base_name,
            )
            for key, path in mask_paths.items():
                if path:
                    rel = os.path.relpath(path, UPLOAD_FOLDER)
                    mask_urls[key] = f"{request.host_url}uploads/{rel}"

        return jsonify(
            {
                "code": 200,
                "msg": "ok",
                "data": {
                    "image_path": image_path,
                    "ratio": result["ratio"],
                    "level": result["level"],
                    "leaf_area": result["leaf_area"],
                    "lesion_area": result["lesion_area"],
                    "mask_paths": mask_paths,
                    "mask_urls": mask_urls,
                },
            }
        )
    except Exception as exc:
        return error_response(500, str(exc))


@app.route("/disease/classify", methods=["POST"])
def disease_classify():
    try:
        image_path, error = _resolve_request_image_path()
        if error is not None:
            return error

        topk = 5
        if "file" not in request.files:
            data = request.get_json(silent=True) or {}
            topk = int(data.get("topk", 5))

        if not os.path.exists(image_path):
            return error_response(404, "image file not found")

        service = get_disease_class_service()
        result = service.infer_from_path(image_path, topk=topk)

        if debug_enabled():
            print("[MODEL-RESULT][disease/classify]")
            print(f"  image_path={image_path}")
            print(f"  pred={json.dumps(result.get('pred', {}), ensure_ascii=False)}")
            print(f"  topk={json.dumps(result.get('topk', []), ensure_ascii=False)}")

        return jsonify(
            {
                "code": 200,
                "msg": "ok",
                "data": {
                    "image_path": image_path,
                    "pred": result["pred"],
                    "topk": result["topk"],
                },
            }
        )
    except Exception as exc:
        return error_response(500, str(exc))


@app.route("/diagnose", methods=["POST"])
def diagnose():
    try:
        data = request.get_json(silent=True) or {}
        query = data.get("query")
        if not query:
            return error_response(400, "missing query parameter")

        image_path = data.get("image_path")
        location = data.get("location")
        history = data.get("history", [])
        stream = bool(data.get("stream", False))
        agent = get_agent()

        if stream:
            def generate():
                try:
                    for event in agent.stream_diagnose(
                        query=query,
                        image_path=image_path,
                        location=location,
                        history=history,
                    ):
                        node_name = list(event.keys())[0]
                        state = event[node_name]
                        yield _sse_event({"node": node_name, "state": {}})
                        if node_name == "synthesize" and isinstance(state, dict) and state.get("final_response"):
                            yield _sse_event({"done": True, "response": state["final_response"]})
                except Exception as exc:
                    yield _sse_event({"error": str(exc)})

            return Response(
                stream_with_context(generate()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        result = agent.diagnose(
            query=query,
            image_path=image_path,
            location=location,
            history=history,
        )
        return jsonify(
            {
                "code": 200,
                "msg": "ok",
                "data": {
                    "query": query,
                    "response": result,
                    "metadata": {
                        "image_path": image_path,
                        "location": location,
                        "model": agent.get_llm_config().get("model"),
                        "base_url": agent.get_llm_config().get("base_url"),
                    },
                },
            }
        )
    except Exception as exc:
        return error_response(500, str(exc))


@app.route("/tools/list", methods=["GET"])
def list_tools():
    try:
        agent = get_agent()
        tools_info = []
        for tool in agent.tools:
            tools_info.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "args_schema": str(tool.args_schema) if hasattr(tool, "args_schema") else None,
                }
            )
        return jsonify({"code": 200, "msg": "ok", "count": len(tools_info), "tools": tools_info})
    except Exception as exc:
        return error_response(500, str(exc))


@app.route("/experts/list", methods=["GET"])
def list_experts():
    experts = [
        {"name": "Coordinator", "role": "Coordinator", "description": "Orchestrates workflow and final report"},
        {"name": "Generalist", "role": "Generalist", "description": "Fast track for simple queries"},
        {"name": "Pathologist", "role": "Pathologist", "description": "Symptom analysis and disease identification"},
        {"name": "Physiologist", "role": "Physiologist", "description": "Environment audit"},
        {"name": "Chemist", "role": "Chemist", "description": "Pesticide prescription and compliance checks"},
    ]
    return jsonify({"code": 200, "msg": "ok", "count": len(experts), "experts": experts})


@app.route("/config", methods=["GET"])
def get_config():
    active_model = os.getenv("MODEL_NAME") or os.getenv("LLM_MODEL", "qwen-plus")
    active_base_url = _model_base_url()
    if _agent is not None:
        llm_cfg = _agent.get_llm_config()
        active_model = llm_cfg.get("model") or active_model
        active_base_url = llm_cfg.get("base_url") or active_base_url

    config = {
        "llm_model": active_model,
        "llm_base_url": active_base_url,
        "image_model": os.getenv("IMAGE_MODEL", "qwen-vl-plus"),
        "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "plant_api_url": os.getenv("PLANT_API_BASE_URL", "http://localhost:10001"),
        "upload_folder": os.getenv("UPLOAD_FOLDER", "./uploads"),
    }
    return jsonify({"code": 200, "msg": "ok", "config": config})


@app.route("/uploads/<path:filename>", methods=["GET"])
def get_uploaded_file(filename: str):
    try:
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
    except Exception:
        return error_response(404, "file not found")


@app.errorhandler(404)
def not_found(_error):
    return error_response(404, "not found")


@app.errorhandler(500)
def internal_error(_error):
    return error_response(500, "internal server error")


def main():
    parser = argparse.ArgumentParser(description="CeresAgents tool service")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=10001, help="Port")
    parser.add_argument("--debug", action="store_true", help="Flask debug")
    args = parser.parse_args()

    print(f"CeresAgents tool service listening on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
