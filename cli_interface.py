#!/usr/bin/env python3

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    from ceres_agents import CeresAgentsGraph
except ImportError as exc:
    print(f"[ERROR] Import failed: {exc}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

load_dotenv()


class CeresCliInterface:
    def __init__(self) -> None:
        self.agent: Optional[CeresAgentsGraph] = None
        self.conversation_history: list[Dict[str, Any]] = []
        self.session_id = f"cli-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.uploads_dir = self._resolve_upload_dir()
        self.logs_dir = os.path.join(PROJECT_ROOT, "logs")
        self.log_file = os.path.join(self.logs_dir, f"{self.session_id}.jsonl")
        self.debug_enabled = False

        os.makedirs(self.uploads_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def _resolve_upload_dir(self) -> str:
        env_upload_dir = (os.getenv("UPLOAD_FOLDER") or "").strip()
        if not env_upload_dir:
            return os.path.join(PROJECT_ROOT, "uploads")
        if os.path.isabs(env_upload_dir):
            return os.path.abspath(os.path.expanduser(env_upload_dir))
        return os.path.abspath(os.path.join(PROJECT_ROOT, env_upload_dir))

    def initialize_agent(self) -> bool:
        print("Initializing CeresAgents...")
        try:
            model = os.getenv("MODEL_NAME") or os.getenv("LLM_MODEL", "qwen-plus")
            self.agent = CeresAgentsGraph(model=model, verbose=True)
        except Exception as exc:
            print(f"[ERROR] Init failed: {exc}")
            return False
        print("CeresAgents ready.")
        return True

    def append_log(self, entry: Dict[str, Any]) -> None:
        try:
            with open(self.log_file, "a", encoding="utf-8") as file:
                file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            print(f"[WARN] Log write failed: {exc}")

    def build_log_entry(
        self,
        query: str,
        response: str,
        image_path: Optional[str],
        location: Optional[str],
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "type": "diagnosis_turn",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "turn_id": len(self.conversation_history),
            "input": {
                "query": query,
                "image_path": image_path,
                "location": location,
            },
            "output": {
                "final_response": response,
            },
        }
        if self.agent and hasattr(self.agent, "get_last_structured_log"):
            structured_log = self.agent.get_last_structured_log()
            if structured_log:
                entry["trace"] = structured_log
        return entry

    def display_welcome(self) -> None:
        print("=" * 60)
        print("CeresAgents CLI")
        print("=" * 60)
        print("Commands:")
        print("  /image <filename>   attach an image from uploads/")
        print("  /location <place>   set location context")
        print("  /history            show recent turns")
        print("  /clear              clear local history")
        print("  /debug on|off       toggle verbose output")
        print("  /help               show help")
        print("  /quit               exit")
        print("=" * 60)

    def display_help(self) -> None:
        print("Examples:")
        print("  My tomato leaves have yellow spots. What disease could this be?")
        print("  /image tomato.jpg")
        print("  /location Shandong, China")

    def list_uploaded_images(self) -> list[str]:
        if not os.path.exists(self.uploads_dir):
            return []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
        return [
            file
            for file in os.listdir(self.uploads_dir)
            if any(file.lower().endswith(ext) for ext in image_extensions)
        ]

    def handle_command(self, user_input: str) -> tuple[bool, Optional[str], Optional[str], Optional[str]]:
        if not user_input.startswith("/"):
            return True, user_input, None, None

        parts = user_input[1:].split(" ", 1)
        command = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if command in {"quit", "q"}:
            return False, None, None, None
        if command in {"help", "h"}:
            self.display_help()
            return True, None, None, None
        if command == "clear":
            self.conversation_history.clear()
            print("History cleared.")
            return True, None, None, None
        if command == "debug":
            if args in {"on", "true", "1"}:
                self.debug_enabled = True
                print("Verbose output ON")
            elif args in {"off", "false", "0"}:
                self.debug_enabled = False
                print("Verbose output OFF")
            else:
                print("Usage: /debug on|off")
            return True, None, None, None
        if command == "history":
            if not self.conversation_history:
                print("(empty)")
            else:
                for index, entry in enumerate(self.conversation_history, start=1):
                    print(f"{index}. [{entry['timestamp']}] {entry['query']}")
            return True, None, None, None
        if command == "image":
            if not args:
                images = self.list_uploaded_images()
                if not images:
                    print(f"No images found in {self.uploads_dir}")
                else:
                    for image in images:
                        print(image)
                return True, None, None, None
            image_path = os.path.join(self.uploads_dir, args)
            if not os.path.exists(image_path):
                print(f"Image not found: {args}")
                return True, None, None, None
            return True, None, args, None
        if command == "location":
            if not args:
                print("Usage: /location <place>")
                return True, None, None, None
            return True, None, None, args

        print(f"Unknown command: {command}")
        return True, None, None, None

    def diagnose_with_context(
        self,
        query: str,
        image_path: Optional[str] = None,
        location: Optional[str] = None,
    ) -> Optional[str]:
        if not self.agent:
            return None

        history_context: list[str] = []
        for entry in self.conversation_history[-3:]:
            history_context.append(f"User: {entry['query']}")
            history_context.append(f"Assistant: {entry['response'][:200]}")

        try:
            result = self.agent.diagnose(
                query=query,
                image_path=image_path,
                location=location,
                history=history_context,
                debug=self.debug_enabled,
            )
        except KeyboardInterrupt:
            print("Interrupted.")
            return None
        except Exception as exc:
            print(f"Diagnosis error: {exc}")
            return None

        self.conversation_history.append(
            {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "query": query,
                "response": result,
                "image_path": image_path,
                "location": location,
            }
        )
        self.append_log(self.build_log_entry(query, result, image_path, location))
        return result

    def run(self) -> None:
        if not self.initialize_agent():
            return

        self.display_welcome()
        current_image: Optional[str] = None
        current_location: Optional[str] = None

        try:
            while True:
                prompt = f"[{current_image}] > " if current_image else "> "
                user_input = input(prompt).strip()
                if not user_input:
                    continue

                should_continue, query, new_image, new_location = self.handle_command(user_input)
                if not should_continue:
                    break
                if new_image is not None:
                    current_image = new_image
                    print(f"Image selected: {current_image}")
                    continue
                if new_location is not None:
                    current_location = new_location
                    print(f"Location set: {current_location}")
                    continue
                if query is None:
                    continue

                result = self.diagnose_with_context(
                    query=query,
                    image_path=current_image,
                    location=current_location,
                )
                if result:
                    print("-" * 40)
                    print(result.strip())
                    print("-" * 40)
                current_image = None
        except KeyboardInterrupt:
            print("\nGoodbye.")


def main() -> None:
    if not (os.getenv("MODEL_API_KEY") or os.getenv("DASHSCOPE_API_KEY")):
        print("[WARN] No model API key found in environment.")
    cli = CeresCliInterface()
    cli.run()


if __name__ == "__main__":
    main()
