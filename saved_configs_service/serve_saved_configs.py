#!/usr/bin/env python3
"""Expose the repository's saved_configs snapshots through a read-only HTTP API."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlsplit


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 3080
MAX_COMMAND_LENGTH = 4096
MAX_SEARCH_LENGTH = 256
MAX_PAGE_SIZE = 1000
MAX_SEARCH_RESULTS = 100
INDEX_HTML_PATH = Path(__file__).resolve().with_name("index.html")


class ResourceNotFound(Exception):
    """Raised when a requested project or node does not exist."""


def _normalise_command(value: str) -> str:
    return " ".join(value.strip().split())


def _command_key_for(value: str) -> str:
    """Apply the legacy snapshot filename conversion as a best-effort fallback."""

    return re.sub(r"[^0-9A-Za-z_.-]", "_", value.strip())


def _is_within(path: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(parent))) == str(parent)
    except ValueError:
        return False


class SavedConfigsApplication:
    """Read-only access layer for one saved_configs directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"saved_configs directory does not exist: {self.root}")

    def _named_directory(self, parent: Path, name: str, kind: str) -> Path:
        if (
            not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or "\x00" in name
        ):
            raise ResourceNotFound(f"{kind} not found: {name}")

        candidate = (parent / name).resolve()
        if (
            not _is_within(candidate, self.root)
            or candidate.parent != parent.resolve()
            or not candidate.is_dir()
        ):
            raise ResourceNotFound(f"{kind} not found: {name}")
        return candidate

    def project_directory(self, project_id: str) -> Path:
        return self._named_directory(self.root, project_id, "project")

    def node_directory(self, project_id: str, node_id: str) -> Path:
        project = self.project_directory(project_id)
        return self._named_directory(project, node_id, "node")

    def _directories(self, parent: Path) -> List[Path]:
        return sorted(
            (
                entry
                for entry in parent.iterdir()
                if entry.is_dir()
                and _is_within(entry.resolve(), self.root)
                and entry.resolve().parent == parent.resolve()
            ),
            key=lambda entry: entry.name.casefold(),
        )

    def _text_files(self, node: Path) -> List[Path]:
        return sorted(
            (
                entry
                for entry in node.iterdir()
                if entry.is_file()
                and entry.suffix.casefold() == ".txt"
                and _is_within(entry.resolve(), self.root)
                and entry.resolve().parent == node.resolve()
            ),
            key=lambda entry: entry.name.casefold(),
        )

    @staticmethod
    def _first_nonempty_line(path: Path) -> str:
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                for _ in range(20):
                    line = handle.readline()
                    if not line:
                        break
                    if line.strip():
                        return line.strip()
        except OSError:
            return ""
        return ""

    @classmethod
    def _stored_command(cls, path: Path) -> Optional[str]:
        """Return an echoed CLI command only when it agrees with the file key."""

        first_line = cls._first_nonempty_line(path)
        if first_line and _command_key_for(first_line) == path.stem:
            return first_line
        return None


    @staticmethod
    def _read_output(path: Path) -> str:
        raw = path.read_bytes()
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                return raw.decode("gb18030")
            except UnicodeDecodeError:
                return raw.decode("utf-8", errors="replace")

    def list_projects(self) -> List[Dict[str, str]]:
        return [
            {"name": project.name, "project_id": project.name}
            for project in self._directories(self.root)
        ]

    def list_nodes(self, project_id: str) -> List[Dict[str, str]]:
        project = self.project_directory(project_id)
        nodes = self._directories(project)
        if not nodes:
            raise ResourceNotFound(f"project has no node data: {project_id}")
        return [
            {"name": node.name, "status": "started", "node_id": node.name}
            for node in nodes
        ]

    def list_commands(
        self,
        project_id: str,
        node_id: str,
        keyword: str,
        offset: int,
        limit: int,
    ) -> Dict[str, object]:
        node = self.node_directory(project_id, node_id)
        needle = keyword.casefold().strip()
        commands: List[Dict[str, object]] = []
        for path in self._text_files(node):
            command = self._stored_command(path)
            if (
                needle
                and needle not in path.stem.casefold()
                and (command is None or needle not in command.casefold())
            ):
                continue
            commands.append({"command": command, "command_key": path.stem})

        total = len(commands)
        return {
            "project_id": project_id,
            "node_id": node_id,
            "keyword": keyword,
            "total": total,
            "offset": offset,
            "limit": limit,
            "commands": commands[offset : offset + limit],
        }

    def _resolve_command_file(self, node: Path, command: str) -> Optional[Path]:
        files = self._text_files(node)
        by_stem = {path.stem: path for path in files}

        supplied_key = command[:-4] if command.casefold().endswith(".txt") else command
        if supplied_key in by_stem:
            return by_stem[supplied_key]

        generated_key = _command_key_for(command)
        if generated_key in by_stem:
            return by_stem[generated_key]

        normalised = _normalise_command(command)
        normalised_folded = normalised.casefold()
        case_insensitive_match: Optional[Path] = None
        for path in files:
            stored_command = _normalise_command(self._stored_command(path) or "")
            if stored_command == normalised:
                return path
            if stored_command.casefold() == normalised_folded:
                case_insensitive_match = case_insensitive_match or path

        if case_insensitive_match is not None:
            return case_insensitive_match
        return None

    def command_output(self, project_id: str, node_id: str, command: str) -> Dict[str, str]:
        node = self.node_directory(project_id, node_id)
        path = self._resolve_command_file(node, command)
        if path is None:
            return {
                "command": command,
                "output": f"Error: Command '{command}' not found in mock data.",
            }
        return {"command": command, "output": self._read_output(path)}

    @staticmethod
    def _matching_line(path: Path, needle: str) -> Iterable[Tuple[int, str]]:
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if needle in line.casefold():
                        clean = line.rstrip("\r\n")
                        position = clean.casefold().find(needle)
                        start = max(0, position - 160)
                        snippet = clean[start : start + 500]
                        if start:
                            snippet = "…" + snippet
                        if start + 500 < len(clean):
                            snippet += "…"
                        yield line_number, snippet
        except OSError:
            return

    def search(
        self,
        project_id: str,
        query: str,
        node_id: str,
        file_keyword: str,
        limit: int,
    ) -> Dict[str, object]:
        project = self.project_directory(project_id)
        if node_id:
            nodes = [self.node_directory(project_id, node_id)]
        else:
            nodes = self._directories(project)

        query_folded = query.casefold().strip()
        filename_folded = file_keyword.casefold().strip()
        matches: List[Dict[str, object]] = []
        stop_after = limit + 1

        for node in nodes:
            for path in self._text_files(node):
                command = self._stored_command(path)
                if (
                    filename_folded
                    and filename_folded not in path.stem.casefold()
                    and (command is None or filename_folded not in command.casefold())
                ):
                    continue

                if query_folded:
                    line_matches: Iterable[Tuple[Optional[int], str]] = self._matching_line(
                        path, query_folded
                    )
                else:
                    line_matches = [(None, "")]

                for line_number, snippet in line_matches:
                    matches.append(
                        {
                            "node_id": node.name,
                            "command": command,
                            "command_key": path.stem,
                            "line_number": line_number,
                            "line": snippet,
                        }
                    )
                    if len(matches) >= stop_after:
                        break
                if len(matches) >= stop_after:
                    break
            if len(matches) >= stop_after:
                break

        return {
            "project_id": project_id,
            "query": query,
            "node_id": node_id or None,
            "file_keyword": file_keyword,
            "limit": limit,
            "truncated": len(matches) > limit,
            "matches": matches[:limit],
        }


class SavedConfigsHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: Tuple[str, int], app: SavedConfigsApplication) -> None:
        super().__init__(server_address, SavedConfigsRequestHandler)
        self.app = app


class SavedConfigsRequestHandler(BaseHTTPRequestHandler):
    server_version = "SavedConfigsService/1.1"

    @property
    def app(self) -> SavedConfigsApplication:
        return self.server.app  # type: ignore[attr-defined]

    def _send_json(self, status: int, value: object) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_index(self) -> None:
        body = INDEX_HTML_PATH.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'unsafe-inline'; "
            "script-src 'unsafe-inline'; connect-src 'self'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _send_empty(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _query_value(self, query: Dict[str, List[str]], name: str) -> str:
        values = query.get(name, [])
        return values[0] if values else ""

    def _integer_query(
        self,
        query: Dict[str, List[str]],
        name: str,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        raw = self._query_value(query, name)
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"query parameter '{name}' must be an integer") from exc
        if not minimum <= value <= maximum:
            raise ValueError(
                f"query parameter '{name}' must be between {minimum} and {maximum}"
            )
        return value

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlsplit(self.path)
        raw_parts = [part for part in parsed.path.strip("/").split("/") if part]
        parts = [unquote(part) for part in raw_parts]
        query = parse_qs(parsed.query, keep_blank_values=True)

        try:
            if not parts or parts == ["index.html"]:
                self._send_index()
                return

            if parts == ["favicon.ico"]:
                self._send_empty(204)
                return

            if parts == ["healthz"]:
                self._send_json(200, {"status": "ok"})
                return

            if parts == ["v3", "projects"]:
                self._send_json(200, self.app.list_projects())
                return

            if len(parts) == 4 and parts[:2] == ["v3", "projects"]:
                project_id = parts[2]
                if parts[3] == "nodes":
                    self._send_json(200, self.app.list_nodes(project_id))
                    return
                if parts[3] == "search":
                    search_query = self._query_value(query, "q")
                    node_id = self._query_value(query, "node_id")
                    file_keyword = self._query_value(query, "file_keyword")
                    if not search_query and not file_keyword:
                        raise ValueError("at least one of 'q' or 'file_keyword' is required")
                    if len(search_query) > MAX_SEARCH_LENGTH:
                        raise ValueError(f"query parameter 'q' is limited to {MAX_SEARCH_LENGTH} characters")
                    limit = self._integer_query(
                        query, "limit", 20, 1, MAX_SEARCH_RESULTS
                    )
                    self._send_json(
                        200,
                        self.app.search(
                            project_id, search_query, node_id, file_keyword, limit
                        ),
                    )
                    return

            if (
                len(parts) == 6
                and parts[:2] == ["v3", "projects"]
                and parts[3] == "nodes"
            ):
                project_id, node_id, action = parts[2], parts[4], parts[5]
                if action == "commands":
                    keyword = self._query_value(query, "keyword")
                    offset = self._integer_query(query, "offset", 0, 0, 1_000_000)
                    limit = self._integer_query(
                        query, "limit", 200, 1, MAX_PAGE_SIZE
                    )
                    self._send_json(
                        200,
                        self.app.list_commands(
                            project_id, node_id, keyword, offset, limit
                        ),
                    )
                    return

                if action == "command":
                    command = self._query_value(query, "cmd")
                    if not command:
                        raise ValueError("query parameter 'cmd' is required")
                    if len(command) > MAX_COMMAND_LENGTH:
                        raise ValueError(
                            f"query parameter 'cmd' is limited to {MAX_COMMAND_LENGTH} characters"
                        )
                    self._send_json(
                        200, self.app.command_output(project_id, node_id, command)
                    )
                    return

            self._send_json(404, {"error": "route not found"})
        except ResourceNotFound as exc:
            self._send_json(404, {"error": str(exc)})
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except OSError as exc:
            self.log_error("snapshot read failed: %s", exc)
            self._send_json(500, {"error": "failed to read snapshot data"})


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    script_directory = Path(__file__).resolve().parent
    default_root = script_directory.parent / "saved_configs"
    parser = argparse.ArgumentParser(
        description="Serve saved_configs through a read-only GNS3-compatible HTTP API."
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("SAVED_CONFIGS_SERVICE_HOST", DEFAULT_HOST),
        help=f"listen address (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=_port(os.environ.get("SAVED_CONFIGS_SERVICE_PORT", str(DEFAULT_PORT))),
        help=f"listen port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--configs-root",
        type=Path,
        default=Path(os.environ.get("SAVED_CONFIGS_ROOT", str(default_root))),
        help="saved_configs directory (default: repository-root/saved_configs)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        app = SavedConfigsApplication(args.configs_root)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        server = SavedConfigsHTTPServer((args.host, args.port), app)
    except OSError as exc:
        print(f"Error: cannot listen on {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 2

    print(f"Saved configs root: {app.root}")
    print(f"Listening on http://{args.host}:{args.port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping service.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
