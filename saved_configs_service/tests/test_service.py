import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import urlopen


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve_saved_configs import (  # noqa: E402
    SavedConfigsApplication,
    SavedConfigsHTTPServer,
)


class SavedConfigsServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(cls.temporary_directory.name) / "saved_configs"
        node = root / "Campus Network" / "Core_SW_01"
        node.mkdir(parents=True)
        (node / "display_ip_interface_brief.txt").write_text(
            "Vlanif10 10.0.10.1/24 up up\n",
            encoding="utf-8",
        )
        (node / "display_current-configuration___include_Vlanif10.txt").write_text(
            "display current-configuration | include Vlanif10\ninterface Vlanif10\n",
            encoding="utf-8",
        )

        cls.server = SavedConfigsHTTPServer(
            ("127.0.0.1", 0), SavedConfigsApplication(root)
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.temporary_directory.cleanup()

    def get_json(self, path, query=None):
        url = self.base_url + path
        if query:
            url += "?" + urlencode(query)
        with urlopen(url, timeout=5) as response:
            self.assertEqual(response.headers.get_content_charset(), "utf-8")
            return response.status, json.load(response)

    def test_browser_index(self):
        with urlopen(self.base_url + "/", timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers.get_content_type(), "text/html")
            self.assertEqual(response.headers.get_content_charset(), "utf-8")
            html = response.read().decode("utf-8")

        self.assertIn("saved_configs 数据浏览器", html)
        self.assertIn("/v3/projects", html)
        self.assertIn("selectCommand", html)

    def test_health_projects_and_nodes(self):
        status, health = self.get_json("/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(health, {"status": "ok"})

        _, projects = self.get_json("/v3/projects")
        self.assertEqual(
            projects, [{"name": "Campus Network", "project_id": "Campus Network"}]
        )

        project = quote("Campus Network", safe="")
        _, nodes = self.get_json(f"/v3/projects/{project}/nodes")
        self.assertEqual(nodes[0]["node_id"], "Core_SW_01")

    def test_command_lookup_by_cli_and_key(self):
        project = quote("Campus Network", safe="")
        path = f"/v3/projects/{project}/nodes/Core_SW_01/command"

        _, by_cli = self.get_json(path, {"cmd": "display ip interface brief"})
        self.assertIn("10.0.10.1/24", by_cli["output"])

        _, by_key = self.get_json(
            path, {"cmd": "display_current-configuration___include_Vlanif10"}
        )
        self.assertIn("interface Vlanif10", by_key["output"])

        _, missing = self.get_json(path, {"cmd": "display missing"})
        self.assertIn("not found in mock data", missing["output"])

    def test_command_discovery_and_search(self):
        project = quote("Campus Network", safe="")
        base = f"/v3/projects/{project}/nodes/Core_SW_01"
        _, commands = self.get_json(base + "/commands", {"keyword": "Vlanif10"})
        self.assertEqual(commands["total"], 1)
        self.assertEqual(
            commands["commands"][0]["command"],
            "display current-configuration | include Vlanif10",
        )

        _, commands_without_echo = self.get_json(
            base + "/commands", {"keyword": "interface_brief"}
        )
        self.assertIsNone(commands_without_echo["commands"][0]["command"])

        _, search = self.get_json(
            f"/v3/projects/{project}/search",
            {"q": "10.0.10.1", "node_id": "Core_SW_01"},
        )
        self.assertFalse(search["truncated"])
        self.assertEqual(search["matches"][0]["line_number"], 1)
        self.assertEqual(
            search["matches"][0]["command_key"], "display_ip_interface_brief"
        )

    def test_encoded_traversal_is_rejected(self):
        with self.assertRaises(HTTPError) as raised:
            self.get_json("/v3/projects/%2E%2E/nodes")
        self.assertEqual(raised.exception.code, 404)

    def test_repository_root_is_not_exposed(self):
        for path in ("/README.md", "/data", "/experiments", "/saved_configs"):
            with self.subTest(path=path), self.assertRaises(HTTPError) as raised:
                urlopen(self.base_url + path, timeout=5)
            self.assertEqual(raised.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
