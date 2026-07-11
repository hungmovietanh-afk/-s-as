import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI

from main import app as deployed_app
from server import app
from server import hash_owner_token, owner_token_matches, runtime_directory


class ServerConfigurationTests(unittest.TestCase):
    def test_deployment_entrypoint_exports_fastapi_app(self) -> None:
        self.assertIsInstance(deployed_app, FastAPI)
        mounted_apps = [route.app for route in deployed_app.routes if route.path == ""]
        self.assertIn(app, mounted_apps)

    def test_owner_token_is_compared_by_digest(self) -> None:
        digest = hash_owner_token("owner-secret")

        self.assertTrue(owner_token_matches("owner-secret", digest))
        self.assertFalse(owner_token_matches("wrong-secret", digest))

    def test_missing_digest_keeps_loopback_development_open(self) -> None:
        self.assertTrue(owner_token_matches("", ""))

    def test_fly_runtime_uses_persistent_volume(self) -> None:
        environment = {"FLY_APP_NAME": "sep-ai"}
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(runtime_directory(), Path("/data/runtime"))
