"""
Feed poller service — fetches news, blog and SNS feeds on a schedule and pushes
their items into the retrieval service (POST /ingest), tagged with a sourceType
('news' | 'blog' | 'sns'). It keeps no search index of its own.

Endpoints:
  GET  /health
  GET  /feeds/status        per-feed last poll, item count, failures, last error
  POST /feeds/<name>/poll   poll a feed now

Feeds are configured in feeds.json (path from FEEDS_CONFIG); see feeds.example.json.

Run:
    python3 feedPollerApp.py          (listens on http://localhost:5002)
"""

import logging
import os

from flask import Flask, jsonify

from Config import Config
from DatabaseService.FeedPollerService import FeedPollerService

_ExampleConfig = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feeds.example.json")


def createApp(service: FeedPollerService) -> Flask:
    """Build the Flask app around an already-constructed poller (lets tests inject mocks)."""
    app = Flask(__name__)

    @app.route("/health")
    def health():
        return jsonify({"ok": True})

    @app.route("/feeds/status")
    def feedsStatus():
        return jsonify(service.status())

    @app.route("/feeds/<name>/poll", methods=["POST"])
    def pollNow(name: str):
        if name not in service.feeds:
            return jsonify({"error": f"unknown feed '{name}'"}), 404
        service.requestPoll(name)
        return jsonify({"queued": name}), 202

    return app


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from Controller.RetrievalClient import RetrievalClient
    from DatabaseRepository.FeedStateRepository import FeedStateRepository
    from DatabaseService.FeedPollerService import loadFeeds

    configPath = Config.Feeds.ConfigPath
    if not os.path.exists(configPath) and os.path.exists(_ExampleConfig):
        logging.warning("%s not found; using %s (copy it to feeds.json to customise)", configPath, _ExampleConfig)
        configPath = _ExampleConfig
    feeds = loadFeeds(configPath)
    service = FeedPollerService(feeds, FeedStateRepository(Config.Feeds.StateDbPath), RetrievalClient())
    service.start()

    app = createApp(service)
    app.run(host="0.0.0.0", port=Config.Feeds.PollerPort, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
