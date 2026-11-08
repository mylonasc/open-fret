"""Pinned upstream parser backend; persistent Node worker per Python process."""

import json
from pathlib import Path
import select
import subprocess

from .alphatex import (AlphaTexParseError, ParsedNote, ParsedTrack, TimedEvent,
                       ALPHATAB_PARSER_VERSION)


class AlphaTabParser:
    def __init__(self):
        path = Path(__file__).resolve().parents[1] / "tools/alphatex_parser/parse.cjs"
        self.process = subprocess.Popen(["node", "--max-old-space-size=512", str(path)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True)
        self.diagnostics = {}

    def parse_text(self, content):
        if self.process.poll() is not None:
            raise RuntimeError("AlphaTab parser worker stopped; run npm ci in tools/alphatex_parser")
        self.process.stdin.write(json.dumps({"content": content}) + "\n")
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 90)[0]:
            self.close()
            raise TimeoutError("AlphaTab parser exceeded 90 seconds for one file")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("AlphaTab worker failed: " + self.process.stderr.read()[-2000:])
        result = json.loads(line)
        if not result["ok"]:
            raise AlphaTexParseError(result["error"])
        self.diagnostics = result["diagnostics"]
        return [ParsedTrack(tuning=t["tuning"], tempo=t["tempo"], events=[
            TimedEvent(start=e["start"], duration=e["duration"],
                       notes=[ParsedNote(**n) for n in e["notes"]]) for e in t["events"]])
                for t in result["tracks"]]

    def parse_file(self, path):
        return self.parse_text(Path(path).read_text(encoding="utf-8"))

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream and not stream.closed:
                stream.close()
