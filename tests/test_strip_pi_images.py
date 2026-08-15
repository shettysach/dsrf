from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_strip_pi_images_preserves_non_image_session_data(tmp_path: Path) -> None:
    source = tmp_path / "session.jsonl"
    destination = tmp_path / "readable.jsonl"
    source.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Observation 2"},
                                {
                                    "type": "image",
                                    "data": "large-base64-payload",
                                    "mimeType": "image/jpeg",
                                },
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Walk left."}],
                        },
                    }
                ),
            )
        )
        + "\n"
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/strip_pi_images.py",
            str(source),
            "--output",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    records = [json.loads(line) for line in destination.read_text().splitlines()]
    image = records[0]["message"]["content"][1]
    assert image == {"type": "image", "omitted": True, "mimeType": "image/jpeg"}
    assert records[1]["message"]["content"][0]["text"] == "Walk left."
    assert "Removed 1 image payload(s)." in result.stderr
