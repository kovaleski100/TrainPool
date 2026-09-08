import os
import socket
import subprocess
import time
from pathlib import Path

import pytest
from trainpool_torch import Client

ROOT = Path(__file__).resolve().parents[2]


def free_port(kind=socket.SOCK_STREAM):
    with socket.socket(type=kind) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def cluster(tmp_path_factory):
    root = tmp_path_factory.mktemp("trainpool-nodes")
    binary = os.getenv("TRAINPOOL_BINARY", str(ROOT / "target/debug/trainpool"))
    ports = [free_port(), free_port()]
    discovery_port = free_port(socket.SOCK_DGRAM)
    processes = []
    try:
        for index, port in enumerate(ports):
            directory = root / str(index)
            processes.append(
                subprocess.Popen(
                    [
                        binary,
                        "--data-dir",
                        str(directory),
                        "daemon",
                        "--listen",
                        f"127.0.0.1:{port}",
                        "--advertise-ip",
                        "127.0.0.1",
                        "--multicast-interface",
                        "127.0.0.1",
                        "--discovery-port",
                        str(discovery_port),
                        "--ram-limit-mib",
                        "48",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                clients = [Client(f"127.0.0.1:{port}") for port in ports]
                if (
                    all(len(c.status["nodes"]) == 2 for c in clients)
                    and len({c.status["leadership"]["leader_id"] for c in clients}) == 1
                ):
                    yield clients, root
                    return
            except (OSError, RuntimeError, KeyError):
                pass
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("A test daemon exited unexpectedly")
            time.sleep(0.2)
        raise RuntimeError("Multicast discovery failed within 20 seconds")
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
