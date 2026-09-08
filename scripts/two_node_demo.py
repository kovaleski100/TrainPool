"""Launch identical binaries, exercise multicast/RAM/benchmark, then kill leader."""

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "examples"))
from memory_demo import demonstrate  # noqa: E402
from trainpool_torch import Client, TrainPoolError  # noqa: E402


def port(kind=socket.SOCK_STREAM):
    with socket.socket(type=kind) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default=str(ROOT / "target/release/trainpool"))
    args = parser.parse_args()
    ports = [port(), port()]
    discovery = port(socket.SOCK_DGRAM)
    processes = []
    with tempfile.TemporaryDirectory(prefix="trainpool-demo-") as directory:
        root = Path(directory)
        try:
            for index in range(2):
                processes.append(
                    subprocess.Popen(
                        [
                            args.binary,
                            "--data-dir",
                            str(root / str(index)),
                            "daemon",
                            "--listen",
                            f"127.0.0.1:{ports[index]}",
                            "--advertise-ip",
                            "127.0.0.1",
                            "--multicast-interface",
                            "127.0.0.1",
                            "--discovery-port",
                            str(discovery),
                            "--ram-limit-mib",
                            "4" if index == 0 else "24",
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                )
            deadline = time.monotonic() + 20
            while True:
                try:
                    clients = [Client(f"127.0.0.1:{p}") for p in ports]
                    if (
                        all(len(c.status["nodes"]) == 2 for c in clients)
                        and clients[0].status["leadership"] == clients[1].status["leadership"]
                    ):
                        break
                except (OSError, RuntimeError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("Multicast discovery timed out")
                time.sleep(0.2)
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            summary = demonstrate(clients[0])
            topology = clients[0].control("benchmark", bytes=1024 * 1024)
            leader = clients[0].status["leadership"]["leader_id"]
            leader_index = next(i for i, c in enumerate(clients) if c.local_node == leader)
            survivor = clients[1 - leader_index]
            job = survivor.control("plan", stages=[])
            lost = survivor.put(b"single-copy-data", job["job_id"], preferred_node=leader)
            processes[leader_index].kill()
            processes[leader_index].wait()
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                status = survivor.control("status")
                if len(status["nodes"]) == 1 and status["leadership"]["leader_id"] == survivor.local_node:
                    break
                time.sleep(0.25)
            else:
                raise RuntimeError("Leader failover timed out")
            try:
                survivor.control("resolve", id=lost["id"], lease_token=lost["lease_token"])
            except TrainPoolError as error:
                assert "TRAINPOOL_DATA_LOST" in str(error)
                lost_error = str(error)
            else:
                raise AssertionError("Lost block incorrectly remained readable")
            after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            assert before == after, "Tensor operation unexpectedly modified node metadata directories"
            summary.update(
                {
                    "discovery": "UDP multicast, no seeds",
                    "directed_links_measured": len(topology["links"]),
                    "leader_failover": True,
                    "lost_block_error": lost_error,
                    "tensor_files_written": 0,
                }
            )
            print(json.dumps(summary, indent=2))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
