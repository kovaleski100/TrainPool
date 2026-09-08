# Protocol v1

All control/authentication messages use:

```
u32 big-endian JSON byte length
UTF-8 JSON: {"protocol_version": 1, "message": ...}
```

An 8 MiB control-frame ceiling is checked before allocation. The version is inspected
before decoding the message schema. Unknown operations and fields are rejected by
the request enum. UUIDs are canonical strings and byte counts are unsigned integers.
No pickle, object constructors, executable code or tensor file paths are accepted.

Each TCP connection authenticates and performs one operation. Multiple data connections
can run simultaneously; a long transfer does not occupy a shared control connection.
The daemon limits accepted in-flight connections to 64 and applies 5-second outbound
connect/authentication and 120-second operation deadlines.

## Authentication

The server sends a fresh UUID challenge. The client sends its fresh UUID nonce, cluster
name and optionally a hex HMAC-SHA256 over:

```
client|SERVER_UUID|CLIENT_UUID|CLUSTER_NAME
```

The server verifies the cluster name and MAC, then returns the HMAC of the same
transcript with `server` as the prefix. Both sides explicitly reject a secret mismatch.
Fresh server challenges prevent replaying a previous authentication response.

This authenticates peer participation, not each subsequent byte or each user.
TCP is plaintext and has no active-MITM protection. BLAKE3 detects accidental payload
corruption, not an attacker able to replace both payload and checksum. Use trusted
LANs or an encrypted VPN. Shared-secret holders are authorized cluster participants.
QUIC with certificate validation is the intended future transport.

## Discovery datagram

A datagram has `{payload: STRING, mac: STRING|null}`. `payload` is the exact JSON string
being authenticated with HMAC-SHA256 and contains protocol/trainpool versions,
cluster name, node ID, hostname, OS, architecture, control address/port, election score
and wall-clock timestamp. Receivers reject different clusters, incompatible versions,
invalid addresses, invalid MACs and timestamps more than 30 seconds from local time.
Clock synchronization is therefore required for automatic discovery. Liveness itself
uses monotonic receipt times from successful capability exchanges.

## Control operations

The authoritative schema is [`src/protocol/mod.rs`](../src/protocol/mod.rs).

| Operation | Purpose |
|---|---|
| `status`, `capabilities`, `ping` | Resource/membership snapshots and small probes |
| `exchange`, `inventory` | Capability announcements and owner block inventories |
| `plan`, `job_status` | Leader-issued plan and explicit job failure reporting |
| `allocate` | Leader placement; optional preferred RAM node |
| `allocate_local` | Fenced reservation on the chosen owner |
| `resolve` | Locate current generation using object ID and lease token |
| `commit` | Verify complete object hash and make it readable |
| `renew`, `free` | Extend a live lease or release a block |
| `register`, `publish` | Metadata updates / migration commit and publication |
| `memory_pressure`, `migrate` | Request migration and instruct direct source-to-destination copy |
| `reserve_staging`, `release_staging` | Account for a bounded SDK host buffer |
| `metrics`, `report_metrics` | Local per-job counters and SDK CUDA/prefetch reports |
| `topology`, `benchmark`, `measure`, `probe` | Directed topology and explicit bounded bulk probes |

Responses contain `ok`, `data`, and `error`. Metadata payloads are JSON. Control requests
such as allocation/resolve/plan are forwarded to the elected leader if necessary.
SDK data operations use `direct=false`; the local daemon resolves the owner, then sends
an owner request with `direct=true`. Internal flags are not security boundaries: the
MVP trusts cluster participants and lease holders.

## Block identity and leases

Handles contain UUID, job UUID, size, owner UUID/incarnation, location enum, checksum,
state, random lease token, expiry timestamp, generation and optional tensor metadata.
Location variants are `ram`, `gpu`, and reserved `disk`; the Rust allocator accepts
only RAM. CUDA residency is managed by the SDK, not by a Rust CUDA allocator.

Leases default to 300 seconds. The SDK renews every one-third lease period, including
already-uploaded blocks of an unfinished tensor. Lease tokens are required for all
payload/free/renew operations. Expired allocations and staging reservations are
reclaimed in the monitor. Busy readers/transfers pin their allocations until they
release their references. Committed block bytes are immutable; updates allocate new
blocks. A generation changes only when the same logical block migrates.

`TensorMetadata` explicitly validates dtype, shape, contiguous layout and byte length,
including multiplication overflow. The Python tensor table records complete tensor
shape/dtype and ordered segments; each RAM segment advertises contiguous `uint8`
metadata. This allows a tensor larger than one owner's RAM to span multiple blocks.

## Write sequence

1. `allocate` returns a writing-state handle; the owner has reserved its whole size.
2. Send `write_chunk` with handle, transfer ID, offset, length, total size and BLAKE3
   chunk checksum. Default maximum is 64 MiB; peers negotiate the smallest limit.
3. Owner sends a successful ready response before accepting bytes.
4. Send exactly `length` raw bytes. Owner receives directly into its allocation slice,
   verifies the chunk, advances its incremental full-object hash and acknowledges.
5. Repeat in strictly increasing contiguous offsets with the same transfer ID.
6. `commit` supplies the complete BLAKE3 digest. Incomplete/wrong-hash uploads remain
   unreadable. The returned handle contains the complete digest and ready state.

The SDK divides tensors into independently placeable blocks, each no larger than its
configured block size. Lower-level callers may use larger RAM blocks and stream chunks
into them without creating a second object-sized buffer.

## Read sequence

`read_chunk` carries a lease-bearing handle, transfer UUID, offset and length. The
owner checks state, generation and bounds, then returns checksum, object/transfer IDs,
offset, length and total size followed by exactly `length` raw bytes. The SDK verifies
all fields, the chunk digest and the full block digest. The local daemon relays through
a reserved 64 KiB buffer. There is no leader payload relay for remote ownership.

## Migration and errors

The source pins the original committed allocation, creates an unpublished destination
at `generation + 1`, streams it directly, and obtains checksum-verified commit. It then
performs leader compare-and-set against the previous generation. After publication is
acknowledged, the source releases its old allocation. A lost compare-and-set ACK is
ambiguous: both copies are retained rather than risking deletion of the only copy.
Before metadata commit, failed destination uploads can be discarded safely.

Representative errors: `TRAINPOOL_OUT_OF_CAPACITY`, `TRAINPOOL_CHECKSUM_MISMATCH`,
`TRAINPOOL_DATA_LOST`, `TRAINPOOL_LEASE_EXPIRED`, `TRAINPOOL_LEASE_DENIED`,
`TRAINPOOL_STALE_LEADER`, `TRAINPOOL_JOB_LOST`, `TRAINPOOL_NO_CUDA`,
`TRAINPOOL_PROTOCOL_VERSION`, `TRAINPOOL_AUTH_FAILED`.

The protocol does not implement exactly-once delivery or resumable migration. Callers
must treat failed updates as failed jobs; they must not blindly retry an optimizer
step. The previous committed block is never modified in place.
