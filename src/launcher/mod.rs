use crate::{
    config::{Config, node_id},
    protocol::{Request, Response},
    runtime::{RuntimeHandle, start_runtime},
    transport::{TcpTransport, Transport},
};
use anyhow::{Context, Result, bail, ensure};
use std::{
    ffi::{OsStr, OsString},
    net::{IpAddr, SocketAddr},
    path::Path,
    process::ExitStatus,
    time::Duration,
};

const SITECUSTOMIZE: &str = r#"import os
if os.environ.get("TRAINPOOL_ACTIVE") == "1":
    from trainpool_torch.bootstrap import autoinstall
    autoinstall()
"#;

fn is_python(program: &OsStr) -> bool {
    Path::new(program)
        .file_name()
        .and_then(OsStr::to_str)
        .is_some_and(|name| {
            let name = name.to_ascii_lowercase();
            name == "python" || name.starts_with("python3") || name.starts_with("pypy")
        })
}

async fn python_sdk_available(program: &OsStr) -> Result<()> {
    let output = tokio::process::Command::new(program)
        .args(["-c", "import trainpool_torch"])
        .env_remove("TRAINPOOL_ACTIVE")
        .output()
        .await
        .with_context(|| format!("could not execute Python interpreter {program:?}"))?;
    ensure!(
        output.status.success(),
        "TRAINPOOL_PYTHON_SDK_NOT_FOUND\n\nThe Python interpreter selected by TrainPool does not contain\ntrainpool_torch.\n\nInstall the TrainPool Python adapter in this environment and retry."
    );
    Ok(())
}

async fn request_plan(transport: &TcpTransport, address: SocketAddr) -> Result<Response> {
    let response = transport
        .control(address, &Request::Plan { stages: vec![] })
        .await?;
    response.check()?;
    Ok(response)
}

async fn plan_with_local_runtime(
    config: &Config,
    data_dir: &Path,
    address: SocketAddr,
) -> Result<(Response, Option<RuntimeHandle>)> {
    let transport = TcpTransport {
        config: config.clone(),
    };
    match request_plan(&transport, address).await {
        Ok(response) => return Ok((response, None)),
        Err(probe_error) => {
            if tokio::net::TcpStream::connect(address).await.is_ok() {
                bail!(
                    "TRAINPOOL_LOCAL_DAEMON_UNAVAILABLE: a process is listening on {address}, but TrainPool could not use it (port occupied or authentication mismatch): {probe_error}"
                );
            }
        }
    }

    ensure!(
        matches!(address.ip(), IpAddr::V4(ip) if ip.is_loopback())
            || matches!(address.ip(), IpAddr::V6(ip) if ip.is_loopback()),
        "TRAINPOOL_LOCAL_DAEMON_UNAVAILABLE: refusing to embed a runtime for non-loopback address {address}"
    );
    let mut embedded_config = config.clone();
    if !config.listen.ip().is_unspecified() || config.listen.port() != address.port() {
        embedded_config.listen = address;
        embedded_config.advertise_ip = match address.ip() {
            IpAddr::V4(ip) => Some(ip),
            IpAddr::V6(_) => embedded_config.advertise_ip,
        };
    }
    let discovery_enabled = embedded_config.discovery_enabled;
    let handle = match start_runtime(embedded_config, node_id(data_dir)?).await {
        Ok(handle) => handle,
        Err(start_error) => {
            if let Ok(response) = request_plan(&transport, address).await {
                return Ok((response, None));
            }
            bail!(
                "TRAINPOOL_LOCAL_DAEMON_START_FAILED: could not start the local runtime on {address}: {start_error}"
            );
        }
    };
    let mut errors = Vec::new();
    if discovery_enabled {
        // One heartbeat interval gives already-running RAM providers a chance to
        // announce themselves before the immutable job plan is created.
        tokio::time::sleep(Duration::from_millis(2200)).await;
    }
    for _ in 0..20 {
        match request_plan(&transport, address).await {
            Ok(response) => return Ok((response, Some(handle))),
            Err(error) => errors.push(error),
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    bail!(
        "TRAINPOOL_LOCAL_DAEMON_START_FAILED: local runtime did not become ready: {}",
        errors
            .last()
            .map(ToString::to_string)
            .unwrap_or_else(|| "unknown error".into())
    )
}

fn exit_code(status: ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        128 + status.signal().unwrap_or(1)
    }
    #[cfg(not(unix))]
    1
}

async fn wait_for_child(mut child: tokio::process::Child) -> Result<ExitStatus> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut interrupt = signal(SignalKind::interrupt())?;
        let mut terminate = signal(SignalKind::terminate())?;
        let forwarded = tokio::select! {
            status = child.wait() => return Ok(status?),
            _ = interrupt.recv() => libc::SIGINT,
            _ = terminate.recv() => libc::SIGTERM,
        };
        if let Some(id) = child.id() {
            // SAFETY: `id` came from this live child and `kill` does not retain the pointer-free value.
            unsafe {
                libc::kill(id as libc::pid_t, forwarded);
            }
        }
        return Ok(child.wait().await?);
    }
    #[cfg(not(unix))]
    Ok(child.wait().await?)
}

/// Launch a training command, returning the child's exact shell-style exit code.
pub async fn launch(
    argv: Vec<OsString>,
    config: Config,
    data_dir: &Path,
    address: SocketAddr,
) -> Result<i32> {
    ensure!(!argv.is_empty(), "missing program to execute");
    if is_python(&argv[0]) {
        python_sdk_available(&argv[0]).await?;
    }
    let (plan, embedded) = plan_with_local_runtime(&config, data_dir, address).await?;
    let job_id = plan.data["job_id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("TRAINPOOL_INVALID_PLAN: missing job_id"))?;
    let bootstrap = tempfile::Builder::new()
        .prefix("trainpool-bootstrap-")
        .tempdir()?;
    std::fs::write(bootstrap.path().join("sitecustomize.py"), SITECUSTOMIZE)?;
    let pythonpath = match std::env::var_os("PYTHONPATH") {
        Some(existing) => {
            let mut paths = vec![bootstrap.path().to_path_buf()];
            paths.extend(std::env::split_paths(&existing));
            std::env::join_paths(paths)?
        }
        None => bootstrap.path().as_os_str().to_owned(),
    };
    let mut command = tokio::process::Command::new(&argv[0]);
    command
        .args(&argv[1..])
        .env("TRAINPOOL_ACTIVE", "1")
        .env("TRAINPOOL_ADDRESS", address.to_string())
        .env("TRAINPOOL_CLUSTER_NAME", &config.cluster_name)
        .env("TRAINPOOL_JOB_ID", job_id)
        .env("PYTHONPATH", pythonpath);
    if let Some(secret) = &config.cluster_secret {
        command.env("TRAINPOOL_CLUSTER_SECRET", secret);
    }
    let child = command
        .spawn()
        .context("failed to launch training process")?;
    let status = wait_for_child(child).await?;
    drop(bootstrap);
    if let Some(handle) = embedded {
        handle.shutdown().await;
    }
    Ok(exit_code(status))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn free_address() -> SocketAddr {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap()
    }

    fn local_config(address: SocketAddr) -> Config {
        Config {
            listen: address,
            advertise_ip: Some(std::net::Ipv4Addr::LOCALHOST),
            discovery_enabled: false,
            ram_limit_bytes: Some(1024 * 1024),
            ..Config::default()
        }
    }

    #[tokio::test]
    async fn absent_daemon_starts_embedded_runtime_and_gets_plan() {
        let directory = tempfile::tempdir().unwrap();
        let address = free_address();
        let config = local_config(address);
        let (response, embedded) = plan_with_local_runtime(&config, directory.path(), address)
            .await
            .unwrap();
        assert!(response.data["job_id"].is_string());
        assert!(embedded.is_some());
    }

    #[tokio::test]
    async fn existing_daemon_is_reused() {
        let directory = tempfile::tempdir().unwrap();
        let address = free_address();
        let config = local_config(address);
        let daemon = start_runtime(config.clone(), node_id(directory.path()).unwrap())
            .await
            .unwrap();
        let (response, embedded) = plan_with_local_runtime(&config, directory.path(), address)
            .await
            .unwrap();
        assert!(response.data["job_id"].is_string());
        assert!(embedded.is_none());
        daemon.shutdown().await;
    }
}
