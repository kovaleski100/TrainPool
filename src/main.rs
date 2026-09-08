use clap::Parser;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "trainpool=info".into()),
        )
        .json()
        .with_writer(std::io::stderr)
        .init();
    trainpool::cli::execute(trainpool::cli::Cli::parse()).await
}
