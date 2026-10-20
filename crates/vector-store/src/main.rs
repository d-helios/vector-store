/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::anyhow;
use clap::Parser;
use rustls::crypto::aws_lc_rs;
use std::io::IsTerminal;
use std::os::fd::AsFd;
use std::os::unix::fs::MetadataExt;
use std::process::ExitCode;
use tracing::Event;
use tracing::Level;
use tracing::Subscriber;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::fmt::FmtContext;
use tracing_subscriber::fmt::FormatEvent;
use tracing_subscriber::fmt::FormatFields;
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;
use vector_store::ConfigManager;
use vector_store::Info;

#[derive(Parser)]
#[clap(version)]
struct Args {}

/// Prefixes each log line with an sd-daemon `<N>` priority so journald records the real level
/// instead of the unit's default (info) for everything written to stdout.
struct SyslogPrefix<F> {
    enabled: bool,
    inner: F,
}

fn syslog_priority(level: &Level) -> u8 {
    match *level {
        Level::ERROR => 3,
        Level::WARN => 4,
        Level::INFO => 6,
        Level::DEBUG | Level::TRACE => 7,
    }
}

impl<S, N, F> FormatEvent<S, N> for SyslogPrefix<F>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
    F: FormatEvent<S, N>,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> std::fmt::Result {
        if self.enabled {
            write!(writer, "<{}>", syslog_priority(event.metadata().level()))?;
        }
        self.inner.format_event(ctx, writer, event)
    }
}

/// systemd sets JOURNAL_STREAM to "<dev>:<ino>" of the journal stream, but children inherit it
/// after redirecting stdout, so check it still names our stdout (systemd.exec(5)).
fn stdout_is_journal() -> bool {
    let Some(stream) = std::env::var_os("JOURNAL_STREAM") else {
        return false;
    };
    let Ok(meta) = std::io::stdout()
        .as_fd()
        .try_clone_to_owned()
        .and_then(|fd| std::fs::File::from(fd).metadata())
    else {
        return false;
    };
    stream.to_str() == Some(format!("{}:{}", meta.dev(), meta.ino()).as_str())
}

/// Logs panics through tracing so journald records them at error priority; the default hook still
/// prints the full message and backtrace afterwards.
fn log_panics() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let payload = info.payload();
        let msg = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("Box<dyn Any>");
        let location = info.location().map(ToString::to_string).unwrap_or_default();
        tracing::error!("panicked at {location}: {msg}");
        default_hook(info);
    }));
}

fn dotenvy_to_std_var(key: &str) -> anyhow::Result<String> {
    Ok(dotenvy::var(key)?)
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            // `{err:#}` keeps the whole anyhow chain on one line, so journald keeps its priority
            if tracing::dispatcher::has_been_set() {
                tracing::error!("{err:#}");
            } else {
                // failed before logging was initialized
                eprintln!("Error: {err:?}");
            }
            ExitCode::FAILURE
        }
    }
}

// Index creating/querying is CPU bound task, so that vector-store uses rayon ThreadPool for them.
// From the start there was no need (network traffic seems to be not so high) to support more than
// one thread per network IO bound tasks.
fn run() -> anyhow::Result<()> {
    // Initialize logging early, before loading configuration, disable colors will be read twice
    let disable_colors: bool = dotenvy::var("VECTOR_STORE_DISABLE_COLORS")
        .unwrap_or("false".to_string())
        .trim()
        .parse()
        .or(Err(anyhow!(
            "Unable to parse VECTOR_STORE_DISABLE_COLORS env (true/false)"
        )))?;

    // Disable ANSI colors when explicitly requested or when stdout is not a terminal
    // (e.g. output is redirected to a file)
    let use_ansi = !disable_colors && std::io::stdout().is_terminal();
    let journal = stdout_is_journal();

    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().or_else(|_| EnvFilter::try_new("info"))?)
        .with(
            fmt::layer()
                .with_ansi(use_ansi)
                .event_format(SyslogPrefix {
                    enabled: journal,
                    inner: fmt::format().with_target(false),
                }),
        )
        .init();

    if journal {
        log_panics();
    }

    aws_lc_rs::default_provider()
        .install_default()
        .expect("install aws-lc-rs crypto provider");

    _ = dotenvy::dotenv();

    // Load configuration early to get disable_colors for logging setup
    let config_future = vector_store::load_config(dotenvy_to_std_var);
    let loaded_config = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(config_future)?;

    _ = Args::parse();

    tracing::info!("Starting {} version {}", Info::name(), Info::version());

    // Create ConfigManager with initial configuration
    let (config_manager, config_rx) = ConfigManager::new(loaded_config);
    let config = config_rx.borrow().clone();

    let threads = config.threads;

    vector_store::block_on(threads, async move || {
        // Start SIGHUP handler now that we're in the Tokio runtime
        config_manager.start(dotenvy_to_std_var);

        let node_state = vector_store::new_node_state().await;

        let opensearch_addr = config.opensearch_addr.clone();

        let index_factory = if let Some(addr) = opensearch_addr {
            tracing::info!("Using OpenSearch index factory at {addr}");
            vector_store::new_index_factory_opensearch(addr, config_rx.clone())?
        } else {
            tracing::info!("Using Usearch index factory");
            vector_store::new_index_factory_usearch(config_rx.clone())?
        };

        let internals = vector_store::new_internals();
        let db_actor =
            vector_store::new_db(node_state.clone(), internals.clone(), config_rx.clone()).await?;

        let (_server_actor, addr) = vector_store::run(
            node_state,
            db_actor,
            internals,
            index_factory,
            config_rx.clone(),
        )
        .await?;
        tracing::info!("listening on {addr}");

        vector_store::wait_for_shutdown().await;

        anyhow::Ok(())
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syslog_priority_matches_sd_daemon_levels() {
        assert_eq!(syslog_priority(&Level::ERROR), 3);
        assert_eq!(syslog_priority(&Level::WARN), 4);
        assert_eq!(syslog_priority(&Level::INFO), 6);
        assert_eq!(syslog_priority(&Level::DEBUG), 7);
        assert_eq!(syslog_priority(&Level::TRACE), 7);
    }
}
