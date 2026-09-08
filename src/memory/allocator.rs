use anyhow::{Result, ensure};
use std::collections::BTreeMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use uuid::Uuid;

#[derive(Default)]
pub struct JobRamAccounting {
    pub owned: AtomicU64,
    pub peak: AtomicU64,
}

#[derive(Default)]
pub struct RamAccounting {
    pub owned: AtomicU64,
    pub budget: AtomicU64,
    pub peak: AtomicU64,
    pub jobs: Mutex<BTreeMap<Uuid, Arc<JobRamAccounting>>>,
}
pub struct Reservation {
    accounting: Arc<RamAccounting>,
    bytes: u64,
    job: Option<Arc<JobRamAccounting>>,
}
impl RamAccounting {
    pub fn reserve(self: &Arc<Self>, bytes: u64) -> Result<Reservation> {
        let result = self
            .owned
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |used| {
                used.checked_add(bytes)
                    .filter(|new| *new <= self.budget.load(Ordering::SeqCst))
            });
        ensure!(
            result.is_ok(),
            "TRAINPOOL_OUT_OF_CAPACITY: RAM budget cannot satisfy {bytes} bytes"
        );
        self.peak
            .fetch_max(result.unwrap() + bytes, Ordering::Relaxed);
        Ok(Reservation {
            accounting: self.clone(),
            bytes,
            job: None,
        })
    }
    pub fn reserve_payload(self: &Arc<Self>, job_id: Uuid, bytes: u64) -> Result<Reservation> {
        let mut reservation = self.reserve(bytes)?;
        let job = self
            .jobs
            .lock()
            .expect("RAM accounting lock")
            .entry(job_id)
            .or_default()
            .clone();
        let current = job.owned.fetch_add(bytes, Ordering::SeqCst) + bytes;
        job.peak.fetch_max(current, Ordering::Relaxed);
        reservation.job = Some(job);
        Ok(reservation)
    }
    pub fn used(&self) -> u64 {
        self.owned.load(Ordering::SeqCst)
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        if let Some(job) = &self.job {
            job.owned.fetch_sub(self.bytes, Ordering::SeqCst);
        }
        self.accounting
            .owned
            .fetch_sub(self.bytes, Ordering::SeqCst);
    }
}
