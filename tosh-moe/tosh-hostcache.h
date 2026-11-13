// EXPERIMENTAL bounded host expert cache (TOSH_DMOE_HOST_CACHE_MIB). The expert bank tensors hold
// no data: whole experts are read from the model file with pread into a locked RAM cache on
// demand, and every host reader of the bank goes through this cache.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

enum tosh_hostcache_kind {
    TOSH_HC_DECODE = 0,     // decode rows the host computes
    TOSH_HC_WIDE,           // rare prefill rows the host computes
    TOSH_HC_PROMOTE,        // uploads into the VRAM arena
    TOSH_HC_PREFILL,        // prefill bank uploads
    TOSH_HC_VERIFY,
    TOSH_HC_KINDS,
};

bool tosh_hostcache_on(void);
bool tosh_hostcache_is_bank_name(const char * name);

// loader: a bank tensor left unread, with where its bytes are in the file
void tosh_hostcache_note_file_tensor(const void * data, size_t nbytes, int fd, uint64_t file_off);
// a gate/up pair the loader fuses: expert e is gate row block e then up row block e of the file
void tosh_hostcache_note_fused_tensor(const void * data, size_t nbytes, int fd, uint64_t off_gate, uint64_t off_up);

// Dynamic MoE init: layer il's part (0 gate, 1 up, 2 down) is the noted tensor at data
bool tosh_hostcache_bind(int il, int part, const void * data, size_t part_bytes, int n_expert);
// allocates and locks the cache and starts the readers; false with tosh_hostcache_error() set
bool tosh_hostcache_start(int n_layer, int n_expert);

// locks [addr, addr+bytes) a step at a time, waiting while the system reports memory pressure;
// false with a message in err (at least 256 bytes) when the lock fails or the pressure stays up
bool tosh_mlock_paced(void * addr, size_t bytes, char * err, size_t err_len);

// pins experts ids[0..n) of layer il, reading the missing ones; false after a read failure
bool tosh_hostcache_acquire(int il, const int32_t * ids, int n, int kind);
const char * tosh_hostcache_part(int il, int e, int part);
void tosh_hostcache_release(int il, const int32_t * ids, int n);

// an expert promoted to VRAM keeps its RAM copy first in line for eviction (drop); one leaving
// VRAM without a copy is read back in the background, behind demand reads (prefetch)
void tosh_hostcache_drop(int il, int e);
void tosh_hostcache_prefetch(int il, int e);
// the same expert, when the caller copies its bytes out of VRAM instead: true when a slot waits
// for them at tosh_hostcache_part(); done publishes it, or falls back to the file read
bool tosh_hostcache_store(int il, int e);
void tosh_hostcache_store_done(int il, int e, bool ok);

// bank tensor data pointer -> layer and part, for uploads that only know the source pointer
bool tosh_hostcache_lookup(const void * data, int * il, int * part);

// full bank loaded: expert e of a noted tensor read with pread equals the loaded bytes
bool tosh_hostcache_bytecheck(const void * data, size_t part_bytes, int e);
int  tosh_hostcache_n_files(void);

const char * tosh_hostcache_error(void);
uint64_t     tosh_hostcache_cold_reads(void);
void         tosh_hostcache_report(void);
// the server finished its warm-up and is about to be ready: warm-up fill stops, reuse is counted
bool         tosh_hostcache_warmup_done(void);   // false: the cache could not be locked, nothing may run

#ifdef __cplusplus
}
#endif
