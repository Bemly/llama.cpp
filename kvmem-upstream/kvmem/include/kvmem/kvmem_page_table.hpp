#pragma once

// P1 paged-KV page table (NInfer-absorb): logical page identity decoupled from
// the physical GPU slot, per-handout generations, per-session ownership.
// Pure logic, no backend dependency, covered by kvmem host tests.
// One logical page == one store block; the token span of a page is decided by
// block_tokens at the pool layer, not here.

#include <cstdint>
#include <string>
#include <vector>

namespace kvmem {

struct KvmemPageAlloc {
    int32_t slot = -1;
    uint32_t gen = 0; // 0 only when slot == -1; live pages always carry gen >= 1
};

class KvmemPageTable {
  public:
    // live tracks pool membership explicitly: legacy/shared allocations carry
    // logical == -1 (unknown), which must NOT read as "on the free list".
    struct Entry {
        int32_t logical = -1;
        uint32_t gen = 0; // bumped on every hand-out, never reused live
        int32_t owner = -1; // session id (-1 = shared/legacy path)
        bool live = false;
    };

    void reset(uint32_t n_slots) {
        entries_.clear();
        entries_.resize(n_slots);
        free_.clear();
        free_.reserve(n_slots);
        for (int32_t i = static_cast<int32_t>(n_slots) - 1; i >= 0; --i) {
            free_.push_back(i);
        }
        double_free_drops_ = 0;
    }

    KvmemPageAlloc alloc(int32_t owner, int32_t logical) {
        KvmemPageAlloc out;
        if (free_.empty()) {
            return out;
        }
        out.slot = free_.back();
        free_.pop_back();
        Entry & e = entries_[static_cast<uint32_t>(out.slot)];
        if (++e.gen == 0) {
            ++e.gen; // skip wrap: 0 stays reserved for "no page"
        }
        e.logical = logical;
        e.owner = owner;
        e.live = true;
        out.gen = e.gen;
        return out;
    }

    // Wildcards: owner < -1 skips the owner check, gen == 0 skips the
    // generation check. Mismatches and double frees are rejected (counted,
    // never silently applied).
    bool free(int32_t slot, int32_t owner, uint32_t gen) {
        if (slot < 0 || static_cast<uint32_t>(slot) >= entries_.size()) {
            return false;
        }
        Entry & e = entries_[static_cast<uint32_t>(slot)];
        if (!e.live) {
            ++double_free_drops_;
            return false;
        }
        if (owner > -2 && e.owner != owner) {
            return false;
        }
        if (gen != 0 && e.gen != gen) {
            return false;
        }
        e.logical = -1;
        e.owner = -1;
        e.live = false;
        free_.push_back(slot);
        return true;
    }

    uint32_t generation(int32_t slot) const {
        if (slot < 0 || static_cast<uint32_t>(slot) >= entries_.size()) {
            return 0;
        }
        return entries_[static_cast<uint32_t>(slot)].gen;
    }

    int32_t owner(int32_t slot) const {
        if (slot < 0 || static_cast<uint32_t>(slot) >= entries_.size()) {
            return -2;
        }
        return entries_[static_cast<uint32_t>(slot)].owner;
    }

    // Free every page owned by seq; seq < 0 releases the whole pool (legacy
    // shared pages included). Returns the released count.
    uint32_t release_session(int32_t seq) {
        uint32_t n = 0;
        for (uint32_t i = 0; i < entries_.size(); ++i) {
            Entry & e = entries_[i];
            if (e.live && (seq < 0 || e.owner == seq)) {
                e.logical = -1;
                e.owner = -1;
                e.live = false;
                free_.push_back(static_cast<int32_t>(i));
                ++n;
            }
        }
        return n;
    }

    // Record the logical block now resident on a live page. Never clears;
    // clearing happens only on free paths. Best-effort mapping refresh: the
    // table tracks live-ness strictly, logical identity on plan commits.
    bool note_resident(int32_t slot, int32_t logical, int32_t owner) {
        if (slot < 0 || static_cast<uint32_t>(slot) >= entries_.size()) {
            return false;
        }
        Entry & e = entries_[static_cast<uint32_t>(slot)];
        if (!e.live) {
            return false; // not live: a free must have been missed, stay loud
        }
        e.logical = logical;
        e.owner = owner;
        return true;
    }

    // Free-list <-> entry agreement: no duplicates, no leaks, live-ness on the
    // entries (not the logical value: legacy pages carry logical == -1 while
    // live). Cheap enough for TRACE-gated validation only.
    // Returns 0 when valid, else a reason code (1 bad index, 2 duplicate free,
    // 3 live entry on free list, 4 dead entry off the free list, 5 count
    // mismatch).
    int valid_detail() const {
        std::vector<char> seen(entries_.size(), 0);
        for (int32_t s : free_) {
            if (s < 0 || static_cast<uint32_t>(s) >= entries_.size()) {
                return 1;
            }
            if (seen[static_cast<uint32_t>(s)]) {
                return 2;
            }
            seen[static_cast<uint32_t>(s)] = 1;
            if (entries_[static_cast<uint32_t>(s)].live) {
                return 3;
            }
        }
        uint32_t live = 0;
        for (uint32_t i = 0; i < entries_.size(); ++i) {
            if (!seen[i]) {
                if (!entries_[i].live) {
                    return 4;
                }
                ++live;
            }
        }
        if (live + free_.size() != entries_.size()) {
            return 5;
        }
        return 0;
    }

    bool valid() const { return valid_detail() == 0; }

    // One-line state dump for post-mortem (TRACE-gated callers only).
    std::string dump() const {
        std::string s = "slots=" + std::to_string(n_slots()) +
            " free=" + std::to_string(n_free()) + " [";
        for (uint32_t i = 0; i < entries_.size(); ++i) {
            const Entry & e = entries_[i];
            s += std::to_string(i) + ":" + (e.live ? "L" : ".") +
                std::to_string(e.logical) + "/" + std::to_string(e.gen) + " ";
        }
        s += "] free=[";
        for (int32_t f : free_) {
            s += std::to_string(f) + " ";
        }
        return s + "]";
    }

    uint32_t n_slots() const { return static_cast<uint32_t>(entries_.size()); }
    uint32_t n_free() const { return static_cast<uint32_t>(free_.size()); }
    uint32_t double_free_drops() const { return double_free_drops_; }

    int32_t peek() const {
        if (free_.empty()) {
            return -1;
        }
        return free_.back();
    }

  private:
    std::vector<Entry> entries_;
    std::vector<int32_t> free_;
    uint32_t double_free_drops_ = 0;
};

} // namespace kvmem
