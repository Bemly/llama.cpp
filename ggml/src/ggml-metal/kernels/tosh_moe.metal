#include <metal_stdlib>
using namespace metal;

// Mixed expert execution. publish copies a layer's activation and routing into a host-visible
// stage and writes each routed expert's resident slot, -1 for the ones the host runs.
struct tosh_dmoe_publish_args {
    int32_t  n_embd;
    int32_t  n_used;
    uint32_t off_ids;
    uint32_t off_slots;
    uint64_t nb_cur1;
    uint64_t nb_ids1;
    int32_t  il;
    int32_t  mask;   // test only: send router positions to the host whatever their residency
};

kernel void tosh_dmoe_publish(
        constant tosh_dmoe_publish_args & args [[buffer(0)]],
        device const char    * cur       [[buffer(1)]],
        device const char    * ids       [[buffer(2)]],
        device const int32_t * residency [[buffer(3)]],
        device       int32_t * slots     [[buffer(4)]],
        device       char    * stage     [[buffer(5)]],
        uint  t    [[threadgroup_position_in_grid]],
        uint  tpit [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]]) {
    device const float4 * src = (device const float4 *) (cur + t*args.nb_cur1);
    device       float4 * act = (device float4 *) stage + t*(args.n_embd/4);
    for (int i = tpit; i < args.n_embd/4; i += ntg) {
        act[i] = src[i];
    }
    if ((int) tpit < args.n_used) {
        const int32_t e = ((device const int32_t *) (ids + t*args.nb_ids1))[tpit];
        int32_t s = residency[e];
        const int j = tpit;
        const bool host = args.mask == 1 ||
                          (args.mask == 2 && (j & 1)) ||
                          (args.mask == 3 && j >= args.n_used/2) ||
                          (args.mask == 4 && (((uint32_t) (e*2654435761u) ^ (uint32_t) (args.il*40503u + t*977u)) >> 16 & 1));
        if (host) {
            s = -1;
        }
        const int k = t*args.n_used + tpit;
        slots[k] = s;
        ((device int32_t *) (stage + args.off_ids))[k]   = e;
        ((device int32_t *) (stage + args.off_slots))[k] = s;
    }
}

// join takes each routed expert's output from the device when it is resident and from the
// host stage otherwise, keeping the router's order.
struct tosh_dmoe_join_args {
    int32_t  n_embd;
    int32_t  n_used;
    uint32_t off_out;
};

kernel void tosh_dmoe_join(
        constant tosh_dmoe_join_args & args [[buffer(0)]],
        device const float4  * gpu   [[buffer(1)]],
        device const int32_t * slots [[buffer(2)]],
        device const char    * stage [[buffer(3)]],
        device       float4  * dst   [[buffer(4)]],
        uint  k    [[threadgroup_position_in_grid]],
        uint  tpit [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]]) {
    const int n4 = args.n_embd/4;
    device const float4 * host = (device const float4 *) (stage + args.off_out);
    device const float4 * src = slots[k] >= 0 ? gpu + k*n4 : host + k*n4;
    for (int i = tpit; i < n4; i += ntg) {
        dst[k*n4 + i] = src[i];
    }
}

// Wide batches. publish_wide_act copies the chunk's activation into the stage; publish_wide_ids
// counts each expert's rows in the chunk and sends the rows of rarely routed, non-resident
// experts to the host (-1) so the upload skips them. One threadgroup sees the whole chunk.
struct tosh_dmoe_wide_args {
    int32_t  n_embd;
    int32_t  n_used;
    int32_t  n_tokens;
    int32_t  n_expert;
    int32_t  rows;
    int32_t  max_host;
    int32_t  tail;
    int32_t  pad;
    uint32_t off_ids;
    uint32_t off_rmp;
    uint32_t off_map;
    uint32_t off_out;
    uint64_t nb_cur1;
    uint64_t nb_ids1;
};

kernel void tosh_dmoe_publish_wide_act(
        constant tosh_dmoe_wide_args & args [[buffer(0)]],
        device const char * cur   [[buffer(1)]],
        device       char * stage [[buffer(2)]],
        uint  t    [[threadgroup_position_in_grid]],
        uint  tpit [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]]) {
    device const float4 * src = (device const float4 *) (cur + t*args.nb_cur1);
    device       float4 * act = (device float4 *) stage + t*(args.n_embd/4);
    for (int i = tpit; i < args.n_embd/4; i += ntg) {
        act[i] = src[i];
    }
}

kernel void tosh_dmoe_publish_wide_ids(
        constant tosh_dmoe_wide_args & args [[buffer(0)]],
        device const char    * ids       [[buffer(1)]],
        device const int32_t * residency [[buffer(2)]],
        device       int32_t * dst       [[buffer(3)]],
        device       char    * stage     [[buffer(4)]],
        threadgroup atomic_uint * hist   [[threadgroup(0)]],
        uint  tpit [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]]) {
    threadgroup atomic_uint * n_host = hist + args.n_expert;
    for (int e = tpit; e < args.n_expert; e += ntg) {
        atomic_store_explicit(&hist[e], 0u, memory_order_relaxed);
    }
    if (tpit == 0) {
        atomic_store_explicit(n_host, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int n = args.n_tokens*args.n_used;
    for (int i = tpit; i < n; i += ntg) {
        const int32_t e = ((device const int32_t *) (ids + (i/args.n_used)*args.nb_ids1))[i % args.n_used];
        atomic_fetch_add_explicit(&hist[e], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device int32_t * s_ids = (device int32_t *) (stage + args.off_ids);
    device int32_t * s_rmp = (device int32_t *) (stage + args.off_rmp);
    for (int i = tpit; i < n; i += ntg) {
        const int32_t e = ((device const int32_t *) (ids + (i/args.n_used)*args.nb_ids1))[i % args.n_used];
        int32_t r = e;
        if ((int) atomic_load_explicit(&hist[e], memory_order_relaxed) <= args.rows && residency[e] < 0 &&
            i/args.n_used < args.n_tokens - args.tail) {
            // the host stage holds a bounded number of rows; past it the GPU keeps them
            if ((int) atomic_fetch_add_explicit(n_host, 1u, memory_order_relaxed) < args.max_host) {
                r = -1;
            }
        }
        dst[i]   = r;
        s_ids[i] = e;
        s_rmp[i] = r;
    }
}

// Writes the host's rows over the positions the GPU skipped; the rest stays as computed.
kernel void tosh_dmoe_join_wide(
        constant tosh_dmoe_wide_args & args [[buffer(0)]],
        device const char   * stage [[buffer(1)]],
        device       float4 * dst   [[buffer(2)]],
        uint  k    [[threadgroup_position_in_grid]],
        uint  tpit [[thread_position_in_threadgroup]],
        uint  ntg  [[threads_per_threadgroup]]) {
    const int32_t r = ((device const int32_t *) (stage + args.off_map))[k];
    if (r < 0) {
        return;
    }
    const int n4 = args.n_embd/4;
    device const float4 * host = (device const float4 *) (stage + args.off_out) + (size_t) r*n4;
    for (int i = tpit; i < n4; i += ntg) {
        dst[(size_t) k*n4 + i] = host[i];
    }
}

// Prefill routing for the residency policy: the ids of every token, packed, into a host ring.
struct tosh_dmoe_observe_args {
    int32_t  n_used;
    int32_t  n_tokens;
    uint64_t nb1;
};

kernel void tosh_dmoe_observe(
        constant tosh_dmoe_observe_args & args [[buffer(0)]],
        device const char    * ids  [[buffer(1)]],
        device       int32_t * ring [[buffer(2)]],
        uint  t    [[threadgroup_position_in_grid]],
        uint  tpit [[thread_position_in_threadgroup]]) {
    for (int j = tpit; j < args.n_used; j += 32) {
        ring[t*args.n_used + j] = ((device const int32_t *) (ids + t*args.nb1))[j];
    }
}

// y + x*sigmoid(g) [+ r] over float4, g one value per row: the gated shared expert and the sums
// that close a MoE block
struct tosh_gate_add_args {
    int32_t ne0_4;
    int32_t n_4;
    int32_t has_res;
};

kernel void tosh_gate_add_f32(
        constant tosh_gate_add_args & args,
        device const float  * g,
        device const float4 * x,
        device const float4 * y,
        device const float4 * r,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    if ((int) tpig >= args.n_4) {
        return;
    }
    const float s = 1.0f/(1.0f + exp(-g[tpig/args.ne0_4]));
    float4 v = y[tpig] + x[tpig]*s;
    if (args.has_res) {
        v += r[tpig];
    }
    dst[tpig] = v;
}
