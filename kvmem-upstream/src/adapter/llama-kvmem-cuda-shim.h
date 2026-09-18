#pragma once

// CUDA API shim for Metal builds (LLAMA_KVMEM_METAL).
//
// Lets CUDA-typed adapter code compile without a CUDA toolkit. Every CUDA
// runtime path in the adapter has a checked host fallback, and raw device
// pointers never occur on discrete Metal (tensor_ptr returns null for
// non-host buffers), so these stubs are unreachable at runtime. If one ever
// fires it returns an error, which the callers already handle.

#include <cstddef>

typedef void * cudaStream_t;
typedef void * cudaEvent_t;
typedef int cudaError_t;
typedef int cudaMemcpyKind;

#define cudaSuccess 0
#define cudaErrorNotSupported 801

#define cudaMemcpyHostToHost 0
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2
#define cudaMemcpyDeviceToDevice 3
#define cudaMemcpyDefault 4

#define cudaStreamPerThread nullptr
#define cudaStreamNonBlocking 1
#define cudaEventDisableTiming 2

struct cudaPointerAttributes {
    int type;
    int device;
    void * devicePointer;
    void * hostPointer;
};

inline const char * cudaGetErrorString(cudaError_t) { return "CUDA unavailable (Metal build)"; }
inline cudaError_t cudaGetDevice(int *) { return cudaErrorNotSupported; }
inline cudaError_t cudaSetDevice(int) { return cudaErrorNotSupported; }
inline cudaError_t cudaMalloc(void **, size_t) { return cudaErrorNotSupported; }
inline cudaError_t cudaMallocHost(void **, size_t) { return cudaErrorNotSupported; }
inline cudaError_t cudaFree(void *) { return cudaErrorNotSupported; }
inline cudaError_t cudaFreeHost(void *) { return cudaErrorNotSupported; }
inline cudaError_t cudaMemcpy(void *, const void *, size_t, cudaMemcpyKind) { return cudaErrorNotSupported; }
inline cudaError_t cudaMemcpyAsync(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t = nullptr) {
    return cudaErrorNotSupported;
}
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t *, unsigned) { return cudaErrorNotSupported; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaErrorNotSupported; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaErrorNotSupported; }
inline cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned) { return cudaErrorNotSupported; }
inline cudaError_t cudaEventCreate(cudaEvent_t *) { return cudaErrorNotSupported; }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t *, unsigned) { return cudaErrorNotSupported; }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr) { return cudaErrorNotSupported; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaErrorNotSupported; }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaErrorNotSupported; }
inline cudaError_t cudaDeviceSynchronize() { return cudaErrorNotSupported; }
inline cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *, const void *) {
    return cudaErrorNotSupported;
}

// Minimal decls for the GDN replay path (dead on Metal: replay_capacity is
// zero unless the backend is CUDA, so these are never called).
struct ggml_cuda_gdn_replay_layer {
    float * state;
    float * conv;
    const float * key;
    const float * value;
    const float * gate;
    const float * beta;
    const float * conv_input;
};

inline bool ggml_backend_cuda_gdn_fold(
        const struct ggml_cuda_gdn_replay_layer *, int, int, int, void *) {
    return false;
}
