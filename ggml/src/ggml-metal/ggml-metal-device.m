#include <os/lock.h>
#import "ggml-metal-device.h"
#import "ggml-metal-fusion.h"

#include <sys/resource.h>

#import "ggml-impl.h"
#import "ggml-backend-impl.h"
#import "ggml-metal-impl.h"

#ifdef TOSH_ENABLE_DYNAMIC_MOE
#import "tosh-dmoe.h"
#endif
#import "ggml-metal-common.h"

#include <Foundation/Foundation.h>

#include <Metal/Metal.h>

#include <stdatomic.h>
#include <unistd.h>

#import <IOKit/IOKitLib.h>
#import <CommonCrypto/CommonDigest.h>
#import <CoreFoundation/CoreFoundation.h>

// The driver's own VRAM counters for this device: free, and in use by every process. Metal only
// reports this process's allocations and a working-set ceiling that host mappings also fill.
static bool ggml_metal_vram_stats(id<MTLDevice> d, int64_t * vfree, int64_t * vused) {
    io_service_t e = IOServiceGetMatchingService(MACH_PORT_NULL, IORegistryEntryIDMatching(d.registryID));
    if (e == 0) {
        return false;
    }
    CFDictionaryRef ps = (CFDictionaryRef) IORegistryEntryCreateCFProperty(e, CFSTR("PerformanceStatistics"), kCFAllocatorDefault, 0);
    IOObjectRelease(e);
    if (ps == NULL) {
        return false;
    }
    CFNumberRef f = (CFNumberRef) CFDictionaryGetValue(ps, CFSTR("vramFreeBytes"));
    CFNumberRef u = (CFNumberRef) CFDictionaryGetValue(ps, CFSTR("inUseVidMemoryBytes"));
    const bool ok = f != NULL && u != NULL &&
        CFNumberGetValue(f, kCFNumberSInt64Type, vfree) && CFNumberGetValue(u, kCFNumberSInt64Type, vused);
    CFRelease(ps);
    return ok;
}

#ifdef TOSH_ENABLE_DYNAMIC_MOE
static id<MTLDevice> g_vram_dev = nil;
static bool ggml_metal_vram_query(int64_t * vfree, int64_t * vused) {
    return g_vram_dev != nil && ggml_metal_vram_stats(g_vram_dev, vfree, vused);
}
#endif


#ifndef TARGET_OS_VISION
#define TARGET_OS_VISION 0
#endif

// create residency sets only on macOS >= 15.0
#if !TARGET_CPU_X86_64 && TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000 || \
    TARGET_OS_IOS && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_TV && __TV_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_VISION && __VISION_OS_VERSION_MAX_ALLOWED >= 200000
#define GGML_METAL_HAS_RESIDENCY_SETS 1
#endif

// overload of MTLGPUFamilyMetalX (not available in some environments)
static const NSInteger MTLGPUFamilyMetal3_GGML = 5001;
static const NSInteger MTLGPUFamilyMetal4_GGML = 5002;

// MTLLanguageVersion4_0 is not present in older SDKs
static const NSUInteger MTLLanguageVersion4_0_GGML = 4 << 16;

#if !GGML_METAL_EMBED_LIBRARY
// Here to assist with NSBundle Path Hack
@interface GGMLMetalClass : NSObject
@end
@implementation GGMLMetalClass
@end
#endif

//
// MTLFunctionConstantValues wrapper
//

struct ggml_metal_cv {
    MTLFunctionConstantValues * obj;
};

ggml_metal_cv_t ggml_metal_cv_init(void) {
    ggml_metal_cv_t res = calloc(1, sizeof(struct ggml_metal_cv));

    res->obj = [[MTLFunctionConstantValues alloc] init];

    return res;
}

void ggml_metal_cv_free(ggml_metal_cv_t cv) {
    [cv->obj release];
    free(cv);
}

void ggml_metal_cv_set_int16(ggml_metal_cv_t cv, int16_t value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeShort atIndex:idx];
}

void ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeInt atIndex:idx];
}

void ggml_metal_cv_set_bool(ggml_metal_cv_t cv, bool value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeBool atIndex:idx];
}

//
// MTLComputePipelineState wrapper
//

struct ggml_metal_pipeline {
    id<MTLComputePipelineState> obj;
};

ggml_metal_pipeline_t ggml_metal_pipeline_init(void) {
    ggml_metal_pipeline_t res = calloc(1, sizeof(struct ggml_metal_pipeline));

    *res = (struct ggml_metal_pipeline) {
        /*.obj  =*/ nil,
    };

    return res;
}

void ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline) {
    [pipeline->obj release];

    free(pipeline);
}

int ggml_metal_pipeline_max_theads_per_threadgroup(struct ggml_metal_pipeline_with_params pipeline) {
    return pipeline.pipeline->obj.maxTotalThreadsPerThreadgroup;
}

//
// MTLLibrary collection (one library per op-source, compiled separately)
//

// Single source of truth for the per-kind metal libraries. The order here
// defines the enum values and every per-kind table below, so adding a library
// is a one-line change here (plus adding its source to CMakeLists.txt).
//   X(suffix, name): name is both the kernels/<name>.metal basename and the
//   ggml_metallib_<name>_{start,end} embed-symbol stem.
#define GGML_METAL_LIBS \
    X(FA,              fa)             \
    X(FA_AUX,          fa_aux)         \
    X(FA_F16,          fa_f16)         \
    X(FA_F32,          fa_f32)         \
    X(FA_Q4_0,         fa_q4_0)        \
    X(FA_Q4_1,         fa_q4_1)        \
    X(FA_Q5_0,         fa_q5_0)        \
    X(FA_Q5_1,         fa_q5_1)        \
    X(FA_Q8_0,         fa_q8_0)        \
    X(FA_VEC_F16,      fa_vec_f16)     \
    X(FA_VEC_F32,      fa_vec_f32)     \
    X(FA_VEC_Q4_0,     fa_vec_q4_0)    \
    X(FA_VEC_Q4_1,     fa_vec_q4_1)    \
    X(FA_VEC_Q5_0,     fa_vec_q5_0)    \
    X(FA_VEC_Q5_1,     fa_vec_q5_1)    \
    X(FA_VEC_Q8_0,     fa_vec_q8_0)    \
    X(FA_W64,          fa_w64)         \
    X(MUL_MV,          mul_mv)         \
    X(MUL_MV_MMA,      mul_mv_mma)     \
    X(MUL_MV_W64,      mul_mv_w64)     \
    X(MUL_MM,          mul_mm)         \
    X(MUL_MM_W64,      mul_mm_w64)     \
    X(QUANTIZE,        quantize)       \
    X(SOFTMAX,         softmax)        \
    X(NORM,            norm)           \
    X(UNARY,           unary)          \
    X(BINBCAST,        binbcast)       \
    X(REDUCE,          reduce)         \
    X(TRI,             tri)            \
    X(SSM,             ssm)            \
    X(WKV,             wkv)            \
    X(GATED_DELTA_NET, gated_delta_net)\
    X(SOLVE_TRI,       solve_tri)      \
    X(ROPE,            rope)           \
    X(CONV,            conv)           \
    X(UPSCALE,         upscale)        \
    X(ARGSORT,         argsort)        \
    X(POOL,            pool)           \
    X(TURBO,           turbo)          \
    X(TOSH_MOE,        tosh_moe)       \
    X(MISC,            misc)

enum ggml_metal_lib_kind {
#define X(e, s) GGML_METAL_LIB_##e,
    GGML_METAL_LIBS
#undef X
    GGML_METAL_LIB_COUNT,
};

static const char * const k_lib_names[GGML_METAL_LIB_COUNT] = {
#define X(e, s) [GGML_METAL_LIB_##e] = #s,
    GGML_METAL_LIBS
#undef X
};

struct ggml_metal_library {
    // Per-kind compiled libraries. When single_library is true, the whole library
    // (e.g. a pre-compiled default.metallib or a from-source build) lives at
    // objs[0] and the remaining slots are nil.
    id<MTLLibrary> objs[GGML_METAL_LIB_COUNT];
    bool single_library; // true: combined library at objs[0]; false: per-kind libs in objs[*]

    // Routing table: kernel function name -> objs[] index, populated from each
    // compiled library's -[MTLLibrary functionNames]. The actual compiled
    // libraries are the single source of truth for which library owns a kernel,
    // so adding kernels later requires no manual routing maintenance.
    // nil in single_library mode (everything resolves to objs[0]).
    NSMutableDictionary<NSString *, NSNumber *> * fn_to_lib;

    // kernels from a second metallib, resolved ahead of the combined library
    NSSet<NSString *> * override_fns;

    ggml_metal_device_t dev;
    ggml_metal_pipelines_t pipelines; // cache of compiled pipelines

    NSLock * lock;
};

// Build the fn_to_lib routing table by querying each compiled library's public
// function names. Call once after all per-kind libraries have been compiled.
static void ggml_metal_library_build_index(ggml_metal_library_t lib) {
    @autoreleasepool {
        NSMutableDictionary<NSString *, NSNumber *> * index = [[NSMutableDictionary alloc] init];
        for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
            for (NSString * fname in [lib->objs[kind] functionNames]) {
                index[fname] = @(kind);
            }
        }
        lib->fn_to_lib = index;
    }
}

// note: defined below, after struct ggml_metal_device
static void ggml_metal_device_disable_tensor(ggml_metal_device_t dev);

// the tensor API headers are exposed to the shader compiler only at Metal language version 4.0
static void ggml_metal_compile_options_set_lang(MTLCompileOptions * options, bool has_tensor) {
    if (!has_tensor) {
        return;
    }

    options.languageVersion = (MTLLanguageVersion) MTLLanguageVersion4_0_GGML;
}

// Parse a `#include "name"` line. Returns the quoted name in *include_name on
// success. Whitespace-tolerant; ignores `#include <...>` (system headers).
static bool ggml_metal_library_parse_quoted_include(NSString * line, NSString ** include_name) {
    NSScanner * scanner = [NSScanner scannerWithString:line];
    scanner.charactersToBeSkipped = [NSCharacterSet whitespaceCharacterSet];

    if (![scanner scanString:@"#" intoString:NULL] ||
        ![scanner scanString:@"include" intoString:NULL] ||
        ![scanner scanString:@"\"" intoString:NULL]) {
        return false;
    }

    NSString * name = nil;
    if (![scanner scanUpToString:@"\"" intoString:&name]) {
        return false;
    }

    if (include_name) {
        *include_name = name;
    }
    return true;
}

// Recursively inline `#include "name"` directives. System includes (<...>),
// `#if/#else/#endif`, and other preprocessor lines are passed through to the
// Metal compiler unchanged. `#pragma once` is dropped since `seen` already
// guards against double-inclusion.
static bool ggml_metal_library_flatten_file(NSMutableString * dst, NSString * path,
                                            NSArray<NSString *> * search_paths,
                                            NSMutableSet<NSString *> * seen, NSError ** error) {
    NSString * key = [path stringByStandardizingPath];
    if ([seen containsObject:key]) {
        return true;
    }
    [seen addObject:key];

    NSString * src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:error];
    if (!src) {
        return false;
    }

    NSFileManager * fm = [NSFileManager defaultManager];
    for (NSString * line in [src componentsSeparatedByString:@"\n"]) {
        NSString * trimmed = [line stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
        if ([trimmed isEqualToString:@"#pragma once"]) {
            continue;
        }

        NSString * include_name = nil;
        if (ggml_metal_library_parse_quoted_include(line, &include_name)) {
            NSString * resolved = nil;
            for (NSString * dir in search_paths) {
                NSString * candidate = [dir stringByAppendingPathComponent:include_name];
                if ([fm isReadableFileAtPath:candidate]) {
                    resolved = candidate;
                    break;
                }
            }
            if (!resolved) {
                if (error) {
                    NSString * msg = [NSString stringWithFormat:@"could not resolve include \"%@\" from '%@'", include_name, path];
                    *error = [NSError errorWithDomain:@"ggml-metal-source-flatten" code:1
                                             userInfo:@{NSLocalizedDescriptionKey: msg}];
                }
                return false;
            }
            if (!ggml_metal_library_flatten_file(dst, resolved, search_paths, seen, error)) {
                return false;
            }
            continue;
        }

        [dst appendString:line];
        [dst appendString:@"\n"];
    }

    return true;
}

static NSString * ggml_metal_library_flatten_source(NSString * path_source, NSError ** error) {
    // Search paths cover both runtime layout (build/bin/kernels + build/bin)
    // and source-tree layout (ggml/src/ggml-metal/kernels + ggml/src/ggml-metal + ggml/src).
    NSString * path_kernels = [path_source stringByDeletingLastPathComponent];
    NSString * path_base    = [path_kernels stringByDeletingLastPathComponent];
    NSArray<NSString *> * search_paths = @[
        path_kernels,
        path_base,
        [path_base stringByDeletingLastPathComponent],
    ];

    NSMutableString * src = [[NSMutableString alloc] init];
    NSMutableSet<NSString *> * seen = [NSMutableSet set];

    if (!ggml_metal_library_flatten_file(src, path_source, search_paths, seen, error)) {
        [src release];
        return nil;
    }
    return src;
}

// Precompiled libraries next to the binary, instead of spending tens of seconds compiling the
// sources on every launch. They are built without the tensor API, so M5-class GPUs fall through.
// kernels/fingerprint lists "<kind> <sha256 of its source>"; a library built from other sources
// than the ones embedded in this binary is stale and is not loaded
static bool ggml_metal_library_precompiled_matches(NSString * dir, NSString * const * want) {
    NSString * text = [NSString stringWithContentsOfFile:[dir stringByAppendingPathComponent:@"fingerprint"]
                                                encoding:NSUTF8StringEncoding error:nil];
    if (text == nil) {
        GGML_LOG_WARN("%s: precompiled kernels carry no fingerprint, compiling the embedded sources\n", __func__);
        return false;
    }
    NSMutableDictionary<NSString *, NSString *> * have = [NSMutableDictionary dictionary];
    for (NSString * line in [text componentsSeparatedByString:@"\n"]) {
        NSArray<NSString *> * f = [line componentsSeparatedByString:@" "];
        if (f.count == 2) have[f[0]] = f[1];
    }
    for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
        NSString * name = [NSString stringWithUTF8String:k_lib_names[kind]];
        if (![have[name] isEqualToString:want[kind]]) {
            GGML_LOG_WARN("%s: precompiled '%s' was built from other sources, compiling the embedded ones\n",
                          __func__, k_lib_names[kind]);
            return false;
        }
    }
    return true;
}

static bool ggml_metal_library_load_precompiled(ggml_metal_library_t res, id<MTLDevice> device, NSString * const * want) {
    if (ggml_metal_device_get_props(res->dev)->has_tensor) {
        return false;
    }

    NSString * bin_cur = [[NSProcessInfo processInfo] arguments][0];
    NSString * dir = [[bin_cur stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"kernels"];

    if (![[NSFileManager defaultManager] fileExistsAtPath:dir]) {
        return false;
    }
    if (!ggml_metal_library_precompiled_matches(dir, want)) {
        return false;
    }

    const int64_t t_start = ggml_time_us();

    for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
        NSString * path = [dir stringByAppendingPathComponent:
            [NSString stringWithFormat:@"%s.metallib", k_lib_names[kind]]];

        NSError * error = nil;
        res->objs[kind] = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
        if (!res->objs[kind]) {
            GGML_LOG_WARN("%s: precompiled '%s' unusable, compiling from source\n", __func__, k_lib_names[kind]);
            for (int i = 0; i < GGML_METAL_LIB_COUNT; ++i) {
                if (res->objs[i]) {
                    [res->objs[i] release];
                    res->objs[i] = nil;
                }
            }
            return false;
        }
    }

    GGML_LOG_INFO("%s: loaded %d precompiled libraries in %.3f sec\n",
                  __func__, GGML_METAL_LIB_COUNT, (ggml_time_us() - t_start) / 1e6);

    ggml_metal_library_build_index(res);

    return true;
}

// Compile all per-kind libraries in parallel. `source_for_kind` returns the MSL
// source for a kind (the helper takes ownership and releases it), or nil with
// *err set on failure. On success the objs[] slots are populated and the routing
// index is built; on any failure every error is logged and false is returned
// (the caller is responsible for freeing `res`).
static bool ggml_metal_library_compile_all(
        ggml_metal_library_t res,
        id<MTLDevice> device,
        NSDictionary * prep,
        NSString * (^source_for_kind)(int kind, NSError ** err),
        const char * origin) {
    const int64_t t_start = ggml_time_us();

    int64_t  * t_per_lib   = calloc(GGML_METAL_LIB_COUNT, sizeof(int64_t));
    NSError ** err_per_lib = calloc(GGML_METAL_LIB_COUNT, sizeof(NSError *));
    __block atomic_bool any_failure = false;

    dispatch_group_t group = dispatch_group_create();
    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

    for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
        dispatch_group_async(group, queue, ^{

            const int64_t t0 = ggml_time_us();

            NSError * error = nil;

            NSString * src = source_for_kind(kind, &error);
            if (!src) {
                err_per_lib[kind] = [error retain];
                atomic_store(&any_failure, true);
                return;
            }

            id<MTLLibrary> lib = nil;

            @autoreleasepool {
                MTLCompileOptions * options = [MTLCompileOptions new];
                options.preprocessorMacros = prep;
                ggml_metal_compile_options_set_lang(options, ggml_metal_device_get_props(res->dev)->has_tensor);

                lib = [device newLibraryWithSource:src options:options error:&error];

                [options release];

                // retain the error before the autorelease pool drains it
                if (!lib) {
                    err_per_lib[kind] = [error retain];
                }
            }

            [src release];

            t_per_lib[kind] = ggml_time_us() - t0;

            if (!lib) {
                atomic_store(&any_failure, true);
                return;
            }

            res->objs[kind] = lib;
        });
    }
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    dispatch_release(group);

    const bool ok = !atomic_load(&any_failure);

    if (ok) {
        const int64_t t_total = ggml_time_us() - t_start;
        int64_t t_max = 0;
        for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
            GGML_LOG_DEBUG("%s: compiled '%s' library in %.3f sec\n",
                           __func__, k_lib_names[kind], t_per_lib[kind] / 1e6);
            if (t_per_lib[kind] > t_max) t_max = t_per_lib[kind];
        }
        GGML_LOG_INFO("%s: loaded %d libraries from %s in %.3f sec (max single = %.3f sec)\n",
                      __func__, GGML_METAL_LIB_COUNT, origin, t_total / 1e6, t_max / 1e6);

        ggml_metal_library_build_index(res);
    } else {
        for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
            if (err_per_lib[kind]) {
                GGML_LOG_ERROR("%s: failed to build '%s' library: %s\n", __func__,
                               k_lib_names[kind], [[err_per_lib[kind] description] UTF8String]);
                [err_per_lib[kind] release];
            }
        }
    }

    free(err_per_lib);
    free(t_per_lib);

    return ok;
}

// look for <name>.metallib as a bundle resource, then next to the running binary
static NSString * ggml_metal_find_metallib(NSBundle * bundle, NSString * name) {
    NSError * error = nil;

    NSString * path_lib = [bundle pathForResource:name ofType:@"metallib"];
    if (path_lib == nil) {
        // Try to find the resource in the directory where the current binary located.
        NSString * bin_cur = [[NSProcessInfo processInfo] arguments][0];
        NSString * bin_dir = [bin_cur stringByDeletingLastPathComponent];

        NSString * path_lib_default = [NSString pathWithComponents:@[bin_dir, [name stringByAppendingPathExtension:@"metallib"]]];
        if ([[NSFileManager defaultManager] isReadableFileAtPath:path_lib_default]) {
            GGML_LOG_INFO("%s: found '%s'\n", __func__, [path_lib_default UTF8String]);

            NSDictionary * atts = [[NSFileManager defaultManager] attributesOfItemAtPath:path_lib_default error:&error];
            if (atts && atts[NSFileType] == NSFileTypeSymbolicLink) {
                // Optionally, if this is a symlink, try to resolve it.
                path_lib_default = [[NSFileManager defaultManager] destinationOfSymbolicLinkAtPath:path_lib_default error:&error];
                if (path_lib_default && [path_lib_default length] > 0 && ![[path_lib_default substringToIndex:1] isEqualToString:@"/"]) {
                    // It is a relative path, adding the binary directory as directory prefix.
                    path_lib_default = [NSString pathWithComponents:@[bin_dir, path_lib_default]];
                }
                if (!path_lib_default || ![[NSFileManager defaultManager] isReadableFileAtPath:path_lib_default]) {
                    // Link to the resource could not be resolved.
                    path_lib_default = nil;
                } else {
                    GGML_LOG_INFO("%s: symlink resolved '%s'\n", __func__, [path_lib_default UTF8String]);
                }
            }
        } else {
            // The resource couldn't be found in the binary's directory.
            path_lib_default = nil;
        }

        path_lib = path_lib_default;
    }

    return path_lib;
}

ggml_metal_library_t ggml_metal_library_init(ggml_metal_device_t dev) {
    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    ggml_metal_library_t res = calloc(1, sizeof(struct ggml_metal_library));
    res->dev       = dev;
    res->pipelines = ggml_metal_pipelines_init();
    res->lock      = [NSLock new];

    // shared MTLCompileOptions preprocessor macros (matches the build-time defines)
    NSMutableDictionary * prep = [NSMutableDictionary dictionary];
    if (ggml_metal_device_get_props(dev)->has_bfloat) {
        [prep setObject:@"1" forKey:@"GGML_METAL_HAS_BF16"];
    }
    if (ggml_metal_device_get_props(dev)->has_tensor) {
        [prep setObject:@"1" forKey:@"GGML_METAL_HAS_TENSOR"];
    }
#if GGML_METAL_EMBED_LIBRARY
    [prep setObject:@"1" forKey:@"GGML_METAL_EMBED_LIBRARY"];
#endif

#if GGML_METAL_EMBED_LIBRARY
    GGML_LOG_INFO("%s: using embedded metal library\n", __func__);

    // start/end symbols emitted by CMake (see CMakeLists.txt), one pair per kind
#define X(e, s) extern const char ggml_metallib_##s##_start[]; extern const char ggml_metallib_##s##_end[];
    GGML_METAL_LIBS
#undef X

    static const char * const lib_start[GGML_METAL_LIB_COUNT] = {
#define X(e, s) [GGML_METAL_LIB_##e] = ggml_metallib_##s##_start,
    GGML_METAL_LIBS
#undef X
    };
    static const char * const lib_end[GGML_METAL_LIB_COUNT] = {
#define X(e, s) [GGML_METAL_LIB_##e] = ggml_metallib_##s##_end,
    GGML_METAL_LIBS
#undef X
    };

    // what these kernels were built from, to tell a stale precompiled library apart in a log
    NSString * want[GGML_METAL_LIB_COUNT];
    {
        CC_SHA256_CTX all;
        CC_SHA256_Init(&all);
        for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
            unsigned char d[CC_SHA256_DIGEST_LENGTH];
            CC_SHA256(lib_start[kind], (CC_LONG) (lib_end[kind] - lib_start[kind]), d);
            NSMutableString * hex = [NSMutableString stringWithCapacity:64];
            for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) [hex appendFormat:@"%02x", d[i]];
            want[kind] = hex;
            CC_SHA256_Update(&all, d, sizeof(d));
        }
        unsigned char d[CC_SHA256_DIGEST_LENGTH];
        CC_SHA256_Final(d, &all);
        GGML_LOG_INFO("%s: kernel fingerprint %02x%02x%02x%02x%02x%02x\n", __func__, d[0], d[1], d[2], d[3], d[4], d[5]);
    }

    if (ggml_metal_library_load_precompiled(res, device, want)) {
        return res;
    }

    const bool ok = ggml_metal_library_compile_all(res, device, prep,
        ^NSString * (int kind, NSError ** err) {
            (void) err;
            return [[NSString alloc] initWithBytes:lib_start[kind]
                                            length:(lib_end[kind] - lib_start[kind])
                                          encoding:NSUTF8StringEncoding];
        }, "embedded data");

    if (!ok) {
        ggml_metal_library_free(res);
        return NULL;
    }

    return res;
#else
#ifdef SWIFT_PACKAGE
    NSBundle * bundle = SWIFTPM_MODULE_BUNDLE;
#else
    NSBundle * bundle = [NSBundle bundleForClass:[GGMLMetalClass class]];
#endif

    const int64_t t_start = ggml_time_us();

    NSError * error = nil;
    NSString * path_lib = ggml_metal_find_metallib(bundle, @"default");

    if (path_lib != nil) {
        // pre-compiled library found: a single combined default.metallib
        NSURL * libURL = [NSURL fileURLWithPath:path_lib];
        GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_lib UTF8String]);

        res->objs[0]        = [device newLibraryWithURL:libURL error:&error];
        res->single_library = true;
        if (!res->objs[0]) {
            GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
            ggml_metal_library_free(res);
            return NULL;
        }

        // the tensor API kernels are built into a separate metallib
        if (ggml_metal_device_get_props(dev)->has_tensor) {
            NSString * path_mm = ggml_metal_find_metallib(bundle, @"ggml-tensor");

            id<MTLLibrary> lib_mm = nil;
            if (path_mm != nil) {
                lib_mm = [device newLibraryWithURL:[NSURL fileURLWithPath:path_mm] error:&error];
                if (!lib_mm && error) {
                    GGML_LOG_ERROR("%s: %s\n", __func__, [[error description] UTF8String]);
                }
            }

            if (lib_mm) {
                GGML_LOG_INFO("%s: loaded '%s'\n", __func__, [path_mm UTF8String]);

                res->objs[GGML_METAL_LIB_MUL_MM] = [lib_mm retain];
                res->override_fns                = [[NSSet setWithArray:[lib_mm functionNames]] retain];
            } else {
                GGML_LOG_INFO("%s: ggml-tensor.metallib not found - disabling the tensor API\n", __func__);

                ggml_metal_device_disable_tensor(dev);
            }
        }

        GGML_LOG_INFO("%s: loaded in %.3f sec\n", __func__, (ggml_time_us() - t_start) / 1e6);
        return res;
    }

    // no pre-compiled metallib: fall back to compiling each kernel source separately
    GGML_LOG_INFO("%s: default.metallib not found, loading kernel sources\n", __func__);

    NSString * path_resource = [[NSProcessInfo processInfo].environment objectForKey:@"GGML_METAL_PATH_RESOURCES"];
    if (path_resource) {
        GGML_LOG_INFO("%s: GGML_METAL_PATH_RESOURCES = %s\n", __func__, [path_resource UTF8String]);
    }

    // resolve each kind's source path up front (file lookup/logging stays on the calling thread)
    NSString ** path_per_kind = calloc(GGML_METAL_LIB_COUNT, sizeof(NSString *));
    for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
        NSString * rel = [NSString stringWithFormat:@"kernels/%s.metal", k_lib_names[kind]];

        NSString * path_source = nil;
        if (path_resource) {
            path_source = [path_resource stringByAppendingPathComponent:rel];
        } else {
            NSString * stem = [NSString stringWithFormat:@"kernels/%s", k_lib_names[kind]];
            path_source = [bundle pathForResource:stem ofType:@"metal"];
        }

        if (path_source == nil || ![[NSFileManager defaultManager] isReadableFileAtPath:path_source]) {
            GGML_LOG_WARN("%s: could not locate %s in bundle, falling back to cwd\n", __func__, [rel UTF8String]);
            path_source = rel;
        }

        GGML_LOG_DEBUG("%s: loading '%s'\n", __func__, [path_source UTF8String]);

        path_per_kind[kind] = [path_source retain];
    }

    const bool ok = ggml_metal_library_compile_all(res, device, prep,
        ^NSString * (int kind, NSError ** err) {
            return ggml_metal_library_flatten_source(path_per_kind[kind], err);
        }, "source");

    for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
        [path_per_kind[kind] release];
    }
    free(path_per_kind);

    if (!ok) {
        ggml_metal_library_free(res);
        return NULL;
    }

    return res;
#endif
}

ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev, const char * source, bool verbose) {
    if (source == NULL) {
        GGML_LOG_ERROR("%s: source is NULL\n", __func__);
        return NULL;
    }

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    id<MTLLibrary> library = nil;
    NSError * error = nil;

    const int64_t t_start = ggml_time_us();

    NSString * src = [[NSString alloc] initWithBytes:source
                                              length:strlen(source)
                                            encoding:NSUTF8StringEncoding];
    if (!src) {
        GGML_LOG_ERROR("%s: failed to create NSString from source\n", __func__);
        return NULL;
    }

    @autoreleasepool {
        NSMutableDictionary * prep = [NSMutableDictionary dictionary];

        MTLCompileOptions * options = [MTLCompileOptions new];
        options.preprocessorMacros = prep;
        ggml_metal_compile_options_set_lang(options, ggml_metal_device_get_props(dev)->has_tensor);

        library = [device newLibraryWithSource:src options:options error:&error];
        if (error) {
            if (verbose) {
                GGML_LOG_ERROR("%s: error compiling source: %s\n", __func__, [[error description] UTF8String]);
            } else {
                GGML_LOG_ERROR("%s: error compiling source\n", __func__);
            }
            library = nil;
        }

        [options release];
    }

    [src release];

    if (!library) {
        if (verbose) {
            GGML_LOG_ERROR("%s: failed to create Metal library from source\n", __func__);
        }

        return NULL;
    }

    if (verbose) {
        GGML_LOG_INFO("%s: compiled in %.3f sec\n", __func__, (ggml_time_us() - t_start) / 1e6);
    }

    ggml_metal_library_t res = calloc(1, sizeof(struct ggml_metal_library));
    if (!res) {
        GGML_LOG_ERROR("%s: calloc failed\n", __func__);
        return NULL;
    }

    res->objs[0]        = library;
    res->single_library = true;
    res->dev            = dev;
    res->pipelines      = ggml_metal_pipelines_init();
    res->lock           = [NSLock new];

    return res;
}

void ggml_metal_library_free(ggml_metal_library_t lib) {
    if (!lib) {
        return;
    }

    for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
        if (lib->objs[kind]) {
            [lib->objs[kind] release];
        }
    }

    if (lib->fn_to_lib) {
        [lib->fn_to_lib release];
    }

    if (lib->override_fns) {
        [lib->override_fns release];
    }

    ggml_metal_pipelines_free(lib->pipelines);

    [lib->lock release];

    free(lib);
}

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib) {
    return lib->dev;
}

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline(ggml_metal_library_t lib, const char * name) {
    [lib->lock lock];

    struct ggml_metal_pipeline_with_params res = {
        /*.pipeline =*/ nil,
        /*.nsg      =*/ 0,
        /*.nr0      =*/ 0,
        /*.nr1      =*/ 0,
        /*.smem     =*/ 0,
        /*.c4       =*/ false,
        /*.cnt      =*/ false,
        /*.sgw      =*/ 0,
        /*.ntg      =*/ 0,
    };

    res.pipeline = ggml_metal_pipelines_get(lib->pipelines, name);

    [lib->lock unlock];

    return res;
}

struct ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, const char * base, const char * name, ggml_metal_cv_t cv) {
    struct ggml_metal_pipeline_with_params res = {
        /*.pipeline =*/ nil,
        /*.nsg      =*/ 0,
        /*.nr0      =*/ 0,
        /*.nr1      =*/ 0,
        /*.smem     =*/ 0,
        /*.c4       =*/ false,
        /*.cnt      =*/ false,
        /*.sgw      =*/ 0,
        /*.ntg      =*/ 0,
    };

    [lib->lock lock];

    res.pipeline = ggml_metal_pipelines_get(lib->pipelines, name);
    if (res.pipeline) {
        [lib->lock unlock];

        return res;
    }

    @autoreleasepool {
        NSError * error = nil;

        NSString * base_func = [NSString stringWithUTF8String:base];

        GGML_LOG_DEBUG("%s: compiling pipeline: base = '%s', name = '%s'\n", __func__, base, name);

        // route to the library that actually defines this kernel; fn_to_lib is
        // built from -[MTLLibrary functionNames] so it's always in sync
        int lib_idx = 0;
        if (lib->override_fns && [lib->override_fns containsObject:base_func]) {
            lib_idx = GGML_METAL_LIB_MUL_MM;
        } else if (!lib->single_library) {
            NSNumber * idx = lib->fn_to_lib[base_func];
            if (!idx) {
                [lib->lock unlock];

                GGML_LOG_ERROR("%s: kernel not found in any metal library: base = '%s', name = '%s'\n", __func__, base, name);

                return res;
            }
            lib_idx = [idx intValue];
        }

        id<MTLLibrary> mtl_lib = lib->objs[lib_idx];

        id<MTLFunction> mtl_function;
        if (!cv) {
            mtl_function = [mtl_lib newFunctionWithName:base_func];
        } else {
            mtl_function = [mtl_lib newFunctionWithName:base_func constantValues:cv->obj error:&error];
        }
        if (!mtl_function) {
            [lib->lock unlock];

            GGML_LOG_ERROR("%s: failed to compile pipeline: base = '%s', name = '%s'\n", __func__, base, name);
            if (error) {
                GGML_LOG_ERROR("%s: %s\n", __func__, [[error description] UTF8String]);
            }

            return res;
        }

        id<MTLDevice> device = ggml_metal_device_get_obj(lib->dev);
        id<MTLComputePipelineState> obj = [device newComputePipelineStateWithFunction:mtl_function error:&error];

        [mtl_function release];

        if (!obj) {
            [lib->lock unlock];

            GGML_LOG_ERROR("%s: failed to create pipeline state: base = '%s', name = '%s'\n", __func__, base, name);
            if (error) {
                GGML_LOG_ERROR("%s: %s\n", __func__, [[error description] UTF8String]);
            }

            return res;
        }

        GGML_LOG_DEBUG("%s: loaded %-40s %16p | th_max = %4d | th_width = %4d\n", __func__, name,
                (void *) obj,
                (int)    obj.maxTotalThreadsPerThreadgroup,
                (int)    obj.threadExecutionWidth);

        if (obj.maxTotalThreadsPerThreadgroup == 0 || obj.threadExecutionWidth == 0) {
            [obj release];

            [lib->lock unlock];

            GGML_LOG_ERROR("%s: incompatible pipeline %s\n", __func__, name);

            return res;
        }

        res.pipeline = ggml_metal_pipeline_init();
        res.pipeline->obj = obj;

        ggml_metal_pipelines_add(lib->pipelines, name, res.pipeline);
    }

    [lib->lock unlock];

    return res;
}

//
// MTLComputeCommandEncoder wrapper
//

struct ggml_metal_encoder {
    id<MTLComputeCommandEncoder> obj;
    bool concurrent;
};

struct ggml_metal_pipeline_with_params ggml_metal_device_pipeline_add3(ggml_metal_device_t dev) {
    static ggml_metal_device_t     devs[8];
    static struct ggml_metal_pipeline pls[8];
    static dispatch_once_t         once[8];

    static os_unfair_lock          lock = OS_UNFAIR_LOCK_INIT;

    struct ggml_metal_pipeline_with_params res = { 0 };
    os_unfair_lock_lock(&lock);
    int i = 0;
    for (; i < 8 && devs[i] != NULL && devs[i] != dev; i++) {}
    if (i < 8) {
        devs[i] = dev;
    }
    os_unfair_lock_unlock(&lock);
    if (i == 8) {
        return res;
    }
    dispatch_once(&once[i], ^{
        static const char * src =
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "kernel void kernel_add3_f32_4(device const float4 * a [[buffer(0)]], device const float4 * b [[buffer(1)]],\n"
            "                              device const float4 * c [[buffer(2)]], device float4 * d [[buffer(3)]],\n"
            "                              constant uint & n [[buffer(4)]], uint i [[thread_position_in_grid]]) {\n"
            "    if (i < n) { d[i] = (a[i] + b[i]) + c[i]; }\n"
            "}\n";
        id<MTLDevice> mtl = (id<MTLDevice>) ggml_metal_device_get_obj(dev);
        NSError * err = nil;
        id<MTLLibrary> lib = [mtl newLibraryWithSource:[NSString stringWithUTF8String:src] options:nil error:&err];
        id<MTLFunction> fn = [lib newFunctionWithName:@"kernel_add3_f32_4"];
        pls[i].obj = fn != nil ? [mtl newComputePipelineStateWithFunction:fn error:&err] : nil;
        [fn release];
        [lib release];
    });
    if (pls[i].obj != nil) {
        res.pipeline = &pls[i];
    }
    return res;
}

struct ggml_metal_pipeline_with_params ggml_metal_device_pipeline_add3_rms_mul(ggml_metal_device_t dev) {
    static ggml_metal_device_t        devs[8];
    static struct ggml_metal_pipeline pls[8];
    static dispatch_once_t            once[8];
    static os_unfair_lock             lock = OS_UNFAIR_LOCK_INIT;

    struct ggml_metal_pipeline_with_params res = { 0 };
    os_unfair_lock_lock(&lock);
    int i = 0;
    for (; i < 8 && devs[i] != NULL && devs[i] != dev; i++) {}
    if (i < 8) {
        devs[i] = dev;
    }
    os_unfair_lock_unlock(&lock);
    if (i == 8) {
        return res;
    }
    dispatch_once(&once[i], ^{
        // the reduction mirrors kernel_rms_norm_fuse_impl so the norm matches it bit for bit
        static const char * src =
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "kernel void kernel_add3_rms_mul_f32_4(device const float4 * a [[buffer(0)]], device const float4 * b [[buffer(1)]],\n"
            "        device const float4 * c [[buffer(2)]], device float4 * x [[buffer(3)]], device const float4 * w [[buffer(4)]],\n"
            "        device float4 * y [[buffer(5)]], constant uint & n4 [[buffer(6)]], constant float & eps [[buffer(7)]],\n"
            "        constant uint & ne00 [[buffer(8)]], threadgroup float * shmem [[threadgroup(0)]],\n"
            "        uint row [[threadgroup_position_in_grid]], ushort tp [[thread_position_in_threadgroup]],\n"
            "        ushort sg [[simdgroup_index_in_threadgroup]], ushort ti [[thread_index_in_simdgroup]],\n"
            "        ushort ntg [[threads_per_threadgroup]]) {\n"
            "    if (sg == 0) { shmem[ti] = 0.0f; }\n"
            "    const uint r = row*n4;\n"
            "    float sumf = 0.0f;\n"
            "    for (uint i = tp; i < n4; i += ntg) {\n"
            "        const float4 v = (a[r + i] + b[r + i]) + c[r + i];\n"
            "        x[r + i] = v;\n"
            "        sumf += dot(v, v);\n"
            "    }\n"
            "    sumf = simd_sum(sumf);\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    if (ti == 0) { shmem[sg] = sumf; }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    sumf = shmem[ti];\n"
            "    sumf = simd_sum(sumf);\n"
            "    const float mean  = sumf/ne00;\n"
            "    const float scale = 1.0f/sqrt(mean + eps);\n"
            "    for (uint i = tp; i < n4; i += ntg) {\n"
            "        y[r + i] = (x[r + i]*scale)*w[i];\n"
            "    }\n"
            "}\n";
        id<MTLDevice> mtl = (id<MTLDevice>) ggml_metal_device_get_obj(dev);
        NSError * err = nil;
        id<MTLLibrary> lib = [mtl newLibraryWithSource:[NSString stringWithUTF8String:src] options:nil error:&err];
        id<MTLFunction> fn = [lib newFunctionWithName:@"kernel_add3_rms_mul_f32_4"];
        pls[i].obj = fn != nil ? [mtl newComputePipelineStateWithFunction:fn error:&err] : nil;
        [fn release];
        [lib release];
    });
    if (pls[i].obj != nil) {
        res.pipeline = &pls[i];
    }
    return res;
}

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent) {
    ggml_metal_encoder_t res = calloc(1, sizeof(struct ggml_metal_encoder));

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    res->concurrent = concurrent;

    if (concurrent) {
        res->obj = [cmd_buf computeCommandEncoderWithDispatchType: MTLDispatchTypeConcurrent];
    } else {
        res->obj = [cmd_buf computeCommandEncoder];
    }

    [res->obj retain];

    return res;
}

void ggml_metal_encoder_free(ggml_metal_encoder_t encoder) {
    [encoder->obj release];
    free(encoder);
}

void ggml_metal_encoder_debug_group_push(ggml_metal_encoder_t encoder, const char * name) {
    @autoreleasepool {
        [encoder->obj pushDebugGroup:[NSString stringWithCString:name encoding:NSUTF8StringEncoding]];
    }
}

void ggml_metal_encoder_debug_group_pop (ggml_metal_encoder_t encoder) {
    [encoder->obj popDebugGroup];
}

void ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, struct ggml_metal_pipeline_with_params pipeline) {
    [encoder->obj setComputePipelineState:pipeline.pipeline->obj];
}

void ggml_metal_encoder_set_bytes(ggml_metal_encoder_t encoder, void * data, size_t size, int idx) {
    [encoder->obj setBytes:data length:size atIndex:idx];
}

void ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, struct ggml_metal_buffer_id buffer, int idx) {
    [encoder->obj setBuffer:buffer.metal offset:buffer.offs atIndex:idx];
}

void ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, size_t size, int idx) {
    [encoder->obj setThreadgroupMemoryLength:size atIndex:idx];
}

void ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, int tg0, int tg1, int tg2, int tptg0, int tptg1, int tptg2) {
    [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2) threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];
}

void ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder) {
    [encoder->obj memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder) {
    [encoder->obj endEncoding];
}

void ggml_metal_encoder_event(ggml_metal_encoder_t encoder, ggml_metal_cmd_buf_t cmd_buf_raw, void * ev, uint64_t value, bool signal) {
    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    [encoder->obj endEncoding];
    [encoder->obj release];

    if (signal) {
        [cmd_buf encodeSignalEvent:(id<MTLSharedEvent>) ev value:value];
    } else {
        [cmd_buf encodeWaitForEvent:(id<MTLSharedEvent>) ev value:value];
    }

    if (encoder->concurrent) {
        encoder->obj = [cmd_buf computeCommandEncoderWithDispatchType: MTLDispatchTypeConcurrent];
    } else {
        encoder->obj = [cmd_buf computeCommandEncoder];
    }
    [encoder->obj retain];
}

#define GGML_METAL_HOST_WRAP_MAX 256

struct ggml_metal_device {
    id<MTLDevice> mtl_device;

    // a single global queue shared by all Metal backends
    // technically not needed for devices with unified memory, but enables discrete GPUs support
    // ref: https://github.com/ggml-org/llama.cpp/pull/15906
    id<MTLCommandQueue> mtl_queue;

    ggml_metal_rsets_t rsets;

    ggml_metal_library_t library;

    struct ggml_metal_device_props props;

    // shared fusion debugging context
    struct ggml_metal_fusion_info * finfo;

    // virtual address for GPU memory allocations
    atomic_uintptr_t addr_virt;

    // reused host-visible staging buffer for small private-buffer transfers;
    // wrapping the caller's pointer allocates a fresh kernel resource per call,
    // which on AMD accumulates across long runs until transfers crawl
    id<MTLBuffer> stage_buf;
    NSRecursiveLock * stage_lock;   // recursive: a read batch holds it across stage_get calls
    // upload ring: the blit shares the compute queue, so the only wait is slot reuse
    id<MTLBuffer>        stage_set_bufs[4];
    id<MTLCommandBuffer> stage_set_cmds[4];
    int                  stage_set_cur;
    // open read batch: blits accumulate into one command buffer, one wait at end
    bool                       stage_batch_open;
    id<MTLCommandBuffer>       stage_batch_cmd;   // created by the first read, released on flush
    id<MTLBlitCommandEncoder>  stage_batch_enc;
    size_t                     stage_batch_used;
    int                        stage_batch_n;
    struct { void * dst; size_t off; size_t size; } stage_batch_items[16];

    // cached no-copy wraps of host memory regions used as blit sources, so
    // repeated uploads of the same weights don't churn kernel resources
    struct {
        const void *  base;
        size_t        size;
        id<MTLBuffer> buf;
    } host_wraps[GGML_METAL_HOST_WRAP_MAX];
    int n_host_wraps;
    size_t host_wrap_bytes;
    NSLock * wrap_lock;

    // mixed expert execution: one event and one host-visible stage, created on first use
    id<MTLSharedEvent> dmoe_ev;
    id<MTLSharedEvent> dmoe_reach;
    id<MTLBuffer>      dmoe_stage;
    NSLock *           dmoe_lock;
    id<MTLBuffer>      dmoe_tables;
    id<MTLBuffer>      dmoe_obs;
    id<MTLBuffer>      dmoe_wide;
};

#ifdef TOSH_ENABLE_DYNAMIC_MOE
static uint64_t ggml_metal_dmoe_ev_get(void * ev) { return [(id<MTLSharedEvent>) ev signaledValue]; }
static void ggml_metal_dmoe_ev_set(void * ev, uint64_t v) { [(id<MTLSharedEvent>) ev setSignaledValue:v]; }

void * ggml_metal_device_dmoe_event(ggml_metal_device_t dev) {
    [dev->dmoe_lock lock];
    if (dev->dmoe_ev == nil) {
        dev->dmoe_ev = [dev->mtl_device newSharedEvent];
        tosh_dmoe_set_event(dev->dmoe_ev, ggml_metal_dmoe_ev_get, ggml_metal_dmoe_ev_set);
    }
    [dev->dmoe_lock unlock];
    return dev->dmoe_ev;
}

// signalled when a wide join is reached, before it waits on the host
void * ggml_metal_device_dmoe_reach_event(ggml_metal_device_t dev) {
    [dev->dmoe_lock lock];
    if (dev->dmoe_reach == nil) {
        dev->dmoe_reach = [dev->mtl_device newSharedEvent];
        tosh_dmoe_set_reach_event(dev->dmoe_reach, ggml_metal_dmoe_ev_get);
    }
    [dev->dmoe_lock unlock];
    return dev->dmoe_reach;
}

struct ggml_metal_buffer_id ggml_metal_device_dmoe_obs_ring(ggml_metal_device_t dev) {
    [dev->dmoe_lock lock];
    if (dev->dmoe_obs == nil) {
        dev->dmoe_obs = [dev->mtl_device newBufferWithLength:TOSH_DMOE_OBS_RING_BYTES options:MTLResourceStorageModeShared];
        if (dev->dmoe_obs != nil) {
            tosh_dmoe_set_obs_ring([dev->dmoe_obs contents]);
        }
    }
    [dev->dmoe_lock unlock];
    struct ggml_metal_buffer_id res = { dev->dmoe_obs, 0 };
    return res;
}

struct ggml_metal_buffer_id ggml_metal_device_dmoe_tables(ggml_metal_device_t dev) {
    struct ggml_metal_buffer_id res = { nil, 0 };
    size_t bytes = 0;
    int32_t * host = tosh_dmoe_tables(&bytes, NULL, NULL);
    if (host == NULL) {
        return res;
    }
    [dev->dmoe_lock lock];
    if (dev->dmoe_tables == nil) {
        dev->dmoe_tables = [dev->mtl_device newBufferWithBytesNoCopy:host length:bytes options:MTLResourceStorageModeShared deallocator:nil];
        GGML_ASSERT(dev->dmoe_tables != nil && "residency tables must be page aligned");
    }
    [dev->dmoe_lock unlock];
    res.metal = dev->dmoe_tables;
    return res;
}

// promotions: a queue of their own and a bounded host-visible stage the canonical bytes pass through
#define DMOE_UP_SLOTS   8
#define DMOE_UP_BYTES   (8u << 20)
#define DMOE_UP_TICKETS 1024

struct ggml_metal_dmoe_up {
    ggml_metal_device_t  dev;
    id<MTLCommandQueue>  queue;
    id<MTLBuffer>        stage[DMOE_UP_SLOTS];
    id<MTLCommandBuffer> stage_cb[DMOE_UP_SLOTS];
    id<MTLCommandBuffer> tickets[DMOE_UP_TICKETS];
    id<MTLBuffer>        down;      // host-visible stage of the copies back to RAM
    uint64_t             n_tickets;
    int                  next;
    long                 fail_at;   // test only
    NSLock *             lock;
};

static uint64_t ggml_metal_dmoe_upload_impl(void * ctx_raw, const struct tosh_dmoe_copy * copies, int n);

static uint64_t ggml_metal_dmoe_upload(void * ctx_raw, const struct tosh_dmoe_copy * copies, int n) {
    @autoreleasepool {
        return ggml_metal_dmoe_upload_impl(ctx_raw, copies, n);
    }
}

static uint64_t ggml_metal_dmoe_upload_impl(void * ctx_raw, const struct tosh_dmoe_copy * copies, int n) {
    struct ggml_metal_dmoe_up * up = ctx_raw;
    [up->lock lock];
    const uint64_t ticket = ++up->n_tickets;
    if ((long) ticket == up->fail_at) {
        [up->lock unlock];
        return 0;
    }
    const int s = up->next;
    up->next = (up->next + 1) % DMOE_UP_SLOTS;
    if (up->stage_cb[s] != nil) {
        [up->stage_cb[s] waitUntilCompleted];
        [up->stage_cb[s] release];
        up->stage_cb[s] = nil;
    }
    size_t total = 0;
    for (int i = 0; i < n; ++i) {
        total += copies[i].size;
    }
    if (total > [up->stage[s] length]) {
        // a stage slot grows once to the largest expert, so staging stays bounded by 8 experts
        id<MTLBuffer> grown = [up->dev->mtl_device newBufferWithLength:(total + (1u << 20) - 1)/(1u << 20)*(1u << 20)
                                                               options:MTLResourceStorageModeShared];
        if (grown == nil) {
            [up->lock unlock];
            return 0;
        }
        [up->stage[s] release];
        up->stage[s] = grown;
    }
    char * stage = [up->stage[s] contents];
    size_t off = 0;
    id<MTLCommandBuffer> cb = [up->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    for (int i = 0; i < n; ++i) {
        const struct ggml_tensor * t = copies[i].dst;
        ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;
        struct ggml_metal_buffer_id dst = ggml_metal_buffer_get_id(buffer->context, t);
        memcpy(stage + off, copies[i].src, copies[i].size);
        [blit copyFromBuffer:up->stage[s] sourceOffset:off toBuffer:dst.metal destinationOffset:dst.offs + copies[i].offset size:copies[i].size];
        off += copies[i].size;
    }
    [blit endEncoding];
    [cb commit];
    [cb retain];
    [cb retain];
    up->stage_cb[s] = cb;
    id<MTLCommandBuffer> old = up->tickets[ticket % DMOE_UP_TICKETS];
    up->tickets[ticket % DMOE_UP_TICKETS] = cb;
    [up->lock unlock];
    if (old != nil) {
        [old release];
    }
    return ticket;
}

static int ggml_metal_dmoe_upload_status(void * ctx_raw, uint64_t ticket) {
    struct ggml_metal_dmoe_up * up = ctx_raw;
    [up->lock lock];
    id<MTLCommandBuffer> cb = up->tickets[ticket % DMOE_UP_TICKETS];
    const MTLCommandBufferStatus st = cb != nil ? [cb status] : MTLCommandBufferStatusError;
    [up->lock unlock];
    if (st == MTLCommandBufferStatusCompleted) return 1;
    if (st == MTLCommandBufferStatusError)     return -1;
    return 0;
}

static double ggml_metal_dmoe_upload_gpu_us(void * ctx_raw, uint64_t ticket) {
    struct ggml_metal_dmoe_up * up = ctx_raw;
    [up->lock lock];
    id<MTLCommandBuffer> cb = up->tickets[ticket % DMOE_UP_TICKETS];
    const double us = cb != nil ? ([cb GPUEndTime] - [cb GPUStartTime])*1e6 : 0.0;
    [up->lock unlock];
    return us;
}

#define DMOE_DOWN_BYTES (32u << 20)

static bool ggml_metal_dmoe_download(void * ctx_raw, const struct tosh_dmoe_copy * copies, int n) {
    struct ggml_metal_dmoe_up * up = ctx_raw;
    @autoreleasepool {
        [up->lock lock];
        if (up->down == nil) {
            up->down = [up->dev->mtl_device newBufferWithLength:DMOE_DOWN_BYTES options:MTLResourceStorageModeShared];
        }
        bool ok = up->down != nil;
        for (int i = 0; ok && i < n;) {
            id<MTLCommandBuffer> cb = [up->queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            size_t off = 0;
            int j = i;
            for (; j < n && off + copies[j].size <= DMOE_DOWN_BYTES; ++j) {
                const struct ggml_tensor * t = copies[j].src;
                ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;
                struct ggml_metal_buffer_id src = ggml_metal_buffer_get_id(buffer->context, t);
                [blit copyFromBuffer:src.metal sourceOffset:src.offs + copies[j].offset toBuffer:up->down destinationOffset:off size:copies[j].size];
                off += copies[j].size;
            }
            [blit endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            ok = j > i && [cb status] == MTLCommandBufferStatusCompleted;
            const char * stage = [up->down contents];
            off = 0;
            for (int k = i; ok && k < j; ++k) {
                memcpy((void *) copies[k].dst, stage + off, copies[k].size);
                off += copies[k].size;
            }
            i = j;
        }
        [up->lock unlock];
        return ok;
    }
}

static void ggml_metal_dmoe_uploader_init(ggml_metal_device_t dev);
static ggml_metal_device_t g_uploader_dev = NULL;
static void ggml_metal_dmoe_uploader_make(void) {
    if (g_uploader_dev != NULL) {
        ggml_metal_dmoe_uploader_init(g_uploader_dev);
    }
}

static void ggml_metal_dmoe_uploader_init(ggml_metal_device_t dev) {
    struct ggml_metal_dmoe_up * up = calloc(1, sizeof(struct ggml_metal_dmoe_up));
    up->dev   = dev;
    up->queue = [dev->mtl_device newCommandQueue];
    for (int i = 0; i < DMOE_UP_SLOTS; ++i) {
        up->stage[i] = [dev->mtl_device newBufferWithLength:DMOE_UP_BYTES options:MTLResourceStorageModeShared];
    }
    up->lock = [[NSLock alloc] init];
    const char * f = getenv("TOSH_DMOE_TEST_UPLOAD_FAIL");
    up->fail_at = f ? atol(f) : -1;
    struct tosh_dmoe_uploader u = { up, ggml_metal_dmoe_upload, ggml_metal_dmoe_upload_status, ggml_metal_dmoe_upload_gpu_us,
                                    ggml_metal_dmoe_download };
    tosh_dmoe_set_uploader(&u);
}

struct ggml_metal_buffer_id ggml_metal_device_dmoe_wide_stage(ggml_metal_device_t dev, size_t bytes) {
    [dev->dmoe_lock lock];
    if (dev->dmoe_wide == nil) {
        dev->dmoe_wide = [dev->mtl_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        GGML_ASSERT(dev->dmoe_wide != nil);
        tosh_dmoe_set_wide_stage([dev->dmoe_wide contents], bytes);
    }
    GGML_ASSERT([dev->dmoe_wide length] >= bytes && "the wide stage size cannot change once allocated");
    [dev->dmoe_lock unlock];
    struct ggml_metal_buffer_id res = { dev->dmoe_wide, 0 };
    return res;
}

struct ggml_metal_buffer_id ggml_metal_device_dmoe_stage(ggml_metal_device_t dev, size_t layer_stride) {
    [dev->dmoe_lock lock];
    const size_t size = layer_stride*TOSH_DMOE_MAX_LAYERS;
    if (dev->dmoe_stage == nil || [dev->dmoe_stage length] < size) {
        GGML_ASSERT(dev->dmoe_stage == nil && "the stage stride cannot change once allocated");
        dev->dmoe_stage = [dev->mtl_device newBufferWithLength:size options:MTLResourceStorageModeShared];
        GGML_ASSERT(dev->dmoe_stage != nil);
        memset([dev->dmoe_stage contents], 0, size);
        tosh_dmoe_set_stage([dev->dmoe_stage contents], layer_stride);
    }
    [dev->dmoe_lock unlock];
    struct ggml_metal_buffer_id res = { dev->dmoe_stage, 0 };
    return res;
}
#endif

//
// MTLResidenceSet wrapper
//

struct ggml_metal_rsets {
    NSLock * lock;

    NSMutableArray * data;

    // number of seconds since the last graph computation
    // keep the residency sets wired for that amount of time to avoid being collected by the OS
    int keep_alive_s;
    int loops_per_s;
    int time_per_loop_ms;

    // background heartbeat thread to keep the residency sets alive
    atomic_bool d_stop;
    atomic_int  d_loop;

    dispatch_group_t d_group;
};

#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
static void ggml_metal_dummy_work(ggml_metal_device_t dev) {
    if (dev->mtl_queue == nil) {
        return;
    }

    @autoreleasepool {
        // perform a minimal dummy operation on the GPU
        id<MTLBuffer> buf = [dev->mtl_device newBufferWithLength:1 options:MTLResourceStorageModePrivate];
        id<MTLCommandBuffer> cmd_buf = [dev->mtl_queue commandBuffer];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:buf range:NSMakeRange(0, 1) value:0];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [buf release];
    }
}
#endif

ggml_metal_rsets_t ggml_metal_rsets_init(ggml_metal_device_t dev) {
    ggml_metal_rsets_t res = calloc(1, sizeof(struct ggml_metal_rsets));

    res->lock = [[NSLock alloc] init];
    res->data = [[NSMutableArray alloc] init];

    // by default keep the memory wired for 3 minutes
    res->keep_alive_s = 3*60;

    const char * GGML_METAL_RESIDENCY_KEEP_ALIVE_S = TOSH_ENV("GGML_METAL_RESIDENCY_KEEP_ALIVE_S");
    if (GGML_METAL_RESIDENCY_KEEP_ALIVE_S) {
        res->keep_alive_s = atoi(GGML_METAL_RESIDENCY_KEEP_ALIVE_S);
    }

    if (res->keep_alive_s <= 0) {
        res->keep_alive_s = 3*60;
    }

    res->time_per_loop_ms = 5;
    res->loops_per_s = 1000/res->time_per_loop_ms;

    GGML_LOG_INFO("%s: creating a residency set collection (keep_alive = %d s)\n", __func__, res->keep_alive_s);

    atomic_store_explicit(&res->d_stop, false, memory_order_relaxed);
    atomic_store_explicit(&res->d_loop, res->loops_per_s*res->keep_alive_s, memory_order_relaxed);

    res->d_group = dispatch_group_create();

    // start a background thread that periodically requests residency for all the currently active sets in the collection
    // the requests stop after a certain amount of time (keep_alive_s) of inactivity
    dispatch_queue_t d_queue = dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);
    dispatch_group_async(res->d_group, d_queue, ^{
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
        if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
              while (!atomic_load_explicit(&res->d_stop, memory_order_relaxed)) {
                  if (atomic_load_explicit(&res->d_loop, memory_order_relaxed) > 0) {
                      [res->lock lock];

                      for (int i = 0; i < (int) res->data.count; ++i) {
                          [res->data[i] requestResidency];
                      }

                      atomic_fetch_sub_explicit(&res->d_loop, 1, memory_order_relaxed);

                      [res->lock unlock];
                  }

                  usleep(res->time_per_loop_ms * 1000);
              }
        }
#endif
    });

#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        // workaround for residency set memory not being released if no GPU operation occurs
        // https://developer.apple.com/forums/thread/839089
        // https://github.com/ggml-org/llama.cpp/issues/25937
        ggml_metal_dummy_work(dev);
    }
#endif

    return res;
}

void ggml_metal_rsets_free(ggml_metal_rsets_t rsets) {
    if (rsets == NULL) {
        return;
    }

    // note: if you hit this assert, most likely you haven't deallocated all Metal resources before exiting
    GGML_ASSERT([rsets->data count] == 0);

    atomic_store_explicit(&rsets->d_stop, true, memory_order_relaxed);

    dispatch_group_wait(rsets->d_group, DISPATCH_TIME_FOREVER);
    dispatch_release(rsets->d_group);

    [rsets->data release];
    [rsets->lock release];

    free(rsets);
}

static const struct {
    const char *              name;
    const char *              token;
    enum ggml_metal_device_id id;
} k_metal_devices[] = {
#define DEV(name, id) { name, #id, id }
    DEV("M1",       GGML_METAL_DEVICE_M1),
    DEV("M1 Pro",   GGML_METAL_DEVICE_M1_PRO),
    DEV("M1 Max",   GGML_METAL_DEVICE_M1_MAX),
    DEV("M1 Ultra", GGML_METAL_DEVICE_M1_ULTRA),
    DEV("M2",       GGML_METAL_DEVICE_M2),
    DEV("M2 Pro",   GGML_METAL_DEVICE_M2_PRO),
    DEV("M2 Max",   GGML_METAL_DEVICE_M2_MAX),
    DEV("M2 Ultra", GGML_METAL_DEVICE_M2_ULTRA),
    DEV("M3",       GGML_METAL_DEVICE_M3),
    DEV("M3 Pro",   GGML_METAL_DEVICE_M3_PRO),
    DEV("M3 Max",   GGML_METAL_DEVICE_M3_MAX),
    DEV("M3 Ultra", GGML_METAL_DEVICE_M3_ULTRA),
    DEV("M4",       GGML_METAL_DEVICE_M4),
    DEV("M4 Pro",   GGML_METAL_DEVICE_M4_PRO),
    DEV("M4 Max",   GGML_METAL_DEVICE_M4_MAX),
    DEV("M5",       GGML_METAL_DEVICE_M5),
    DEV("M5 Pro",   GGML_METAL_DEVICE_M5_PRO),
    DEV("M5 Max",   GGML_METAL_DEVICE_M5_MAX),
    DEV("M5 Ultra", GGML_METAL_DEVICE_M5_ULTRA),
    DEV("A18 Pro",  GGML_METAL_DEVICE_A18_PRO),
#undef DEV
};

// tensor split blocks the main thread on the queue's in-flight command buffer limit (64 by
// default), so allow raising it: TOSH_MTL_QUEUE_DEPTH
// macOS can hand out an integrated iGPU (isLowPower, ~1 GB) as the system
// default. Auto-selection skips it; explicit DEVICE_INDEX/LIST can still pin it.
static id<MTLDevice> ggml_metal_device_default_discrete(void) {
    id<MTLDevice> def = MTLCreateSystemDefaultDevice();
    if (def == nil || !def.isLowPower) {
        return def;
    }
    NSArray<id<MTLDevice>> * all = MTLCopyAllDevices();
    id<MTLDevice> best = nil;
    for (id<MTLDevice> d in all) {
        if (d.isLowPower) continue;
        if (best == nil || d.recommendedMaxWorkingSetSize > best.recommendedMaxWorkingSetSize) {
            best = d;
        }
    }
    if (best != nil) {
        fprintf(stderr, "ggml_metal: system default GPU is integrated (%s); using %s instead (pin GGML_METAL_DEVICE_INDEX to override)\n",
                def.name.UTF8String, best.name.UTF8String);
        [best retain];
        [def release];
        [all release];
        return best;
    }
    [all release];
    return def;
}


static id<MTLCommandQueue> ggml_metal_new_queue(id<MTLDevice> device) {
    const char * s = TOSH_ENV("TOSH_MTL_QUEUE_DEPTH");
    const int depth = s ? atoi(s) : 0;

    if (depth > 0) {
        return [device newCommandQueueWithMaxCommandBufferCount:depth];
    }

    return [device newCommandQueue];
}


static enum ggml_metal_device_id ggml_metal_device_id_parse(const char * name) {
    if (!name) {
        return GGML_METAL_DEVICE_GENERIC;
    }

    static const char prefix[] = "Apple ";
    if (strncmp(name, prefix, sizeof(prefix) - 1) != 0) {
        return GGML_METAL_DEVICE_GENERIC;
    }
    const char * suffix = name + sizeof(prefix) - 1;

    for (size_t i = 0; i < sizeof(k_metal_devices)/sizeof(k_metal_devices[0]); ++i) {
        if (strcmp(suffix, k_metal_devices[i].name) == 0) {
            return k_metal_devices[i].id;
        }
    }
    return GGML_METAL_DEVICE_GENERIC;
}

const char * ggml_metal_device_id_token(enum ggml_metal_device_id id) {
    for (size_t i = 0; i < sizeof(k_metal_devices)/sizeof(k_metal_devices[0]); ++i) {
        if (k_metal_devices[i].id == id) {
            return k_metal_devices[i].token;
        }
    }
    return "GGML_METAL_DEVICE_GENERIC";
}

ggml_metal_device_t ggml_metal_device_init(int device, int n_devices) {
    ggml_metal_device_t dev = calloc(1, sizeof(struct ggml_metal_device));

    assert(dev != NULL);

    @autoreleasepool {
        if (dev->mtl_device == nil) {
            // `device` is the logical slot and indexes buffer types downstream, so it must stay as
            // given; the env vars below only choose which MTLCopyAllDevices() GPU each slot maps to.
            //   GGML_METAL_DEVICE_INDEX=N    pin the single device to physical GPU N
            //   GGML_METAL_DEVICES=K         register K devices, slot i -> physical i
            //   GGML_METAL_DEVICE_LIST=a,b   register the listed GPUs, slot i -> physical list[i]
            const char * env_idx  = TOSH_ENV("GGML_METAL_DEVICE_INDEX");
            const char * env_num  = TOSH_ENV("GGML_METAL_DEVICES");
            const char * env_list = TOSH_ENV("GGML_METAL_DEVICE_LIST");
            const int    num_dev = env_num ? atoi(env_num) : 1;
            if (env_idx != NULL || num_dev > 1 || env_list != NULL) {
                const int base = env_idx ? atoi(env_idx) : 0;
                int phys = base + device;
                if (env_list != NULL) {
                    phys = -1;
                    const char * p = env_list;
                    for (int i = 0; ; ++i) {
                        if (i == device) { phys = atoi(p); break; }
                        while (*p != '\0' && *p != ',') ++p;
                        if (*p == '\0') break;
                        ++p;
                    }
                }
                NSArray<id<MTLDevice>> * all = MTLCopyAllDevices();
                if (env_list == NULL && env_idx == NULL && num_dev > 1) {
                    // DEVICES=K without index/list: slot i -> i-th discrete GPU;
                    // raw index when there aren't enough discrete devices.
                    int found = -1;
                    int discrete = -1;
                    for (int i = 0; i < (int) all.count; ++i) {
                        if (all[i].isLowPower) continue;
                        if (++found == device) { discrete = i; break; }
                    }
                    if (discrete >= 0) {
                        phys = discrete;
                    }
                }
                if (phys >= 0 && phys < (int) all.count) {
                    dev->mtl_device = [all[phys] retain];
                } else {
                    dev->mtl_device = ggml_metal_device_default_discrete();
                }
                [all release];
            } else {
                dev->mtl_device = ggml_metal_device_default_discrete();
            }

            if (dev->mtl_device) {
                dev->mtl_queue = ggml_metal_new_queue(dev->mtl_device);
                if (dev->mtl_queue == nil) {
                    GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
                }

                dev->stage_lock = [[NSRecursiveLock alloc] init];
                dev->wrap_lock  = [[NSLock alloc] init];
                dev->dmoe_lock  = [[NSLock alloc] init];
#ifdef TOSH_ENABLE_DYNAMIC_MOE
                if (g_uploader_dev == NULL) {
                    // created on first use: mixed execution can be chosen after the device starts
                    g_uploader_dev = dev;
                    tosh_dmoe_set_uploader_factory(ggml_metal_dmoe_uploader_make);
                }
#endif

                dev->addr_virt = 0x000000400ULL;

                dev->props.device = device;

                // the Metal backend uses the system default device as the single physical device;
                // additional (virtual) devices are emulated on top of it via GGML_METAL_DEVICES
                dev->props.device_phys = 0;
                dev->props.device_virt = device;

                // Probe the hardware SIMD-group width (threadExecutionWidth). All Metal
                // kernels here assume a 32-wide simdgroup, which holds on Apple Silicon and
                // AMD RDNA. AMD GCN/Vega report 64, where that assumption breaks (garbled
                // output). We compile a trivial pipeline and read its threadExecutionWidth.
                dev->props.simd_width = 32; // safe default
                {
                    NSError * probe_err = nil;
                    id<MTLLibrary> probe_lib = [dev->mtl_device newLibraryWithSource:
                        @"#include <metal_stdlib>\nusing namespace metal;\nkernel void ggml_probe_simd(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = x[i]; }"
                        options:nil error:&probe_err];
                    if (probe_lib) {
                        id<MTLFunction> probe_fn = [probe_lib newFunctionWithName:@"ggml_probe_simd"];
                        if (probe_fn) {
                            id<MTLComputePipelineState> probe_ps =
                                [dev->mtl_device newComputePipelineStateWithFunction:probe_fn error:&probe_err];
                            if (probe_ps && probe_ps.threadExecutionWidth > 0) {
                                dev->props.simd_width = (int) probe_ps.threadExecutionWidth;
                            }
                            [probe_ps release];
                            [probe_fn release];
                        }
                        [probe_lib release];
                    }
                    GGML_LOG_INFO("%s: probed SIMD-group width = %d\n", __func__, dev->props.simd_width);
                    // Also print unconditionally so the value is visible in captured logs even when
                    // the ggml log callback filters init-time INFO messages. macOS renumbers the
                    // devices on every reboot, so the slot index alone does not identify the card.
                    fprintf(stderr, "ggml_metal: device %d: %s (peer group %llu, %s) probed SIMD-group width = %d (32 = Apple/AMD RDNA, 64 = AMD GCN/Vega)\n",
                            device, dev->mtl_device.name.UTF8String,
                            (unsigned long long) dev->mtl_device.peerGroupID,
                            dev->mtl_device.peerGroupID == 0 ? "not bridged" : "bridged",
                            dev->props.simd_width);
                    // Expose the measured width to ToshLLM's graph-level AMD FA policy.
                    // A wave64 process records 64 and therefore never takes the RDNA-only
                    // head-64 prefill path below.
                    char simd_width_env[16];
                    snprintf(simd_width_env, sizeof(simd_width_env), "%d", dev->props.simd_width);
                    setenv("TOSH_METAL_SIMD_WIDTH", simd_width_env, 1);
                }

                dev->props.has_simdgroup_reduction  = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];
                dev->props.has_simdgroup_reduction |= [dev->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];
                // macOS 12 has no Metal3 family, but AMD cards already reduce across a simdgroup on Mac2
                if (strstr(dev->mtl_device.name.UTF8String ?: "", "AMD") != NULL) {
                    dev->props.has_simdgroup_reduction |= [dev->mtl_device supportsFamily:MTLGPUFamilyMac2];
                }
                // simdgroup reduction kernels produce corrupted output on AMD RDNA 2 (e.g. RX 6700 XT)
                if (TOSH_ENV("GGML_METAL_SIMDGROUP_REDUCTION_DISABLE") != NULL) {
                    dev->props.has_simdgroup_reduction = false;
                }

                // wave64 GPUs (AMD GCN/Vega): the 32-wide simdgroup reduction and mat-vec
                // kernels corrupt output, so route them to the CPU; only the wave-agnostic
                // manual mul_mm below runs on the GPU. Gated off for wave32 (Apple/RDNA).
                // GGML_METAL_WAVE64_UNSAFE=1 keeps the GPU paths for debugging.
                const bool is_wave64 =
                    dev->props.simd_width != 32 &&
                    ![dev->mtl_device supportsFamily:MTLGPUFamilyApple1] &&
                    TOSH_ENV("GGML_METAL_WAVE64_UNSAFE") == NULL;
                if (is_wave64) {
                    fprintf(stderr, "ggml_metal: wave64 mode (SIMD width %d): GPU prefill matmul, "
                                    "CPU decode/reductions for correct output\n",
                            dev->props.simd_width);
                    dev->props.has_simdgroup_reduction = false;
                }

                // A dword read off a multiple of four is undefined in Metal, and the GCN cards
                // before Vega return the bytes of the aligned address instead. Assume a wave64
                // card needs the aligned path unless it is one of the families measured to
                // tolerate it, so an unknown card errs on the side of correct output.
                {
                    // amdgpu_gfx<N> names the ISA target: gfx6xx-8xx are GCN 1 to 4, gfx9xx is
                    // Vega. Older than Vega reads a dword off a word boundary as the aligned
                    // one. The name is the fallback for macOS 12 and 13, which have no
                    // architecture property, and it grants the fast path rather than the safe
                    // one so an unknown card stays correct.
                    bool tolerant = false;
                    bool known    = false;
                    char how[64]  = "unknown";

                    if (@available(macOS 14.0, *)) {
                        const char * arch = dev->mtl_device.architecture.name.UTF8String;
                        int gfx = 0;
                        if (arch != NULL && sscanf(arch, "amdgpu_gfx%d", &gfx) == 1 && gfx > 0) {
                            tolerant = gfx >= 900;
                            known    = true;
                            snprintf(how, sizeof(how), "%s", arch);
                        }
                    }
                    if (!known) {
                        const char * name = dev->mtl_device.name.UTF8String;
                        tolerant = name != NULL &&
                            (strstr(name, "Vega")          != NULL ||
                             strstr(name, "Radeon VII")    != NULL ||
                             strstr(name, "Radeon Pro VII")!= NULL ||
                             strstr(name, "WX 8200")       != NULL ||
                             strstr(name, "WX 9100")       != NULL ||
                             strstr(name, "Instinct MI25") != NULL ||
                             strstr(name, "Instinct MI50") != NULL ||
                             strstr(name, "Instinct MI60") != NULL);
                        snprintf(how, sizeof(how), "name (no architecture before macOS 14)");
                    }
                    dev->props.needs_aligned_loads = is_wave64 && !tolerant;

                    // the macOS 12 driver misreads unaligned mat-vec loads on RDNA too
                    const bool monterey = [[NSProcessInfo processInfo] operatingSystemVersion].majorVersion < 13 &&
                                          ![dev->mtl_device supportsFamily:MTLGPUFamilyApple1];
                    if (monterey && !dev->props.needs_aligned_loads) {
                        dev->props.needs_aligned_loads = true;
                        snprintf(how, sizeof(how), "macOS 12");
                    }

                    const char * force = TOSH_ENV("TOSH_MV_ALIGN");
                    if (force != NULL) {
                        dev->props.needs_aligned_loads = atoi(force) != 0;
                        snprintf(how, sizeof(how), "TOSH_MV_ALIGN=%s", force);
                    }
                    if (is_wave64 || monterey || force != NULL) {
                        fprintf(stderr, "ggml_metal: aligned mat-vec reads %s by %s "
                                        "(override with TOSH_MV_ALIGN=1 or =0)\n",
                                dev->props.needs_aligned_loads ? "ON" : "off", how);
                    }
                }

                // Auto-on for wave64: run the quantized/f16 mat-vec decode on the GPU
                // (large tg win, validated on GCN); escape hatch forces it off.
                // GGML_METAL_WAVE64_SAFEMODE is the user-facing alias the app documents.
                dev->props.wave64_decode = is_wave64 && TOSH_ENV("GGML_METAL_WAVE64_DECODE_DISABLE") == NULL
                                                     && TOSH_ENV("GGML_METAL_WAVE64_SAFEMODE") == NULL;
                if (dev->props.wave64_decode) {
                    fprintf(stderr, "ggml_metal: wave64 decode ON: quantized/f16/bf16 mat-vec on GPU "
                                    "(see the allowlist in ggml_metal_library_get_pipeline_mul_mv)\n");
                }

                dev->props.has_simdgroup_mm = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];

                // Manual tiled mul_mm replaces the missing simdgroup-matrix path on AMD.
                // wave32 gates on simdgroup reduction; wave64 uses the same wave-agnostic kernel.
                dev->props.use_mm_manual =
                    !dev->props.has_simdgroup_mm &&
                    ![dev->mtl_device supportsFamily:MTLGPUFamilyApple1] &&
                    (dev->props.simd_width == 32 ? dev->props.has_simdgroup_reduction : is_wave64);
                if (TOSH_ENV("GGML_METAL_MM_MANUAL_DISABLE") != NULL) {
                    dev->props.use_mm_manual = false;
                }

                dev->props.has_unified_memory = dev->mtl_device.hasUnifiedMemory;

                dev->props.has_bfloat  = [dev->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];
                dev->props.has_bfloat |= [dev->mtl_device supportsFamily:MTLGPUFamilyApple6];
                if (TOSH_ENV("GGML_METAL_BF16_DISABLE") != NULL) {
                    dev->props.has_bfloat = false;
                }

                dev->props.has_tensor = [dev->mtl_device supportsFamily:MTLGPUFamilyMetal4_GGML];
                if (TOSH_ENV("GGML_METAL_TENSOR_DISABLE") != NULL) {
                    dev->props.has_tensor = false;
                }

                // note: disable the tensor API by default for old chips because with the current implementation it is not useful
                // - M2 Ultra:   ~5% slower
                // - M4, M4 Max: no significant difference
                //
                // TODO: try to update the tensor API kernels to at least match the simdgroup performance
                if (TOSH_ENV("GGML_METAL_TENSOR_ENABLE") == NULL &&
                    ![[dev->mtl_device name] containsString:@"M5"] &&
                    ![[dev->mtl_device name] containsString:@"M6"] &&
                    ![[dev->mtl_device name] containsString:@"A19"] &&
                    ![[dev->mtl_device name] containsString:@"A20"]) {
                    GGML_LOG_INFO("%s: tensor API disabled for pre-M5 and pre-A19 devices\n", __func__);
                    dev->props.has_tensor = false;
                }

                // double-check that the tensor API compiles
                if (dev->props.has_tensor) {
                    const char * src_tensor_f16 = "\n"
                        "#include <metal_stdlib> \n"
                        "#include <metal_tensor> \n"
                        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h> \n"
                        " \n"
                        "using namespace metal; \n"
                        "using namespace mpp::tensor_ops; \n"
                        " \n"
                        "kernel void dummy_kernel( \n"
                        "    tensor<device  half, dextents<int32_t, 2>> A [[buffer(0)]], \n"
                        "    tensor<device  half, dextents<int32_t, 2>> B [[buffer(1)]], \n"
                        "    device float * C [[buffer(2)]], \n"
                        "    uint2 tgid [[threadgroup_position_in_grid]]) \n"
                        "{ \n"
                        "    auto tA = A.slice(0, (int)tgid.y); \n"
                        "    auto tB = B.slice((int)tgid.x, 0); \n"
                        " \n"
                        "    matmul2d< \n"
                        "        matmul2d_descriptor(16, 16, dynamic_extent), \n"
                        "        execution_simdgroups<4>> mm; \n"
                        " \n"
                        "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>(); \n"
                        " \n"
                        "    auto sA = tA.slice(0, 0); \n"
                        "    auto sB = tB.slice(0, 0); \n"
                        "    mm.run(sB, sA, cT); \n"
                        " \n"
                        "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16)); \n"
                        " \n"
                        "    cT.store(tC); \n"
                        "}";

                    GGML_LOG_INFO("%s: testing tensor API for f16 support\n", __func__);
                    ggml_metal_library_t lib = ggml_metal_library_init_from_source(dev, src_tensor_f16, false);
                    if (lib == NULL) {
                        GGML_LOG_WARN("%s: - the tensor API is not supported in this environment - disabling\n", __func__);
                        dev->props.has_tensor = false;
                    } else {
                        struct ggml_metal_pipeline_with_params ppl = ggml_metal_library_compile_pipeline(lib, "dummy_kernel", "dummy_kernel", nil);
                        if (!ppl.pipeline) {
                            GGML_LOG_WARN("%s: - the tensor API is not supported in this environment - disabling\n", __func__);
                            dev->props.has_tensor = false;
                        }

                        ggml_metal_library_free(lib);
                    }
                }

                // try to compile a dummy kernel to determine if the tensor API is supported for bfloat
                if (dev->props.has_tensor && dev->props.has_bfloat) {
                    const char * src_tensor_bf16 = "\n"
                        "#include <metal_stdlib> \n"
                        "#include <metal_tensor> \n"
                        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h> \n"
                        " \n"
                        "using namespace metal; \n"
                        "using namespace mpp::tensor_ops; \n"
                        " \n"
                        "kernel void dummy_kernel( \n"
                        "    tensor<device bfloat, dextents<int32_t, 2>> A [[buffer(0)]], \n"
                        "    tensor<device bfloat, dextents<int32_t, 2>> B [[buffer(1)]], \n"
                        "    device float * C [[buffer(2)]], \n"
                        "    uint2 tgid [[threadgroup_position_in_grid]]) \n"
                        "{ \n"
                        "    auto tA = A.slice(0, (int)tgid.y); \n"
                        "    auto tB = B.slice((int)tgid.x, 0); \n"
                        " \n"
                        "    matmul2d< \n"
                        "        matmul2d_descriptor(16, 16, dynamic_extent), \n"
                        "        execution_simdgroups<4>> mm; \n"
                        " \n"
                        "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>(); \n"
                        " \n"
                        "    auto sA = tA.slice(0, 0); \n"
                        "    auto sB = tB.slice(0, 0); \n"
                        "    mm.run(sB, sA, cT); \n"
                        " \n"
                        "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16)); \n"
                        " \n"
                        "    cT.store(tC); \n"
                        "}";

                    GGML_LOG_INFO("%s: testing tensor API for bfloat support\n", __func__);
                    ggml_metal_library_t lib = ggml_metal_library_init_from_source(dev, src_tensor_bf16, false);
                    if (lib == NULL) {
                        GGML_LOG_WARN("%s: - the tensor API does not support bfloat - disabling bfloat support\n", __func__);
                        dev->props.has_bfloat = false;
                    } else {
                        struct ggml_metal_pipeline_with_params ppl = ggml_metal_library_compile_pipeline(lib, "dummy_kernel", "dummy_kernel", nil);
                        if (!ppl.pipeline) {
                            GGML_LOG_WARN("%s: - the tensor API does not support bfloat - disabling bfloat support\n", __func__);
                            dev->props.has_bfloat = false;
                        }

                        ggml_metal_library_free(lib);
                    }
                }

                dev->props.use_residency_sets = true;
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
                dev->props.use_residency_sets = TOSH_ENV("GGML_METAL_NO_RESIDENCY") == nil;
#endif

                dev->props.use_shared_buffers = dev->props.has_unified_memory;
#if TARGET_OS_OSX && TARGET_CPU_X86_64
                // In case of eGPU, shared memory may be preferable.
                dev->props.use_shared_buffers |= [dev->mtl_device location] == MTLDeviceLocationExternal;
#endif
                if (TOSH_ENV("GGML_METAL_SHARED_BUFFERS_DISABLE") != NULL) {
                    dev->props.use_shared_buffers = false;
                }
                if (TOSH_ENV("GGML_METAL_SHARED_BUFFERS_ENABLE") != NULL) {
                    dev->props.use_shared_buffers = true;
                }

                dev->props.device_id = ggml_metal_device_id_parse([[dev->mtl_device name] UTF8String]);

                dev->props.op_offload_min_batch_size  = TOSH_ENV("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(TOSH_ENV("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

                dev->props.max_buffer_size            = dev->mtl_device.maxBufferLength;

                // a driver that advertises a large maxBufferLength can still refuse one allocation
                // of that size. Capping what we advertise makes ggml split the weights into several
                // buffers itself, which is the one place that knows no tensor may straddle them.
                // The 4 GiB failure that motivated this was reported by Slice on an RX 570:
                // https://www.insanelymac.com/forum/profile/112217-slice/
                if (TOSH_ENV("TOSH_METAL_MAX_BUFFER_MB")) {
                    const size_t cap = (size_t) atoll(TOSH_ENV("TOSH_METAL_MAX_BUFFER_MB")) * 1024 * 1024;
                    if (cap > 0 && cap < dev->props.max_buffer_size) {
                        dev->props.max_buffer_size = cap;
                        fprintf(stderr, "ggml_metal: max single buffer capped to %zu MiB by TOSH_METAL_MAX_BUFFER_MB\n", cap / 1024 / 1024);
                    }
                }
                dev->props.max_theadgroup_memory_size = dev->mtl_device.maxThreadgroupMemoryLength;
                if (@available(macOS 10.12, iOS 16.0, *)) {
                    dev->props.max_working_set_size   = dev->mtl_device.recommendedMaxWorkingSetSize;
                } else {
                    dev->props.max_working_set_size   = dev->mtl_device.maxBufferLength;
                }
#ifdef TOSH_ENABLE_DYNAMIC_MOE
                // the expert arena is sized from what the card has free before anything is allocated;
                // taken for every process, since mixed execution can be chosen after the device starts
                {
                    tosh_dmoe_set_private_query(ggml_metal_private_bytes);
                    if (g_vram_dev == nil) {
                        g_vram_dev = dev->mtl_device;
                        tosh_dmoe_set_vram_query(ggml_metal_vram_query);
                    }
                    size_t vfree = 0, vtotal = 0;
                    if (tosh_vram_read(&vfree, &vtotal)) {
                        tosh_dmoe_set_vram(vfree, vtotal);
                    } else {
                        tosh_dmoe_set_vram(dev->props.max_working_set_size, dev->props.max_working_set_size);
                    }
                }
#endif

                {
                    const char * val = getenv("GGML_METAL_FUSION_DEBUG");
                    dev->finfo = ggml_metal_fusion_info_init(
                            getenv("GGML_METAL_FUSION_DISABLE") == nil,
                            val ? atoi(val) : 0);
                }

                snprintf(dev->props.name, sizeof(dev->props.name), "%s%d", "MTL", device);
                const char * gpu_name = [[dev->mtl_device name] UTF8String];
                if (n_devices > 1) {
                    snprintf(dev->props.desc, sizeof(dev->props.desc), "%s (dev p%d/v%d)",
                             gpu_name, dev->props.device_phys, dev->props.device_virt);
                } else {
                    snprintf(dev->props.desc, sizeof(dev->props.desc), "%s", gpu_name);
                }

                dev->library = ggml_metal_library_init(dev);
                if (!dev->library) {
                    GGML_LOG_ERROR("%s: error: failed to create library\n", __func__);
                }

                if (dev->props.use_residency_sets) {
                    dev->rsets = ggml_metal_rsets_init(dev);
                } else {
                    dev->rsets = nil;
                }

                // print MTL GPU family:
                GGML_LOG_INFO("%s: GPU name:   %s (%s)\n", __func__, dev->props.name, dev->props.desc);

                // determine max supported GPU family
                // https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
                // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
                {
                    for (int i = MTLGPUFamilyApple1 + 20; i >= MTLGPUFamilyApple1; --i) {
                        if ([dev->mtl_device supportsFamily:i]) {
                            dev->props.gpu_family = i - (int) MTLGPUFamilyApple1 + 1;
                            GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyApple%d  (%d)\n", __func__, dev->props.gpu_family, i);
                            break;
                        }
                    }

#if TARGET_CPU_X86_64
                    for (int i = MTLGPUFamilyCommon1 + 5; i >= MTLGPUFamilyCommon1; --i) {
                        if ([dev->mtl_device supportsFamily:i]) {
                            GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyCommon%d (%d)\n", __func__, i - (int) MTLGPUFamilyCommon1 + 1, i);
                            break;
                        }
                    }
#endif

                    for (int i = MTLGPUFamilyMetal3_GGML + 5; i >= MTLGPUFamilyMetal3_GGML; --i) {
                        if ([dev->mtl_device supportsFamily:i]) {
                            GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyMetal%d  (%d)\n", __func__, i - (int) MTLGPUFamilyMetal3_GGML + 3, i);
                            break;
                        }
                    }
                }

                GGML_LOG_INFO("%s: simdgroup reduction   = %s\n", __func__, dev->props.has_simdgroup_reduction ? "true" : "false");
                GGML_LOG_INFO("%s: simdgroup matrix mul. = %s\n", __func__, dev->props.has_simdgroup_mm        ? "true" : "false");
                GGML_LOG_INFO("%s: has unified memory    = %s\n", __func__, dev->props.has_unified_memory      ? "true" : "false");
                GGML_LOG_INFO("%s: has bfloat            = %s\n", __func__, dev->props.has_bfloat              ? "true" : "false");
                GGML_LOG_INFO("%s: has tensor            = %s\n", __func__, dev->props.has_tensor              ? "true" : "false");
                GGML_LOG_INFO("%s: use residency sets    = %s\n", __func__, dev->props.use_residency_sets      ? "true" : "false");
                GGML_LOG_INFO("%s: use shared buffers    = %s\n", __func__, dev->props.use_shared_buffers      ? "true" : "false");

#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
                if (@available(macOS 10.12, iOS 16.0, *)) {
                    GGML_LOG_INFO("%s: recommendedMaxWorkingSetSize  = %8.2f MB\n", __func__, dev->props.max_working_set_size / 1e6);
                }
#endif
            }
        }
    }

    return dev;
}

void ggml_metal_device_free(ggml_metal_device_t dev) {
    assert(dev != NULL);

    @autoreleasepool {
        ggml_metal_fusion_info_free(dev->finfo);

        ggml_metal_rsets_free(dev->rsets);

        ggml_metal_library_free(dev->library);
        dev->library = NULL;

        if (dev->stage_buf) {
            [dev->stage_buf release];
            dev->stage_buf = nil;
        }

        ggml_metal_device_upload_drain(dev);
        for (int i = 0; i < 4; i++) {
            [dev->stage_set_bufs[i] release];
            dev->stage_set_bufs[i] = nil;
        }

        if (dev->stage_lock) {
            [dev->stage_lock release];
            dev->stage_lock = nil;
        }

        for (int i = 0; i < dev->n_host_wraps; i++) {
            [dev->host_wraps[i].buf release];
        }
        dev->n_host_wraps = 0;
        dev->host_wrap_bytes = 0;

        if (dev->wrap_lock) {
            [dev->wrap_lock release];
            dev->wrap_lock = nil;
        }

        if (dev->mtl_queue) {
            [dev->mtl_queue release];
            dev->mtl_queue = nil;
        }

        if (dev->mtl_device) {
            [dev->mtl_device release];
            dev->mtl_device = nil;
        }
    }

    free(dev);
}

void * ggml_metal_device_get_obj(ggml_metal_device_t dev) {
    return dev->mtl_device;
}

void * ggml_metal_device_get_queue(ggml_metal_device_t dev) {
    return dev->mtl_queue;
}

// prefetch contexts get their own queue so their blits can overlap the
// primary context's compute; falls back to the shared queue on failure
void * ggml_metal_device_acquire_queue(ggml_metal_device_t dev, bool * owned) {
    *owned = false;

    id<MTLCommandQueue> queue = ggml_metal_new_queue(dev->mtl_device);
    if (queue == nil) {
        return dev->mtl_queue;
    }

    GGML_LOG_INFO("%s: created a secondary command queue for a prefetch context\n", __func__);

    *owned = true;
    return queue;
}

void ggml_metal_device_release_queue(ggml_metal_device_t dev, void * queue_raw, bool owned) {
    if (owned) {
        id<MTLCommandQueue> queue = (id<MTLCommandQueue>) queue_raw;
        [queue release];
    }

    GGML_UNUSED(dev);
}

// cached no-copy wrap of the host pages containing [data, data + size), so
// repeated uploads of stable host memory need no allocation or memcpy
void * ggml_metal_device_wrap_host(ggml_metal_device_t dev, const void * data, size_t size, size_t * offs) {
    if (size == 0 || data == NULL) {
        return NULL;
    }

    const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);

    const uintptr_t base = (uintptr_t) data & ~(page - 1);
    const size_t    len  = (((uintptr_t) data + size + page - 1) & ~(page - 1)) - base;

    [dev->wrap_lock lock];

    for (int i = 0; i < dev->n_host_wraps; i++) {
        if (base >= (uintptr_t) dev->host_wraps[i].base &&
            base + len <= (uintptr_t) dev->host_wraps[i].base + dev->host_wraps[i].size) {
            id<MTLBuffer> buf = [[dev->host_wraps[i].buf retain] autorelease];
            [dev->wrap_lock unlock];

            *offs = (uintptr_t) data - (uintptr_t) dev->host_wraps[i].base;
            return buf;
        }
    }

    // A no-copy host buffer only makes already allocated system-RAM pages addressable by Metal;
    // it does not make them private VRAM residents. recommendedMaxWorkingSetSize is therefore the
    // wrong ceiling: on a discrete GPU it rejects the exact oversized models this cache exists to
    // serve. The device's maximum *single buffer* size is the real structural constraint. Keep the
    // aggregate limit only as an explicit laboratory/debug override.
    if (len > dev->props.max_buffer_size) {
        [dev->wrap_lock unlock];
        return NULL;
    }

    // Left uncapped, a large MoE offload maps its whole host-resident expert set here and the
    // device stops granting any new shared buffer: measured failing a 1.21 GiB allocation with
    // 142.75 GiB wrapped against a 34.34 GiB working set. The cap costs no speed (measured
    // identical at 8 and 16 GiB) and sends the overflow to the bounded staging ring instead.
    // GGML_METAL_HOST_WRAP_LIMIT_MB overrides it; 0 restores the uncapped behaviour.
    size_t limit = (size_t) dev->props.max_working_set_size;
    const char * limit_mb = TOSH_ENV("GGML_METAL_HOST_WRAP_LIMIT_MB");
    if (limit_mb != NULL) {
        limit = (size_t) strtoull(limit_mb, NULL, 10)*1024*1024;
    }
#ifdef TOSH_ENABLE_DYNAMIC_MOE
    // with every bank in host memory a full working set of wraps collapses the uploads;
    // two thirds of it keeps the fast path
    else if (tosh_dmoe_enabled()) {
        limit = limit/3*2;
    }
#endif
    if (limit > 0 && (len > limit || dev->host_wrap_bytes > limit - len)) {
        [dev->wrap_lock unlock];
        return NULL;
    }

    id<MTLBuffer> buf = [dev->mtl_device newBufferWithBytesNoCopy:(void *) base
                                                           length:len
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
    if (buf == nil) {
        [dev->wrap_lock unlock];
        return NULL;
    }

    if (len >= (size_t) 1024*1024*1024) {
        GGML_LOG_INFO("%s: mapped %.2f GiB of existing host RAM as one Metal buffer (not VRAM)\n",
                __func__, (double) len/(1024.0*1024.0*1024.0));
    }

    if (dev->n_host_wraps == GGML_METAL_HOST_WRAP_MAX) {
        // evict the oldest entry; in-flight command buffers retain the resource
        dev->host_wrap_bytes -= dev->host_wraps[0].size;
        [dev->host_wraps[0].buf release];
        memmove(&dev->host_wraps[0], &dev->host_wraps[1], (GGML_METAL_HOST_WRAP_MAX - 1)*sizeof(dev->host_wraps[0]));
        dev->n_host_wraps--;
    }

    dev->host_wraps[dev->n_host_wraps].base = (const void *) base;
    dev->host_wraps[dev->n_host_wraps].size = len;
    dev->host_wraps[dev->n_host_wraps].buf  = buf;
    dev->n_host_wraps++;
    dev->host_wrap_bytes += len;

    [[buf retain] autorelease];

    [dev->wrap_lock unlock];

    *offs = (uintptr_t) data - base;
    return buf;
}

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev) {
    return dev->library;
}

void ggml_metal_device_rsets_add(ggml_metal_device_t dev, ggml_metal_rset_t rset) {
    if (rset == nil) {
        return;
    }

    GGML_ASSERT(dev->rsets);

    [dev->rsets->lock lock];

    [dev->rsets->data addObject:rset];

    [dev->rsets->lock unlock];
}

void ggml_metal_device_rsets_rm(ggml_metal_device_t dev, ggml_metal_rset_t rset) {
    if (rset == nil) {
        return;
    }

    GGML_ASSERT(dev->rsets);

    [dev->rsets->lock lock];

    [dev->rsets->data removeObject:rset];

    [dev->rsets->lock unlock];
}

void ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev) {
    if (dev->rsets == NULL) {
        return;
    }

    atomic_store_explicit(&dev->rsets->d_loop, dev->rsets->loops_per_s*dev->rsets->keep_alive_s, memory_order_relaxed);
}

struct ggml_metal_event {
    void * obj; // id<MTLSharedEvent>

    atomic_int value;
};

void ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf_raw) {
    id<MTLSharedEvent> event = (id<MTLSharedEvent>)ev->obj;

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    [cmd_buf encodeSignalEvent:event value:atomic_fetch_add_explicit(&ev->value, 1, memory_order_relaxed) + 1];
}

void ggml_metal_event_encode_wait(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf_raw) {
    id<MTLSharedEvent> event = (id<MTLSharedEvent>)ev->obj;

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    [cmd_buf encodeWaitForEvent:event value:atomic_load_explicit(&ev->value, memory_order_relaxed)];
}

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev) {
    id<MTLSharedEvent> event = [dev->mtl_device newSharedEvent];

    ggml_metal_event_t ev = calloc(1, sizeof(struct ggml_metal_event));

    ev->obj = (__bridge void *)event;
    ev->value = 0;

    return ev;
}

void ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLSharedEvent> event = ev->obj;
        [event release];

        free(ev);

        GGML_UNUSED(dev);
    }
}

void ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev) {
    id<MTLSharedEvent> event = ev->obj;
    const bool res = [event waitUntilSignaledValue:atomic_load_explicit(&ev->value, memory_order_relaxed) timeoutMS:60000];
    if (!res) {
        GGML_ABORT("%s: failed to wait for event\n", __func__);
    }

    GGML_UNUSED(dev);
}


void ggml_metal_device_get_memory(ggml_metal_device_t dev, size_t * free, size_t * total) {
    if (@available(macOS 10.12, iOS 16.0, *)) {
        *total     = dev->mtl_device.recommendedMaxWorkingSetSize;
        size_t cur = dev->mtl_device.currentAllocatedSize;
        // it's possible to allocate more than `recommendedMaxWorkingSetSize`
        *free      = *total > cur ? *total - cur : 0;
        // on a discrete card the display and other processes hold VRAM too, which only the
        // driver's counters see; --fit would otherwise place layers in memory that is taken
        int64_t vfree = 0;
        int64_t vused = 0;
        if (!dev->props.has_unified_memory && ggml_metal_vram_stats(dev->mtl_device, &vfree, &vused) && vfree >= 0) {
            *free = MIN(*free, (size_t) vfree);
        }
    } else {
        *free = 0;
        *total = 0;
    }
}

static bool ggml_metal_supports_mul_mat_op(
        bool has_simdgroup_reduction,
        const struct ggml_tensor * op,
        bool src0_f16_has_mv,
        bool mm_path) {
    if (!has_simdgroup_reduction ||
        op->src[0]->type == GGML_TYPE_NVFP4 ||
        op->src[0]->type == GGML_TYPE_TQ1_0) {
        return false;
    }

    if (op->src[1]->type != GGML_TYPE_F16) {
        return true;
    }

    if (op->src[0]->type == GGML_TYPE_BF16) {
        return false;
    }

    if (src0_f16_has_mv && op->src[0]->type == GGML_TYPE_F16) {
        return true;
    }

    return mm_path;
}

// TQ1_0/TQ2_0 have no Metal kernels at all: claiming them compiles a pipeline that does
// not exist, which aborts instead of falling back to the CPU backend
static bool ggml_metal_type_is_ternary(enum ggml_type t) {
    return t == GGML_TYPE_TQ1_0 || t == GGML_TYPE_TQ2_0;
}

bool ggml_metal_device_supports_op(ggml_metal_device_t dev, const struct ggml_tensor * op) {
    const bool has_simdgroup_mm        = dev->props.has_simdgroup_mm;
    const bool has_simdgroup_reduction = dev->props.has_simdgroup_reduction;
    const bool use_mm_manual           = dev->props.use_mm_manual;
    const bool wave64_decode           = dev->props.wave64_decode;
    const bool has_bfloat              = dev->props.has_bfloat;

    // No bfloat in Metal: bf16 runs on the kernels that widen the raw bits by hand, and only
    // the ops that have one. Anything else has to stay off the device, weights included: a
    // pre-allocated tensor the scheduler cannot place aborts the whole load.
    if (!has_bfloat) {
        bool bf16 = op->type == GGML_TYPE_BF16;
        for (size_t i = 0, n = 3; i < n; ++i) {
            bf16 = bf16 || (op->src[i] != NULL && op->src[i]->type == GGML_TYPE_BF16);
        }

        if (bf16) {
            switch (op->op) {
                case GGML_OP_NONE:
                case GGML_OP_RESHAPE:
                case GGML_OP_VIEW:
                case GGML_OP_PERMUTE:
                case GGML_OP_TRANSPOSE:
                    return true;
                case GGML_OP_GET_ROWS:
                    return op->src[0]->type == GGML_TYPE_BF16 && op->type == GGML_TYPE_F32;
                case GGML_OP_MUL_MAT:
                case GGML_OP_MUL_MAT_ID:
                    // the emulated set covers bf16 weights against f32 activations
                    if (op->src[0]->type != GGML_TYPE_BF16 || op->src[1]->type != GGML_TYPE_F32) {
                        return false;
                    }
                    break;
                case GGML_OP_CPY:
                case GGML_OP_CONT:
                case GGML_OP_DUP:
                    return (op->src[0]->type == GGML_TYPE_F32  && op->type == GGML_TYPE_BF16) ||
                           (op->src[0]->type == GGML_TYPE_BF16 && op->type == GGML_TYPE_F32);
                default:
                    return false;
            }
        }
    }

    switch (op->op) {
#ifdef TOSH_ENABLE_DYNAMIC_MOE
        case GGML_OP_TOSH_DMOE_PUBLISH:
        case GGML_OP_TOSH_DMOE_JOIN:
        case GGML_OP_TOSH_DMOE_OBSERVE:
            return true;
#endif
        case GGML_OP_SCALE:
        case GGML_OP_FILL:
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
            return ggml_is_contiguous_rows(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                case GGML_UNARY_OP_XIELU:
                    return ggml_is_contiguous_rows(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
                default:
                    return false;
            }
        case GGML_OP_SILU_BACK:
            return (op->src[0]->type == GGML_TYPE_F32) &&
                (op->src[1]->type == GGML_TYPE_F32) &&
                (op->type == GGML_TYPE_F32) &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_is_contiguous(op) &&
                ggml_are_same_shape(op->src[0], op->src[1]);
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    return ggml_is_contiguous_1(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
               default:
                    return false;
            }
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
            return true;
        case GGML_OP_CONCAT:
            {
                const enum ggml_type src0_type = op->src[0]->type;
                const enum ggml_type src1_type = op->src[1]->type;
                if (src0_type != src1_type || src0_type != op->type) {
                    return false;
                }
                switch (src0_type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_I8:
                    case GGML_TYPE_I16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_I64:
                        return true;
                    case GGML_TYPE_BF16:
                        return has_bfloat;
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_ADD_ID:
            return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) && (op->src[0]->type == op->src[1]->type);
        case GGML_OP_ACC:
            return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_REPEAT:
        case GGML_OP_CONV_TRANSPOSE_1D:
            return true;
        case GGML_OP_CONV_TRANSPOSE_2D:
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) &&
                (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_COL2IM_1D:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) &&
                op->type == op->src[0]->type &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op);
        case GGML_OP_CONV_3D:
            return ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_SUM:
            // wave64: nsg and shmem are sized by the real width, so the reduction is width-agnostic.
            return (has_simdgroup_reduction || wave64_decode) && ggml_is_contiguous(op->src[0]);
        case GGML_OP_TRI:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_SOFT_MAX:
            // wave64: soft_max is now width-agnostic, so allow it on GPU behind the decode opt-in.
            return (has_simdgroup_reduction || wave64_decode) && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_SUM_ROWS:
        case GGML_OP_CUMSUM:
        case GGML_OP_MEAN:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_L2_NORM:
            // wave64: width-agnostic kernels (real-width shmem and lane indexing).
            return (has_simdgroup_reduction || wave64_decode) && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_COUNT_EQUAL:
            return (has_simdgroup_reduction || wave64_decode) &&
                op->src[0]->type == GGML_TYPE_I32 &&
                op->src[1]->type == GGML_TYPE_I32 &&
                op->type == GGML_TYPE_I64;
        case GGML_OP_ARGMAX:
            // wave64: shmem is sized and split by the real width instead of N_SIMDWIDTH.
            return has_simdgroup_reduction || wave64_decode;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            // wave64: norm/rms_norm kernels are width-agnostic (simd_sum + shmem sized by
            // the real width), so allow them on GPU behind the decode opt-in.
            return (has_simdgroup_reduction || wave64_decode) && (ggml_is_contiguous_rows(op->src[0]));
        // the backward kernels reduce with 32-lane shared memory, so 64-lane GPUs keep them on the CPU
        case GGML_OP_RMS_NORM_BACK:
            return has_simdgroup_reduction && dev->props.simd_width == 32 &&
                   op->type == GGML_TYPE_F32 &&
                   op->src[0] != NULL && op->src[1] != NULL &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   op->ne[0] % 4 == 0 &&
                   ggml_is_contiguous_1(op->src[0]) &&
                   ggml_is_contiguous_1(op->src[1]) &&
                   ggml_is_contiguous_1(op) &&
                   ggml_are_same_shape(op, op->src[0]) &&
                   ggml_are_same_shape(op, op->src[1]);
        case GGML_OP_REPEAT_BACK:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        // the training loss kernels reduce with 32-lane shared memory, so 64-lane GPUs keep them on the CPU
        case GGML_OP_CROSS_ENTROPY_ROWS:
            return has_simdgroup_reduction && dev->props.simd_width == 32 &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
        case GGML_OP_CROSS_ENTROPY_ROWS_BACK:
            return has_simdgroup_reduction && dev->props.simd_width == 32 &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) &&
                   ggml_is_contiguous(op->src[2]) && ggml_is_contiguous(op);
        case GGML_OP_FLASH_ATTN_EXT_BACK:
            {
                const struct ggml_tensor * q = op->src[0];
                const struct ggml_tensor * k = op->src[1];
                const struct ggml_tensor * v = op->src[2];
                const struct ggml_tensor * m = op->src[3];
                const int64_t d = k->ne[0];
                return has_simdgroup_reduction && dev->props.simd_width == 32 &&
                       (d == 64 || d == 128 || d == 256) && v->ne[0] == d && q->ne[0] == d &&
                       q->type == GGML_TYPE_F32 && (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16) && v->type == k->type &&
                       op->src[4]->type == GGML_TYPE_F32 && op->src[5]->type == GGML_TYPE_F32 &&
                       (!m || m->type == GGML_TYPE_F16) &&
                       q->nb[0] == sizeof(float) && k->nb[0] == ggml_type_size(k->type) && v->nb[0] == ggml_type_size(v->type) &&
                       op->src[4]->nb[0] == sizeof(float) && op->src[5]->nb[0] == sizeof(float) &&
                       ggml_is_contiguous(op);
            }
        case GGML_OP_OUT_PROD:
            {
                // the transposed loader is written for the manual 32-lane GEMM only
                const struct ggml_tensor * a = op->src[0];
                const struct ggml_tensor * b = op->src[1];
                const bool flat = a->ne[2] == 1 && a->ne[3] == 1 && b->ne[2] == 1 && b->ne[3] == 1 &&
                                  op->type == GGML_TYPE_F32 && ggml_is_contiguous(op);
                if (flat && a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && a->nb[0] == sizeof(float) &&
                    b->nb[0] == sizeof(float) && (a->ne[0] <= 32 || b->ne[0] <= 32)) {
                    return true;
                }
                return dev->props.use_mm_manual && dev->props.simd_width == 32 && dev->props.has_bfloat &&
                       (a->type == GGML_TYPE_Q8_0 || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_F32) &&
                       a->nb[0] == ggml_type_size(a->type) && a->ne[0] % 32 == 0 &&
                       b->type == GGML_TYPE_F32 && b->nb[1] == sizeof(float) && b->nb[0] % 16 == 0 &&
                       a->ne[2] == op->ne[2] && a->ne[3] == op->ne[3] && b->ne[2] == op->ne[2] && b->ne[3] == op->ne[3] &&
                       op->type == GGML_TYPE_F32 && ggml_is_contiguous(op);
            }
        case GGML_OP_GATED_DELTA_NET_BACK:
            {
                bool ok = has_simdgroup_reduction && dev->props.simd_width == 32 &&
                          dev->props.max_theadgroup_memory_size >= 52*1024 &&
                          op->src[2]->ne[0] == 128 && op->src[0]->ne[0] == 128 && op->src[3]->ne[0] == 1 &&
                          ggml_is_contiguous(op->src[5]) && ggml_is_contiguous(op->src[6]);
                for (int i = 0; i < 7; ++i) {
                    ok = ok && op->src[i]->type == GGML_TYPE_F32 && op->src[i]->nb[0] == sizeof(float);
                }
                return ok;
            }
        case GGML_OP_COUNT_EQUAL_MASKED:
            return has_simdgroup_reduction && dev->props.simd_width == 32 &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op->src[2]);
        case GGML_OP_SOFT_MAX_BACK:
            {
                if (!has_simdgroup_reduction || dev->props.simd_width != 32 ||
                    op->type != GGML_TYPE_F32 ||
                    op->src[0] == NULL || op->src[1] == NULL ||
                    op->src[0]->type != GGML_TYPE_F32 ||
                    op->src[1]->type != GGML_TYPE_F32 ||
                    !ggml_is_contiguous_1(op->src[0]) ||
                    !ggml_is_contiguous_1(op->src[1]) ||
                    !ggml_is_contiguous_1(op) ||
                    !ggml_are_same_shape(op, op->src[0]) ||
                    !ggml_are_same_shape(op, op->src[1])) {
                    return false;
                }

                float max_bias = 0.0f;
                memcpy(&max_bias, ((const float *) op->op_params) + 1, sizeof(float));
                return max_bias == 0.0f;
            }
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return true;
        case GGML_OP_IM2COL:
            return ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32 && (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);
        case GGML_OP_CONV_2D:
            return ggml_is_contiguous(op->src[0]) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
        case GGML_OP_CONV_2D_DW:
            return op->src[1]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
        case GGML_OP_UPSCALE:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_POOL_1D:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_POOL_2D:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_PAD:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_TIMESTEP_EMBEDDING:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_LEAKY_RELU:
            return op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16;
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_ARANGE:
            return true;
        case GGML_OP_ROLL:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_FLASH_ATTN_EXT:
            // for new head sizes, add checks here
            if (op->src[0]->ne[0] != 32 &&
                op->src[0]->ne[0] != 40 &&
                op->src[0]->ne[0] != 48 &&
                op->src[0]->ne[0] != 64 &&
                op->src[0]->ne[0] != 72 &&
                op->src[0]->ne[0] != 80 &&
                op->src[0]->ne[0] != 96 &&
                op->src[0]->ne[0] != 112 &&
                op->src[0]->ne[0] != 128 &&
                op->src[0]->ne[0] != 192 &&
                op->src[0]->ne[0] != 256 &&
                op->src[0]->ne[0] != 320 &&
                op->src[0]->ne[0] != 384 &&
                op->src[0]->ne[0] != 512 &&
                op->src[0]->ne[0] != 576 &&
                op->src[0]->ne[0] != 640) {
                return false;
            }
            // AMD dGPUs report no simdgroup_mm and miscompile the upstream vec kernel, so FA is
            // allowed only for the KV pairs our own vec kernel instantiates. This sits before
            // the K==V gate because that kernel takes K and V independently.
            const bool fa_amd_width_ok = dev->props.simd_width == 32
                ? has_simdgroup_reduction
                : wave64_decode;
            if (TOSH_ENV("TOSH_FA_AMD") != NULL &&
                !has_simdgroup_mm && fa_amd_width_ok &&
                (op->src[0]->ne[0] % 32 == 0 || op->src[0]->ne[0] == 72 ||
                 op->src[0]->ne[0] == 40 || op->src[0]->ne[0] == 80 || op->src[0]->ne[0] == 160)) {
                float max_bias = 0.0f;
                float softcap  = 0.0f;
                memcpy(&max_bias, ((const int32_t *) op->op_params) + 1, sizeof(float));
                memcpy(&softcap,  ((const int32_t *) op->op_params) + 2, sizeof(float));
                const enum ggml_type kt = op->src[1]->type;
                const enum ggml_type vt = op->src[2]->type;
                const bool k_turbo = (kt == GGML_TYPE_TURBO2_0 || kt == GGML_TYPE_TURBO3_0 || kt == GGML_TYPE_TURBO4_0);
                const bool v_turbo = (vt == GGML_TYPE_TURBO2_0 || vt == GGML_TYPE_TURBO3_0 || vt == GGML_TYPE_TURBO4_0);
                // f32 K/V only have prefill and tile kernels, which need enough queries to be picked
                const bool kv_f32 = kt == GGML_TYPE_F32 && vt == GGML_TYPE_F32 && dev->props.simd_width == 32 &&
                                    op->src[0]->ne[1] >= 32;
                const bool k_std = (kt == GGML_TYPE_F16 || kt == GGML_TYPE_Q8_0 || kt == GGML_TYPE_Q4_0 || k_turbo || kv_f32);
                const bool v_std = (vt == GGML_TYPE_F16 || vt == GGML_TYPE_Q8_0 || vt == GGML_TYPE_Q4_0 || v_turbo || kv_f32);
                const int64_t dk = op->src[0]->ne[0];
                // Turbo KV is instantiated for every supported 128-element padded head.
                const bool turbo_dk_ok = (!k_turbo && !v_turbo) ||
                    dk == 128 || dk == 256 || dk == 384 || dk == 512 || dk == 640;
                // A NULL mask is bidirectional attention (vision towers); the kernels
                // read args.has_mask and treat it as a zero bias.
                // dk 72 is vision-only and f16-only: it is not a whole number of
                // 32-element quant blocks, so K/V can never be quantized there.
                const bool f16_kv = kt == GGML_TYPE_F16 && vt == GGML_TYPE_F16;
                const bool dk_ok =
                    dk == 64 || dk == 128 || dk == 256 || dk == 512 ||
                    // dk 384/640 are instantiated for Turbo KV only
                    ((dk == 384 || dk == 640) && (k_turbo || v_turbo)) ||
                    (dk == 72 && f16_kv) ||
                    // UNet diffusion heads (SD 1.5): not whole quant blocks, so f16 only
                    ((dk == 40 || dk == 80 || dk == 160) && f16_kv);
                const bool simple =
                    dk_ok &&
                    op->src[2]->ne[0] == dk &&
                    k_std && v_std && turbo_dk_ok &&
                    max_bias == 0.0f;
                if (simple) {
                    return true;
                }
                // MLA: asymmetric head, K is the latent plus the rope tail and V is the bare
                // latent. deepseek2 is 576/512; mistral4 has a 256-wide latent, so 320/256.
                const int64_t dv = op->src[2]->ne[0];
                const bool mla_amd =
                    ((dk == 576 && dv == 512) || (dk == 320 && dv == 256)) &&
                    (kt == GGML_TYPE_F16 || kt == GGML_TYPE_Q8_0) && kt == vt &&
                    max_bias == 0.0f && softcap == 0.0f;
                if (mla_amd) {
                    return true;
                }
            }
            // dk 384/640 have no generic instantiation; Turbo KV there is AMD-only (above)
            if (op->src[0]->ne[0] == 384 || op->src[0]->ne[0] == 640) {
                return false;
            }
            if (op->src[1]->ne[0] == 72 && op->src[1]->ne[0] != op->src[2]->ne[0]) {
                return false;
            }
            if (op->src[1]->ne[0] < op->src[2]->ne[0]) {
                return false;
            }
            if (op->src[1]->type != op->src[2]->type) {
                return false;
            }
            switch (op->src[1]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                    break;
                case GGML_TYPE_BF16:
                    if (!has_bfloat) {
                        return false;
                    }
                    break;
                default:
                    return false;
            }
            return has_simdgroup_mm; // TODO: over-restricted for vec-kernels
        case GGML_OP_LIGHTNING_INDEXER:
            if (op->src[0]->ne[0] != OP_LIGHTNING_INDEXER_DK) {
                return false;
            }
            if ((!has_simdgroup_mm && op->src[1]->type == GGML_TYPE_BF16) ||
                op->src[0]->type != GGML_TYPE_F32 ||
                op->src[2]->type != GGML_TYPE_F32 ||
                op->src[3]->type != GGML_TYPE_F16 ||
                op->type         != GGML_TYPE_F32 ||
                !ggml_is_contiguous_rows(op->src[0]) ||
                !ggml_is_contiguous_rows(op->src[1]) ||
                !ggml_is_contiguous_rows(op->src[2]) ||
                !ggml_is_contiguous_rows(op->src[3])) {
                return false;
            }
            switch (op->src[1]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return true;
                case GGML_TYPE_BF16:
                    return has_bfloat;
                default:
                    return false;
            }
        case GGML_OP_DSV4_HC_COMB:
            return has_simdgroup_reduction &&
                op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->type         == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == 24 &&
                op->src[1]->ne[0] >= 3 &&
                op->src[2]->ne[0] == 24 &&
                ggml_is_contiguous_rows(op->src[0]) &&
                ggml_is_contiguous_rows(op->src[1]) &&
                ggml_is_contiguous_rows(op->src[2]);
        case GGML_OP_DSV4_HC_PRE:
            return (has_simdgroup_reduction || wave64_decode) &&
                op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type         == GGML_TYPE_F32 &&
                ggml_is_contiguous_rows(op->src[0]) &&
                ggml_is_contiguous_rows(op->src[1]);
        case GGML_OP_DSV4_HC_POST:
            // 64-lane cards only have the kernel without a combination matrix
            return (has_simdgroup_reduction || (wave64_decode && op->src[3] == NULL)) &&
                op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                (op->src[3] == NULL || op->src[3]->type == GGML_TYPE_F32) &&
                op->type         == GGML_TYPE_F32 &&
                op->src[1]->ne[1] == 4 &&
                op->src[2]->ne[0] == 4 &&
                (op->src[3] == NULL || (op->src[3]->ne[0] == 4 && op->src[3]->ne[1] == 4)) &&
                ggml_is_contiguous_rows(op->src[0]) &&
                ggml_is_contiguous_rows(op->src[1]) &&
                ggml_is_contiguous_rows(op->src[2]) &&
                (op->src[3] == NULL || ggml_is_contiguous_rows(op->src[3]));
        case GGML_OP_SSM_SCAN:
            // wave64: NW, nsg and the shmem layout follow the real width.
            return has_simdgroup_reduction || wave64_decode;
        case GGML_OP_SSM_CONV:
            // wave64: per-thread serial dot, no simdgroup reduction needed.
            return has_simdgroup_reduction || wave64_decode;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            // wave64: the kernel reduces within 32-lane halves (gdn_sum32).
            return (has_simdgroup_reduction || wave64_decode) && op->src[2]->ne[0] % 32 == 0;
        case GGML_OP_TURBO_WHT:
            // 128-element group WHT rotation; head_dim must be a multiple of 128.
            return op->src[0]->type == GGML_TYPE_F32 && op->src[0]->ne[0] % 128 == 0;
        case GGML_OP_SOLVE_TRI:
            return has_simdgroup_reduction && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_MUL_MAT: {
            // the FWHT kernels read an F16 source directly; every other F16 src1 path
            // still goes through ggml_metal_supports_mul_mat_op
            if (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F16 &&
                ggml_metal_op_mul_mat_use_fwht(op, dev->props.max_theadgroup_memory_size)) {
                return has_simdgroup_reduction;
            }
            if (op->src[0]->type == GGML_TYPE_NVFP4 || ggml_metal_type_is_ternary(op->src[0]->type)) {
                return false;
            }
            if (has_simdgroup_reduction) {
                return ggml_metal_supports_mul_mat_op(
                        has_simdgroup_reduction, op, true,
                        ggml_metal_op_mul_mat_use_mm(op, has_simdgroup_mm));
            }
            // wave64 prefill: the manual tiled mul_mm (batched). Mirror the dispatch check.
            const bool prefill =
                use_mm_manual &&
                !ggml_is_transposed(op->src[0]) && !ggml_is_transposed(op->src[1]) &&
                op->src[0]->ne[0] >= 64 && op->src[1]->ne[1] > 8;
            // wave64 decode (experimental, opt-in): the width-parameterized mat-vec kernels.
            const bool decode =
                wave64_decode && op->src[0]->ne[0] >= 32 &&
                (op->src[0]->type == GGML_TYPE_F16 ||
                 op->src[0]->type == GGML_TYPE_F32 ||
                 op->src[0]->type == GGML_TYPE_BF16 ||
                 op->src[0]->type == GGML_TYPE_Q8_0 ||
                 op->src[0]->type == GGML_TYPE_Q4_K ||
                 op->src[0]->type == GGML_TYPE_Q5_K ||
                 op->src[0]->type == GGML_TYPE_Q6_K ||
                 op->src[0]->type == GGML_TYPE_Q2_K ||
                 op->src[0]->type == GGML_TYPE_Q3_K ||
                 op->src[0]->type == GGML_TYPE_MXFP4 ||
                 op->src[0]->type == GGML_TYPE_IQ2_XXS ||
                 op->src[0]->type == GGML_TYPE_IQ2_XS ||
                 op->src[0]->type == GGML_TYPE_IQ3_XXS ||
                 op->src[0]->type == GGML_TYPE_IQ3_S ||
                 op->src[0]->type == GGML_TYPE_IQ2_S ||
                 op->src[0]->type == GGML_TYPE_IQ1_S ||
                 op->src[0]->type == GGML_TYPE_IQ1_M ||
                 op->src[0]->type == GGML_TYPE_IQ4_NL ||
                 op->src[0]->type == GGML_TYPE_IQ4_XS ||
                 op->src[0]->type == GGML_TYPE_Q4_0 ||
                 op->src[0]->type == GGML_TYPE_Q4_1 ||
                 op->src[0]->type == GGML_TYPE_Q5_0 ||
                 op->src[0]->type == GGML_TYPE_Q5_1 ||
                 op->src[0]->type == GGML_TYPE_Q1_0 ||
                 op->src[0]->type == GGML_TYPE_Q2_0 ||
                 op->src[0]->type == GGML_TYPE_PQ2_0 ||
                 op->src[0]->type == GGML_TYPE_PTQ1_0);
            return prefill || decode;
        }
        case GGML_OP_MUL_MAT_ID: {
            if (op->src[0]->type == GGML_TYPE_NVFP4 || ggml_metal_type_is_ternary(op->src[0]->type)) {
                return false;
            }
            if (has_simdgroup_reduction) {
                return ggml_metal_supports_mul_mat_op(
                        has_simdgroup_reduction, op, false,
                        ggml_metal_op_mul_mat_id_use_mm(op, has_simdgroup_mm));
            }
            // wave64: mul_mv_id reuses the width-parameterized decode impls and the
            // manual mm_id prefill uses virtual 32-wide indices; both run on the GPU.
            return wave64_decode && op->src[0]->ne[0] >= 32 &&
                (op->src[0]->type == GGML_TYPE_F16 ||
                 op->src[0]->type == GGML_TYPE_F32 ||
                 op->src[0]->type == GGML_TYPE_BF16 ||
                 op->src[0]->type == GGML_TYPE_Q8_0 ||
                 op->src[0]->type == GGML_TYPE_Q4_K ||
                 op->src[0]->type == GGML_TYPE_Q5_K ||
                 op->src[0]->type == GGML_TYPE_Q6_K ||
                 op->src[0]->type == GGML_TYPE_Q2_K ||
                 op->src[0]->type == GGML_TYPE_Q3_K ||
                 op->src[0]->type == GGML_TYPE_MXFP4 ||
                 op->src[0]->type == GGML_TYPE_IQ2_XXS ||
                 op->src[0]->type == GGML_TYPE_IQ2_XS ||
                 op->src[0]->type == GGML_TYPE_IQ3_XXS ||
                 op->src[0]->type == GGML_TYPE_IQ3_S ||
                 op->src[0]->type == GGML_TYPE_IQ2_S ||
                 op->src[0]->type == GGML_TYPE_IQ1_S ||
                 op->src[0]->type == GGML_TYPE_IQ1_M ||
                 op->src[0]->type == GGML_TYPE_IQ4_NL ||
                 op->src[0]->type == GGML_TYPE_IQ4_XS ||
                 op->src[0]->type == GGML_TYPE_Q4_0 ||
                 op->src[0]->type == GGML_TYPE_Q4_1 ||
                 op->src[0]->type == GGML_TYPE_Q5_0 ||
                 op->src[0]->type == GGML_TYPE_Q5_1 ||
                 op->src[0]->type == GGML_TYPE_Q1_0 ||
                 op->src[0]->type == GGML_TYPE_Q2_0 ||
                 op->src[0]->type == GGML_TYPE_PQ2_0 ||
                 op->src[0]->type == GGML_TYPE_PTQ1_0);
        }
        case GGML_OP_SET:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        switch (op->type) {
                           case GGML_TYPE_F32:
                           case GGML_TYPE_F16:
                           case GGML_TYPE_BF16:
                           case GGML_TYPE_Q8_0:
                           case GGML_TYPE_Q1_0:
                           case GGML_TYPE_Q2_0:
                           case GGML_TYPE_PQ2_0:
                           case GGML_TYPE_Q4_0:
                           case GGML_TYPE_Q4_1:
                           case GGML_TYPE_Q5_0:
                           case GGML_TYPE_Q5_1:
                           case GGML_TYPE_IQ4_NL:
                           case GGML_TYPE_TQ2_0:
                           case GGML_TYPE_TURBO2_0:
                           case GGML_TYPE_TURBO3_0:
                           case GGML_TYPE_TURBO4_0:
                           case GGML_TYPE_I32:
                                return true;
                           default:
                                return false;
                        }
                    case GGML_TYPE_F16:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_BF16:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_BF16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_PQ2_0:
                    case GGML_TYPE_PTQ1_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_TQ2_0:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_TURBO2_0:
                    case GGML_TYPE_TURBO3_0:
                    case GGML_TYPE_TURBO4_0:
                        // dequant-only (inverse WHT to f32); no turbo->f16 kernel
                        return op->type == GGML_TYPE_F32;
                    case GGML_TYPE_I32:
                        return op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32;
                    default:
                        return false;
                };
            }
        case GGML_OP_GET_ROWS:
            return op->src[0]->type != GGML_TYPE_NVFP4 &&
                   op->src[0]->type != GGML_TYPE_TQ1_0 &&
                   !ggml_metal_type_is_ternary(op->src[0]->type);
        case GGML_OP_SET_ROWS:
            {
                if (op->src[0]->type == GGML_TYPE_F16) {
                    return op->type == GGML_TYPE_F16;
                }

                if (op->src[0]->type != GGML_TYPE_F32) {
                    return false;
                }

                switch (op->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_TQ2_0:
                    case GGML_TYPE_TURBO2_0:
                    case GGML_TYPE_TURBO3_0:
                    case GGML_TYPE_TURBO4_0:
                        return true;
                    default:
                        return false;
                };
            }
        case GGML_OP_DIAG:
            return true;
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            return has_simdgroup_reduction;
        default:
            return false;
    }
}

const struct ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev) {
    return &dev->props;
}

static void ggml_metal_device_disable_tensor(ggml_metal_device_t dev) {
    dev->props.has_tensor = false;
}

struct ggml_metal_fusion_info * ggml_metal_device_get_fusion_info(ggml_metal_device_t dev) {
    return dev->finfo;
}

//
// device buffers
//

// max memory buffers that can be mapped to the device
#define GGML_METAL_MAX_BUFFERS 64

struct ggml_metal_buffer_wrapper {
    void   * data;
    size_t   size;

    id<MTLBuffer> metal;
};

struct ggml_metal_buffer {
    void * all_data;
    size_t all_size;

    // if false, the Metal buffer data is allocated in private GPU memory and is not shared with the host
    bool is_shared;
    bool owned;

    // multiple buffers are used only to avoid the maximum buffer size limitation when using mmap
    int n_buffers;
    struct ggml_metal_buffer_wrapper buffers[GGML_METAL_MAX_BUFFERS];

    bool use_residency_sets;

    // optional MTLResidencySet
    // note: cannot use explicitly "id<MTLResidencySet>" here because it is not available on certain OSes
    id rset;

    // pointers to global device
    ggml_metal_device_t dev;
};

static void ggml_metal_log_allocated_size(id<MTLDevice> device, size_t size_aligned) {
#ifndef GGML_METAL_NDEBUG
#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
    if (@available(macOS 10.12, iOS 16.0, *)) {
        GGML_LOG_DEBUG("%s: allocated buffer, size = %8.2f MiB, (%8.2f / %8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0,
                device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);

        if (device.currentAllocatedSize > device.recommendedMaxWorkingSetSize) {
            GGML_LOG_WARN("%s: warning: current allocated size is greater than the recommended max working set size\n", __func__);
        }
    } else {
        GGML_LOG_INFO("%s: allocated buffer, size = %8.2f MiB, (%8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0);
    }
#endif
#endif
    GGML_UNUSED(device);
    GGML_UNUSED(size_aligned);
}

// rset init
static bool ggml_metal_buffer_rset_init(ggml_metal_buffer_t buf) {
    buf->rset = nil;

    if (!buf->use_residency_sets) {
        return true;
    }

#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        MTLResidencySetDescriptor * desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"ggml_metal";
        desc.initialCapacity = buf->n_buffers;

        NSError * error;
        buf->rset = [buf->dev->mtl_device newResidencySetWithDescriptor:desc error:&error];
        if (error) {
            GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
            [desc release];
            return false;
        }

        [desc release];

        for (int i = 0; i < buf->n_buffers; i++) {
            [buf->rset addAllocation:buf->buffers[i].metal];
        }

        [buf->rset commit];
        [buf->rset requestResidency];

        return true;
    }
#endif

    return true;
}

// rset free
static void ggml_metal_buffer_rset_free(ggml_metal_buffer_t buf) {
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        if (buf->rset) {
            [buf->rset endResidency];
            [buf->rset removeAllAllocations];
            [buf->rset commit];
            [buf->rset release];
        }
    }
#else
    GGML_UNUSED(buf);
#endif
}

static void * ggml_metal_host_malloc(size_t n) {
    void * data = NULL;

#if TARGET_OS_OSX
    kern_return_t err = vm_allocate((vm_map_t) mach_task_self(), (void *) &data, n, VM_FLAGS_ANYWHERE);
    if (err != KERN_SUCCESS) {
        GGML_LOG_ERROR("%s: error: vm_allocate failed\n", __func__);
        return NULL;
    }
#else
    const int result = posix_memalign((void **) &data, sysconf(_SC_PAGESIZE), n);
    if (result != 0) {
        GGML_LOG_ERROR("%s: error: posix_memalign failed\n", __func__);
        return NULL;
    }
#endif

    return data;
}

// release what a half-built buffer already owns, without touching the residency set
// private (VRAM) bytes held by ggml buffers of this process, for sizing the expert arena
static _Atomic size_t g_metal_private_bytes;
size_t ggml_metal_private_bytes(void) { return atomic_load(&g_metal_private_bytes); }

static void ggml_metal_buffer_free_owned(ggml_metal_buffer_t buf) {
    for (int i = 0; i < buf->n_buffers; i++) {
        if (!buf->is_shared && buf->buffers[i].metal != nil) {
            atomic_fetch_sub(&g_metal_private_bytes, (size_t) [buf->buffers[i].metal length]);
        }
        [buf->buffers[i].metal release];
    }

    if (buf->is_shared && buf->owned && buf->all_data != NULL) {
#if TARGET_OS_OSX
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)buf->all_data, buf->all_size);
#else
        free(buf->all_data);
#endif
    }
}

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t dev, size_t size, bool shared) {
    ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));

    res->dev = dev;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    shared = shared && props_dev->use_shared_buffers;

    // allocate shared buffer if the device supports it and it is required by the buffer type
    if (shared) {
        res->all_data = ggml_metal_host_malloc(size_aligned);
        res->is_shared = true;
    } else {
        // use virtual address
        res->all_data = (void *) atomic_fetch_add_explicit(&dev->addr_virt, size_aligned, memory_order_relaxed);
        res->is_shared = false;
    }
    res->all_size = size_aligned;

    res->owned = true;

    res->n_buffers = 1;

    if (res->all_data != NULL) {
        res->buffers[0].size  = size;
        res->buffers[0].metal = nil;

        if (size_aligned > 0) {
            // `shared` already carries the device's answer plus the host-buffer opt-in, and
            // taking the device flag again here wrapped host memory in a private buffer: the
            // writes landed in one place and the GPU read the other
            if (shared) {
                res->buffers[0].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:res->all_data
                                                                  length:size_aligned
                                                                 options:MTLResourceStorageModeShared
                                                             deallocator:nil];
            } else {
                res->buffers[0].metal = [res->dev->mtl_device newBufferWithLength:size_aligned options:MTLResourceStorageModePrivate];
                if (res->buffers[0].metal != nil) {
                    atomic_fetch_add(&g_metal_private_bytes, size_aligned);
                }
            }
        }

        res->buffers[0].data = res->all_data;
    }

    // the host allocation outlives res on these paths unless it is handed back
    if (size_aligned > 0 && (res->all_data == NULL || res->buffers[0].metal == nil)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
        ggml_metal_buffer_free_owned(res);
        free(res);
        return NULL;
    }

    res->use_residency_sets = props_dev->use_residency_sets;

    if (!ggml_metal_buffer_rset_init(res)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        ggml_metal_buffer_free_owned(res);
        free(res);
        return NULL;
    }

    ggml_metal_device_rsets_add(dev, res->rset);

    //ggml_metal_log_allocated_size(device, size_aligned);

    return res;
}

ggml_metal_buffer_t ggml_metal_buffer_map(ggml_metal_device_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));

    res->dev = dev;

    res->all_data = ptr;
    res->all_size = size;

    res->is_shared = true;
    res->owned = false;

    res->n_buffers = 0;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    // page-align the data ptr
    {
        const uintptr_t offs = (uintptr_t) ptr % size_page;
        ptr  = (void *) ((char *) ptr - offs);
        size += offs;
    }

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    // the buffer fits into the max buffer size allowed by the device
    if (size_aligned <= props_dev->max_buffer_size) {
        res->buffers[res->n_buffers].data  = ptr;
        res->buffers[res->n_buffers].size  = size;
        res->buffers[res->n_buffers].metal = nil;

        if (size_aligned > 0) {
            res->buffers[res->n_buffers].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:ptr length:size_aligned options:MTLResourceStorageModeShared deallocator:nil];

            if (res->buffers[res->n_buffers].metal == nil) {
                GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
                ggml_metal_buffer_free_owned(res);
                free(res);
                return NULL;
            }
        }

        ggml_metal_log_allocated_size(res->dev->mtl_device, size_aligned);

        ++res->n_buffers;
    } else {
        // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
        // one of the views
        const size_t size_ovlp = ((max_tensor_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
        const size_t size_step = props_dev->max_buffer_size - size_ovlp;
        const size_t size_view = props_dev->max_buffer_size;

        for (size_t i = 0; i < size; i += size_step) {
            const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

            res->buffers[res->n_buffers].data  = (void *) ((uint8_t *) ptr + i);
            res->buffers[res->n_buffers].size  = size_step_aligned;
            res->buffers[res->n_buffers].metal = nil;

            if (size_step_aligned > 0) {
                res->buffers[res->n_buffers].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:(void *) ((uint8_t *) ptr + i) length:size_step_aligned options:MTLResourceStorageModeShared deallocator:nil];

                if (res->buffers[res->n_buffers].metal == nil) {
                    GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_step_aligned / 1024.0 / 1024.0);
                    ggml_metal_buffer_free_owned(res);
                    free(res);
                    return NULL;
                }
            }

            ggml_metal_log_allocated_size(res->dev->mtl_device, size_step_aligned);

            if (i + size_step < size) {
                GGML_LOG_INFO("\n");
            }

            ++res->n_buffers;
        }
    }

    res->use_residency_sets = props_dev->use_residency_sets;

    if (!ggml_metal_buffer_rset_init(res)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        ggml_metal_buffer_free_owned(res);
        free(res);
        return NULL;
    }

    ggml_metal_device_rsets_add(dev, res->rset);

    return res;
}

void ggml_metal_buffer_free(ggml_metal_buffer_t buf) {
    @autoreleasepool {
        ggml_metal_device_rsets_rm(buf->dev, buf->rset);

        for (int i = 0; i < buf->n_buffers; i++) {
            if (!buf->is_shared && buf->owned && buf->buffers[i].metal != nil) {
                atomic_fetch_sub(&g_metal_private_bytes, (size_t) [buf->buffers[i].metal length]);
            }
            [buf->buffers[i].metal release];
        }

        ggml_metal_buffer_rset_free(buf);
    }

    if (buf->is_shared && buf->owned) {
#if TARGET_OS_OSX
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)buf->all_data, buf->all_size);
#else
        free(buf->all_data);
#endif
    }

    free(buf);
}

void * ggml_metal_buffer_get_base(ggml_metal_buffer_t buf) {
    return buf->all_data;
}

bool ggml_metal_buffer_is_shared(ggml_metal_buffer_t buf) {
    return buf->is_shared;
}

void ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    if (buf->is_shared) {
        memset((char *) tensor->data + offset, value, size);
        return;
    }

    @autoreleasepool {
        // dst
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
        bid_dst.offs += offset;

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:bid_dst.metal
                          range:NSMakeRange(bid_dst.offs, size)
                          value:value];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

// transfers up to this size reuse one persistent staging buffer; larger one-shot
// transfers (model load, KV persistence) keep the direct no-copy wrap
#define GGML_METAL_STAGE_BUF_MAX ((size_t) 16*1024*1024)

// TOSH_MOE_PROFILE=1: count the synchronous host<->VRAM transfers (the per-token
// traffic of MoE-offload / multi-GPU) and log a summary every ~5 s.
enum { TOSH_PROF_D2H_STAGE, TOSH_PROF_H2D_STAGE, TOSH_PROF_D2H_DIRECT, TOSH_PROF_H2D_DIRECT, TOSH_PROF_N };

static struct {
    _Atomic int      state; // 0 unknown, 1 off, 2 on
    _Atomic uint64_t calls  [TOSH_PROF_N];
    _Atomic uint64_t bytes  [TOSH_PROF_N];
    _Atomic uint64_t wait_ns[TOSH_PROF_N];
    _Atomic uint64_t t_last;
    // big D2H reads (logits class) split into GPU-busy time vs the blit itself
    _Atomic uint64_t big_calls, big_dep_ns, big_xfer_ns;
} g_tosh_prof;

static bool tosh_prof_on(void) {
    int s = atomic_load_explicit(&g_tosh_prof.state, memory_order_relaxed);
    if (s == 0) {
        const char * v = TOSH_ENV("TOSH_MOE_PROFILE");
        s = (v && v[0] == '1') ? 2 : 1;
        atomic_store_explicit(&g_tosh_prof.state, s, memory_order_relaxed);
    }
    return s == 2;
}

// name-class buckets, filled by the callers that know the tensor
enum { TOSH_CLS_MOE, TOSH_CLS_LOGITS, TOSH_CLS_OTHER, TOSH_CLS_N };
static _Atomic uint64_t g_tosh_cls_calls[2][TOSH_CLS_N]; // [d2h][cls]
static _Atomic uint64_t g_tosh_cls_bytes[2][TOSH_CLS_N];

void ggml_metal_prof_note(const char * name, size_t size, bool d2h) {
    if (!tosh_prof_on()) {
        return;
    }
    int cls = TOSH_CLS_OTHER;
    if (name) {
        if (strstr(name, "moe") || strstr(name, "exps")) {
            cls = TOSH_CLS_MOE;
        } else if (strstr(name, "result_output")) {
            cls = TOSH_CLS_LOGITS;
        }
    }
    atomic_fetch_add_explicit(&g_tosh_cls_calls[d2h][cls], 1,    memory_order_relaxed);
    atomic_fetch_add_explicit(&g_tosh_cls_bytes[d2h][cls], size, memory_order_relaxed);

    // first distinct "other" names, to identify unexplained traffic
    if (cls == TOSH_CLS_OTHER && name &&
        (d2h || atomic_load_explicit(&g_tosh_prof.t_last, memory_order_relaxed) != 0)) {
        static char seen[16][48];
        static atomic_flag lock = ATOMIC_FLAG_INIT;
        if (!atomic_flag_test_and_set(&lock)) {
            for (int i = 0; i < 16; i++) {
                if (seen[i][0] == '\0') {
                    snprintf(seen[i], sizeof(seen[i]), "%s", name);
                    fprintf(stderr, "ggml_metal: moe-profile: other[%s] %s %zuB\n",
                            d2h ? "D2H" : "H2D", name, size);
                    break;
                }
                if (strcmp(seen[i], name) == 0) {
                    break;
                }
            }
            atomic_flag_clear(&lock);
        }
    }
}

static void tosh_prof_cls_flush(void) {
    static const char * dirs[2] = { "H2D", "D2H" };
    static const char * clss[TOSH_CLS_N] = { "moe", "logits", "other" };
    char line[256];
    int n = 0;
    for (int d = 0; d < 2; d++) {
        for (int c = 0; c < TOSH_CLS_N; c++) {
            const uint64_t calls = atomic_exchange(&g_tosh_cls_calls[d][c], 0);
            const uint64_t bytes = atomic_exchange(&g_tosh_cls_bytes[d][c], 0);
            if (calls == 0) {
                continue;
            }
            n += snprintf(line + n, sizeof(line) - n, "%s%s-%s %llux %.1fMB",
                          n > 0 ? " | " : "", dirs[d], clss[c],
                          (unsigned long long) calls, bytes / 1e6);
        }
    }
    if (n > 0) {
        fprintf(stderr, "ggml_metal: moe-profile:   %s\n", line);
    }
}

static void tosh_prof_add(int idx, size_t size, uint64_t t0) {
    const uint64_t now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    atomic_fetch_add_explicit(&g_tosh_prof.calls[idx],   1,        memory_order_relaxed);
    atomic_fetch_add_explicit(&g_tosh_prof.bytes[idx],   size,     memory_order_relaxed);
    atomic_fetch_add_explicit(&g_tosh_prof.wait_ns[idx], now - t0, memory_order_relaxed);

    uint64_t last = atomic_load_explicit(&g_tosh_prof.t_last, memory_order_relaxed);
    if (now - last < 5000000000ULL ||
        !atomic_compare_exchange_strong(&g_tosh_prof.t_last, &last, now)) {
        return;
    }
    static const char * names[TOSH_PROF_N] = { "D2H-stage", "H2D-stage", "D2H-direct", "H2D-direct" };
    char line[256];
    int n = 0;
    for (int i = 0; i < TOSH_PROF_N; i++) {
        const uint64_t c = atomic_exchange(&g_tosh_prof.calls[i],   0);
        const uint64_t b = atomic_exchange(&g_tosh_prof.bytes[i],   0);
        const uint64_t w = atomic_exchange(&g_tosh_prof.wait_ns[i], 0);
        if (c == 0) {
            continue;
        }
        n += snprintf(line + n, sizeof(line) - n, "%s%s %llux %.1fMB %.0fms",
                      n > 0 ? " | " : "", names[i],
                      (unsigned long long) c, b / 1e6, w / 1e6);
    }
    if (n > 0) {
        // raw stderr: llama-server filters backend INFO logs at default verbosity
        fprintf(stderr, "ggml_metal: moe-profile: %s\n", line);
    }
    const uint64_t bc = atomic_exchange(&g_tosh_prof.big_calls, 0);
    if (bc > 0) {
        const uint64_t bd = atomic_exchange(&g_tosh_prof.big_dep_ns,  0);
        const uint64_t bx = atomic_exchange(&g_tosh_prof.big_xfer_ns, 0);
        fprintf(stderr, "ggml_metal: moe-profile:   D2H-big %llux gpu-busy %.0fms blit %.1fms\n",
                (unsigned long long) bc, bd / 1e6, bx / 1e6);
    }
    tosh_prof_cls_flush();
}

// caller must hold dev->stage_lock; nil on allocation failure
static id<MTLBuffer> ggml_metal_device_stage_buf(ggml_metal_device_t dev) {
    if (dev->stage_buf == nil) {
        dev->stage_buf = [dev->mtl_device newBufferWithLength:GGML_METAL_STAGE_BUF_MAX
                                                      options:MTLResourceStorageModeShared];
    }
    return dev->stage_buf;
}

// waits out and releases every in-flight upload; caller must hold stage_lock
static void ggml_metal_device_stage_set_drain(ggml_metal_device_t dev) {
    for (int i = 0; i < 4; i++) {
        if (dev->stage_set_cmds[i]) {
            [dev->stage_set_cmds[i] waitUntilCompleted];
            [dev->stage_set_cmds[i] release];
            dev->stage_set_cmds[i] = nil;
        }
    }
}

void ggml_metal_device_upload_drain(ggml_metal_device_t dev) {
    [dev->stage_lock lock];
    ggml_metal_device_stage_set_drain(dev);
    [dev->stage_lock unlock];
}

bool ggml_metal_device_stage_set(ggml_metal_device_t dev, struct ggml_metal_buffer_id bid_dst, const void * data, size_t size) {
    if (size == 0) {
        return true;
    }
    if (size > GGML_METAL_STAGE_BUF_MAX || bid_dst.metal == nil) {
        return false;
    }

    const uint64_t t0 = tosh_prof_on() ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;

    [dev->stage_lock lock];

    const int slot = dev->stage_set_cur;
    dev->stage_set_cur = (slot + 1) % 4;

    if (dev->stage_set_cmds[slot]) {
        [dev->stage_set_cmds[slot] waitUntilCompleted];
        [dev->stage_set_cmds[slot] release];
        dev->stage_set_cmds[slot] = nil;
    }

    id<MTLBuffer> stage = dev->stage_set_bufs[slot];
    if (stage == nil || stage.length < size) {
        [stage release];
        static int wc_off = -1;
        if (wc_off < 0) { wc_off = TOSH_ENV("TOSH_STAGE_WC_OFF") != NULL; }
        stage = [dev->mtl_device newBufferWithLength:MAX(size, (size_t) 256*1024)
                                             options:wc_off ? MTLResourceStorageModeShared
                                                            : (MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined)];
        dev->stage_set_bufs[slot] = stage;
    }
    if (stage == nil) {
        [dev->stage_lock unlock];
        return false;
    }

    memcpy(stage.contents, data, size);

    id<MTLCommandBuffer> cmd_buf = [dev->mtl_queue commandBuffer];

    {
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:stage
                   sourceOffset:0
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:size];

        [encoder endEncoding];
    }

    [cmd_buf commit];
    // no wait: every consumer is queued behind the blit on the same queue
    dev->stage_set_cmds[slot] = [cmd_buf retain];

    [dev->stage_lock unlock];

    if (t0) {
        tosh_prof_add(TOSH_PROF_H2D_STAGE, size, t0);
    }

    return true;
}

// A tensor-split weight upload arrives one row-slice at a time. Routing each through
// ggml_metal_device_stage_set costs a command buffer per row; pack as many rows as the
// staging buffer holds into a single one instead.
bool ggml_metal_device_stage_set_2d(ggml_metal_device_t dev, struct ggml_metal_buffer_id bid_dst,
        const void * data, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    if (size == 0 || n_copies == 0) {
        return true;
    }
    if (size > GGML_METAL_STAGE_BUF_MAX || bid_dst.metal == nil) {
        return false;
    }

    const uint64_t t0 = tosh_prof_on() ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;

    const size_t per_batch = MAX((size_t) 1, GGML_METAL_STAGE_BUF_MAX / size);

    [dev->stage_lock lock];

    for (size_t i = 0; i < n_copies;) {
        const size_t k    = MIN(per_batch, n_copies - i);
        const size_t need = k*size;

        const int slot = dev->stage_set_cur;
        dev->stage_set_cur = (slot + 1) % 4;

        if (dev->stage_set_cmds[slot]) {
            [dev->stage_set_cmds[slot] waitUntilCompleted];
            [dev->stage_set_cmds[slot] release];
            dev->stage_set_cmds[slot] = nil;
        }

        id<MTLBuffer> stage = dev->stage_set_bufs[slot];
        if (stage == nil || stage.length < need) {
            [stage release];
            stage = [dev->mtl_device newBufferWithLength:MAX(need, (size_t) 256*1024)
                                                 options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
            dev->stage_set_bufs[slot] = stage;
        }
        if (stage == nil) {
            [dev->stage_lock unlock];
            return false;
        }

        for (size_t r = 0; r < k; r++) {
            memcpy((char *) stage.contents + r*size, (const char *) data + (i + r)*stride_data, size);
        }

        id<MTLCommandBuffer> cmd_buf = [dev->mtl_queue commandBuffer];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            for (size_t r = 0; r < k; r++) {
                [encoder copyFromBuffer:stage
                           sourceOffset:r*size
                               toBuffer:bid_dst.metal
                      destinationOffset:bid_dst.offs + (i + r)*stride_tensor
                                   size:size];
            }

            [encoder endEncoding];
        }

        [cmd_buf commit];
        dev->stage_set_cmds[slot] = [cmd_buf retain];

        i += k;
    }

    [dev->stage_lock unlock];

    if (t0) {
        tosh_prof_add(TOSH_PROF_H2D_STAGE, size*n_copies, t0);
    }

    return true;
}

// one commit + wait covers every queued blit; caller must hold stage_lock
static void ggml_metal_device_stage_batch_flush(ggml_metal_device_t dev) {
    if (dev->stage_batch_n == 0) {
        return;
    }
    [dev->stage_batch_enc endEncoding];
    [dev->stage_batch_cmd commit];
    [dev->stage_batch_cmd waitUntilCompleted];
    for (int i = 0; i < dev->stage_batch_n; i++) {
        memcpy(dev->stage_batch_items[i].dst,
               (char *) dev->stage_buf.contents + dev->stage_batch_items[i].off,
               dev->stage_batch_items[i].size);
    }
    [dev->stage_batch_enc release];
    [dev->stage_batch_cmd release];
    dev->stage_batch_n    = 0;
    dev->stage_batch_used = 0;
    dev->stage_batch_enc  = nil;
    dev->stage_batch_cmd  = nil;
}

void ggml_metal_device_read_batch_begin(ggml_metal_device_t dev) {
    // shared-buffer devices never take the stage path; skip the empty batch
    if (dev->props.use_shared_buffers) {
        return;
    }
    [dev->stage_lock lock];   // held until read_batch_end
    // no command buffer yet: one left uncommitted by an empty batch keeps its slot in the queue
    dev->stage_batch_open = true;
}

void ggml_metal_device_read_batch_end(ggml_metal_device_t dev) {
    if (dev->props.use_shared_buffers) {
        return;
    }
    ggml_metal_device_stage_batch_flush(dev);
    dev->stage_batch_open = false;
    [dev->stage_lock unlock];
}

static void ggml_metal_device_stage_batch_flush(ggml_metal_device_t dev);

// a read batch drains every few MiB: larger batches wait less on a desktop host but copy out
// of the staging buffer slower on a Xeon W one, and 4 MiB was the size that won on both
static size_t ggml_metal_stage_batch_cap(void) {
    return (size_t) 4*1024*1024;
}

bool ggml_metal_device_stage_get(ggml_metal_device_t dev, struct ggml_metal_buffer_id bid_src, void * data, size_t size) {
    if (size == 0) {
        return true;
    }
    if (size > GGML_METAL_STAGE_BUF_MAX || bid_src.metal == nil) {
        return false;
    }

    const uint64_t t0 = tosh_prof_on() ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;

    [dev->stage_lock lock];

    // inside an open read batch: queue the blit, defer the wait to batch end
    if (dev->stage_batch_open) {
        if (ggml_metal_device_stage_buf(dev) == nil) {
            [dev->stage_lock unlock];
            return false;   // no buffer: caller falls back to the direct path
        }
        // a full batch drains and reopens, so a long run of reads still pays one wait per
        // staging buffer instead of one per read
        if (dev->stage_batch_n == 16 || dev->stage_batch_used + size > ggml_metal_stage_batch_cap()) {
            ggml_metal_device_stage_batch_flush(dev);
        }
        if (dev->stage_batch_cmd == nil) {
            // the caller sits in an autorelease pool that drains before the batch ends
            dev->stage_batch_cmd = [[dev->mtl_queue commandBufferWithUnretainedReferences] retain];
            dev->stage_batch_enc = [[dev->stage_batch_cmd blitCommandEncoder] retain];
        }
        [dev->stage_batch_enc copyFromBuffer:bid_src.metal
                                sourceOffset:bid_src.offs
                                    toBuffer:dev->stage_buf
                           destinationOffset:dev->stage_batch_used
                                        size:size];
        dev->stage_batch_items[dev->stage_batch_n++] =
            (typeof(dev->stage_batch_items[0])) { data, dev->stage_batch_used, size };
        dev->stage_batch_used += size;
        [dev->stage_lock unlock];
        if (t0) {
            tosh_prof_add(TOSH_PROF_D2H_STAGE, size, t0);
        }
        return true;
    }

    id<MTLBuffer> stage = ggml_metal_device_stage_buf(dev);
    if (stage == nil) {
        [dev->stage_lock unlock];
        return false;
    }

    id<MTLCommandBuffer> cmd_buf = [dev->mtl_queue commandBufferWithUnretainedReferences];

    {
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:stage
              destinationOffset:0
                           size:size];

        [encoder endEncoding];
    }

    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];

    // split the big-read wait: time before the blit ran is GPU compute, not readback
    if (t0 && size >= (size_t) 256*1024) {
        atomic_fetch_add_explicit(&g_tosh_prof.big_calls, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_tosh_prof.big_dep_ns,
            (uint64_t) ((cmd_buf.GPUStartTime - cmd_buf.kernelStartTime) * 1e9), memory_order_relaxed);
        atomic_fetch_add_explicit(&g_tosh_prof.big_xfer_ns,
            (uint64_t) ((cmd_buf.GPUEndTime - cmd_buf.GPUStartTime) * 1e9), memory_order_relaxed);
    }

    memcpy(data, stage.contents, size);

    [dev->stage_lock unlock];

    if (t0) {
        tosh_prof_add(TOSH_PROF_D2H_STAGE, size, t0);
    }

    return true;
}

_Atomic uint64_t g_tosh_d2h_ns, g_tosh_d2h_bytes, g_tosh_d2h_n;
_Atomic uint64_t g_tosh_h2d_ns, g_tosh_h2d_bytes, g_tosh_h2d_n;
static _Atomic int g_tosh_xfer_state = 0;

static bool tosh_xfer_on(void);
static void ggml_metal_buffer_set_tensor_impl(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
static void ggml_metal_buffer_get_tensor_impl(ggml_metal_buffer_t buf, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);

void ggml_metal_buffer_set_tensor_2d(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
        size_t n_copies, size_t stride_tensor, size_t stride_data) {
    if (size == 0 || n_copies == 0) {
        return;
    }

    if (!buf->is_shared) {
        @autoreleasepool {
            ggml_metal_prof_note(tensor->name, size*n_copies, false);

            struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
            bid_dst.offs += offset;

            if (ggml_metal_device_stage_set_2d(buf->dev, bid_dst, data, size, n_copies, stride_tensor, stride_data)) {
                return;
            }
        }
    }

    for (size_t i = 0; i < n_copies; i++) {
        ggml_metal_buffer_set_tensor(buf, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
    }
}

void ggml_metal_buffer_set_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const bool xf = tosh_xfer_on();
    const uint64_t xf_t0 = xf ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;
    if (xf) {
        atomic_fetch_add_explicit(&g_tosh_h2d_n, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_tosh_h2d_bytes, size, memory_order_relaxed);
    }
    ggml_metal_buffer_set_tensor_impl(buf, tensor, data, offset, size);
    if (xf) {
        atomic_fetch_add_explicit(&g_tosh_h2d_ns,
                clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - xf_t0, memory_order_relaxed);
    }
}

static void ggml_metal_buffer_set_tensor_impl(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (size == 0) {
        return;
    }

    if (buf->is_shared) {
        memcpy((char *) tensor->data + offset, data, size);
        return;
    }

    @autoreleasepool {
        ggml_metal_prof_note(tensor->name, size, false);
        // Small transfers (the per-token copies of MoE-offload and multi-GPU
        // runs) go through the reused staging buffer: wrapping the caller's
        // pointer below allocates a kernel resource per call, and on AMD those
        // accumulate across a long generation until transfers crawl.
        {
            struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
            bid_dst.offs += offset;

            if (ggml_metal_device_stage_set(buf->dev, bid_dst, data, size)) {
                return;
            }

            // A large transfer wrapped whole registers the caller's pages with the device for the
            // whole blit; feeding it through the ring in slices keeps the copy pipelined instead.
            // Only measured with one device so far, so several cards keep the wrap until it is.
            static int stage_load = -1;
            if (stage_load < 0) {
                if (TOSH_ENV("TOSH_STAGE_LOAD_DISABLE") != NULL) {
                    stage_load = 0;
                } else if (TOSH_ENV("TOSH_STAGE_LOAD") != NULL) {
                    stage_load = 1;
                } else {
                    const char * n = TOSH_ENV("GGML_METAL_DEVICES");
                    stage_load = (n == NULL || atoi(n) <= 1) && TOSH_ENV("GGML_METAL_DEVICE_LIST") == NULL;
                }
            }
            if (stage_load) {
                bool done_all = true;
                for (size_t done = 0; done < size;) {
                    static size_t slice = 0;
                    if (slice == 0) {
                        const char * v = TOSH_ENV("TOSH_STAGE_SLICE_MB");
                        slice = v ? MIN((size_t) atoi(v)*1024*1024, GGML_METAL_STAGE_BUF_MAX) : GGML_METAL_STAGE_BUF_MAX;
                        if (slice == 0) { slice = GGML_METAL_STAGE_BUF_MAX; }
                    }
                    const size_t n = MIN(slice, size - done);

                    struct ggml_metal_buffer_id bid = bid_dst;
                    bid.offs += done;

                    if (!ggml_metal_device_stage_set(buf->dev, bid, (const char *) data + done, n)) {
                        done_all = false;
                        break;
                    }

                    done += n;
                }
                if (done_all) {
                    return;
                }
            }
        }

        const uint64_t t0 = tosh_prof_on() ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;

        // src
        void * data_ptr = (void *)(uintptr_t) data; // "const cast" the src data
        // wrapping the caller's pointer registers the whole tensor as a Metal resource for the
        // blit; TOSH_NO_HOST_WRAP skips it so the chunked staging path below is used instead
        static int no_wrap = -1;
        if (no_wrap < 0) { no_wrap = TOSH_ENV("TOSH_NO_HOST_WRAP") != NULL; }
        id<MTLBuffer> buf_src = no_wrap ? nil : [buf->dev->mtl_device newBufferWithBytesNoCopy:data_ptr
                                                               length:size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
        // dst
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
        bid_dst.offs += offset;

        if (buf_src == nil) {
            // newBufferWithBytesNoCopy requires page-aligned data, and some drivers cap the size of
            // host-visible allocations; copy through a small staging buffer in chunks instead
            size_t stage_size = MIN(size, (size_t) 8*1024*1024);
            id<MTLBuffer> buf_stage = nil;
            while (stage_size > 0) {
                buf_stage = [buf->dev->mtl_device newBufferWithLength:stage_size options:MTLResourceStorageModeShared];
                if (buf_stage != nil) {
                    break;
                }
                stage_size /= 2;
            }
            if (buf_stage == nil) {
                GGML_LOG_ERROR("%s: failed to allocate staging buffer, size = %zu\n", __func__, size);
            }
            GGML_ASSERT(buf_stage);

            for (size_t done = 0; done < size;) {
                const size_t n = MIN(stage_size, size - done);

                memcpy(buf_stage.contents, (const char *) data + done, n);

                id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

                {
                    id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

                    [encoder copyFromBuffer:buf_stage
                               sourceOffset:0
                                   toBuffer:bid_dst.metal
                          destinationOffset:bid_dst.offs + done
                                       size:n];

                    [encoder endEncoding];
                }

                [cmd_buf commit];
                [cmd_buf waitUntilCompleted];

                done += n;
            }

            if (t0) {
                tosh_prof_add(TOSH_PROF_H2D_DIRECT, size, t0);
            }

            return;
        }

        // note: for experimentation purposes, here we use a semaphore to wait for the copy to complete
        //       this is alternative to waitUntilCompleted, which should be faster, but don't seem to make much difference
        dispatch_semaphore_t completion_semaphore = dispatch_semaphore_create(0);

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:buf_src
                       sourceOffset:0
                           toBuffer:bid_dst.metal
                  destinationOffset:bid_dst.offs
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                             // TODO: can check for errors here
            GGML_UNUSED(cb);

            dispatch_semaphore_signal(completion_semaphore);
        }];

        [cmd_buf commit];

        dispatch_semaphore_wait(completion_semaphore, DISPATCH_TIME_FOREVER);
        dispatch_release(completion_semaphore);
        // the wrap is an owned object here, and keeping it leaves the host pages registered
        // with the device for the rest of the load
        [buf_src release];

        //[cmd_buf waitUntilCompleted];

        if (t0) {
            tosh_prof_add(TOSH_PROF_H2D_DIRECT, size, t0);
        }
    }
}

// TOSH_CB_PROFILE: coste de las lecturas de vuelta (logits) y de las escrituras, que
// son lo que queda del hueco de decode una vez descartados commit, creacion y espera
static bool tosh_xfer_on(void) {
    int st = atomic_load_explicit(&g_tosh_xfer_state, memory_order_relaxed);
    if (st == 0) {
        const char * e = TOSH_ENV("TOSH_CB_PROFILE");
        st = (e != NULL && strcmp(e, "0") != 0) ? 2 : 1;
        atomic_store_explicit(&g_tosh_xfer_state, st, memory_order_relaxed);
    }
    return st == 2;
}

void ggml_metal_buffer_get_tensor(ggml_metal_buffer_t buf, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const bool xf = tosh_xfer_on();
    const uint64_t xf_t0 = xf ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;
    if (xf) {
        atomic_fetch_add_explicit(&g_tosh_d2h_n, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_tosh_d2h_bytes, size, memory_order_relaxed);
    }
    ggml_metal_buffer_get_tensor_impl(buf, tensor, data, offset, size);
    if (xf) {
        atomic_fetch_add_explicit(&g_tosh_d2h_ns,
                clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - xf_t0, memory_order_relaxed);
    }
}

static void ggml_metal_buffer_get_tensor_impl(ggml_metal_buffer_t buf, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (size == 0) {
        return;
    }

    if (buf->is_shared) {
        memcpy(data, (const char *) tensor->data + offset, size);
        return;
    }

    @autoreleasepool {
        ggml_metal_prof_note(tensor->name, size, true);
        // src
        struct ggml_metal_buffer_id bid_src = ggml_metal_buffer_get_id(buf, tensor);
        bid_src.offs += offset;

        // small transfers reuse the persistent staging buffer (see set_tensor)
        if (ggml_metal_device_stage_get(buf->dev, bid_src, data, size)) {
            return;
        }

        const uint64_t t0 = tosh_prof_on() ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;

        // dst
        id<MTLBuffer> buf_dst = [buf->dev->mtl_device newBufferWithBytesNoCopy:data
                                                               length:size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        if (buf_dst != nil) {
            id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

            {
                id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

                [encoder copyFromBuffer:bid_src.metal
                           sourceOffset:bid_src.offs
                               toBuffer:buf_dst
                      destinationOffset:0
                                   size:size];

                [encoder endEncoding];
            }

            [cmd_buf commit];
            [cmd_buf waitUntilCompleted];

            // the wrap is owned here too; the write path already releases its own
            [buf_dst release];

            if (t0) {
                tosh_prof_add(TOSH_PROF_D2H_DIRECT, size, t0);
            }

            return;
        }

        // newBufferWithBytesNoCopy requires page-aligned data, and some drivers cap the size of
        // host-visible allocations; copy through a small staging buffer in chunks instead
        size_t stage_size = MIN(size, (size_t) 8*1024*1024);
        id<MTLBuffer> buf_stage = nil;
        while (stage_size > 0) {
            buf_stage = [buf->dev->mtl_device newBufferWithLength:stage_size options:MTLResourceStorageModeShared];
            if (buf_stage != nil) {
                break;
            }
            stage_size /= 2;
        }
        if (buf_stage == nil) {
            GGML_LOG_ERROR("%s: failed to allocate staging buffer, size = %zu\n", __func__, size);
        }
        GGML_ASSERT(buf_stage);

        for (size_t done = 0; done < size;) {
            const size_t n = MIN(stage_size, size - done);

            id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

            {
                id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

                [encoder copyFromBuffer:bid_src.metal
                           sourceOffset:bid_src.offs + done
                               toBuffer:buf_stage
                      destinationOffset:0
                                   size:n];

                [encoder endEncoding];
            }

            [cmd_buf commit];
            [cmd_buf waitUntilCompleted];

            memcpy((char *) data + done, buf_stage.contents, n);

            done += n;
        }

        if (t0) {
            tosh_prof_add(TOSH_PROF_D2H_DIRECT, size, t0);
        }
    }
}

bool ggml_metal_buffer_cpy_tensor(ggml_metal_buffer_t buf_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_metal_buffer_t buf_src = (ggml_metal_buffer_t)src->buffer->context;

    const size_t size = ggml_nbytes(src);

    // if both buffers are shared, we can use memcpy directly
    if (buf_dst->is_shared && buf_src->is_shared) {
        memcpy(dst->data, src->data, size);
        return true;
    }

    // a blit can't reference a buffer owned by another device
    if (buf_src->dev != buf_dst->dev) {
        return false;
    }

    // for private buffers, we need to use Metal blit commands
    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_buffer_get_id(buf_src, src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf_dst, dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        id<MTLCommandBuffer> cmd_buf = [buf_dst->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:bid_src.metal
                       sourceOffset:bid_src.offs
                           toBuffer:bid_dst.metal
                  destinationOffset:bid_dst.offs
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }

    return true;
}

void ggml_metal_buffer_clear(ggml_metal_buffer_t buf, uint8_t value) {
    if (buf->is_shared) {
        memset(buf->all_data, value, buf->all_size);
        return;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            for (int i = 0; i < buf->n_buffers; ++i) {
                [encoder fillBuffer:buf->buffers[i].metal
                              range:NSMakeRange(0, buf->buffers[i].size)
                              value:value];
            }

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

struct ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t buf, const struct ggml_tensor * t) {
    struct ggml_metal_buffer_id res = { nil, 0 };

    const int64_t tsize = ggml_nbytes(t);

    // find the view that contains the tensor fully
    for (int i = 0; i < buf->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) buf->buffers[i].data;

        //GGML_LOG_INFO("ioffs = %10ld, tsize = %10ld, sum = %10ld, buf->buffers[%d].size = %10ld\n", ioffs, tsize, ioffs + tsize, i, buf->buffers[i].size);
        if (ioffs >= 0 && ioffs + tsize <= (int64_t) buf->buffers[i].size) {
            res.metal = buf->buffers[i].metal;
            res.offs  = (size_t) ioffs;

            //GGML_LOG_INFO("%s: tensor '%16s', offs = %8ld\n", __func__, t->name, *offs);

            return res;
        }
    }

    GGML_LOG_ERROR("%s: error: tensor '%s' buffer is nil\n", __func__, t->name);

    return res;
}
