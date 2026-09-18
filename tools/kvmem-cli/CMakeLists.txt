# Evaluation scaffolding: builds kvmem's own CLI against this tree.
# Only registered when LLAMA_KVMEM is on (see tools/CMakeLists.txt).

set(TARGET llama-kvmem-cli)

add_executable(${TARGET}
    ${LLAMA_KVMEM_ROOT}/tools/llama-kvmem-cli.cpp
    ${LLAMA_KVMEM_ROOT}/tools/kvmem-spec.cpp)
target_include_directories(${TARGET} PRIVATE
    ${LLAMA_KVMEM_ROOT}/src/adapter)
target_link_libraries(${TARGET} PRIVATE llama llama-common)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
