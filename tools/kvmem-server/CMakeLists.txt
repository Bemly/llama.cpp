# KVMem OpenAI-compatible server (text + tools; vision via --mmproj).
# Mirrors upstream kvmem-llama.cpp root CMakeLists server recipe.
# Only registered when LLAMA_KVMEM is on (see tools/CMakeLists.txt).

set(TARGET llama-kvmem-server)

add_executable(${TARGET}
    ${LLAMA_KVMEM_ROOT}/tools/llama-kvmem-server.cpp
    ${LLAMA_KVMEM_ROOT}/tools/kvmem-spec.cpp
    ${LLAMA_KVMEM_ROOT}/tools/kvmem-vision.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../server/server-common.cpp)
target_include_directories(${TARGET} PRIVATE
    ${LLAMA_KVMEM_ROOT}/tools
    ${LLAMA_KVMEM_ROOT}/src/adapter
    ${CMAKE_CURRENT_SOURCE_DIR}/../server
    ${CMAKE_CURRENT_SOURCE_DIR}/../mtmd
    ${PROJECT_SOURCE_DIR}/vendor
    ${PROJECT_SOURCE_DIR}/vendor/cpp-httplib)
target_link_libraries(${TARGET} PRIVATE llama llama-common cpp-httplib mtmd)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
