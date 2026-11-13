#include "llama-io.h"

#include "ggml-backend.h"

#include <vector>

void llama_io_write_i::write_tensor_quantized(ggml_tensor * tensor, size_t offset, size_t size, ggml_type type, int64_t n_per_row) {
    std::vector<float> f(size/sizeof(float));
    ggml_backend_tensor_get(tensor, f.data(), offset, size);
    const int64_t nrows = (int64_t) f.size()/n_per_row;
    std::vector<uint8_t> q(ggml_row_size(type, n_per_row)*nrows);
    ggml_quantize_chunk(type, f.data(), q.data(), 0, nrows, n_per_row, nullptr);
    write(q.data(), q.size());
}

void llama_io_write_i::write_string(const std::string & str) {
    uint32_t str_size = str.size();

    write(&str_size,  sizeof(str_size));
    write(str.data(), str_size);
}

void llama_io_read_i::read_string(std::string & str) {
    uint32_t str_size;
    read(&str_size, sizeof(str_size));

    std::vector<char> buf(str_size);
    read(buf.data(), str_size);

    str.assign(buf.data(), str_size);
}
