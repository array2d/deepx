#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace deepx::shmem {

// POSIX shared memory tensor allocation.
// On Apple Silicon (UMA), this memory is directly GPU-accessible via
// MTLBuffer(newBufferWithBytesNoCopy).

struct ShmTensor {
    std::string shm_name;   // e.g. "/deepx_t_<uuid>"
    void      *addr = nullptr;
    size_t     byte_size = 0;
    int        fd = -1;
    int        refcount = 0;
};

// Create a new POSIX shm region for a tensor.
// Returns true on success, fills `out`.
bool shm_tensor_create(const std::string &shm_name, size_t byte_size, ShmTensor &out);

// Open an existing shm region. Returns true on success.
bool shm_tensor_open(const std::string &shm_name, size_t byte_size, ShmTensor &out);

// Close (unmap + close fd). Does NOT unlink.
void shm_tensor_close(ShmTensor &t);

// Unlink the shm from the filesystem (after all users closed).
void shm_tensor_unlink(const std::string &shm_name);

// Page-aligned size for the given byte count.
size_t shm_page_align(size_t byte_size);

} // namespace deepx::shmem
