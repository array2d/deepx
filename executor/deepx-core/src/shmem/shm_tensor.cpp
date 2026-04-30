#include "deepx/shm_tensor.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <cstdio>

namespace deepx::shmem {

size_t shm_page_align(size_t byte_size) {
    static long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) ps = 16384; // Apple Silicon default
    return (byte_size + ps - 1) & ~(ps - 1);
}

bool shm_tensor_create(const std::string &shm_name, size_t byte_size, ShmTensor &out) {
    size_t aligned = shm_page_align(byte_size);

    int fd = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
        fprintf(stderr, "shm_tensor_create: shm_open(%s) failed: %s\n",
                shm_name.c_str(), strerror(errno));
        return false;
    }

    if (ftruncate(fd, aligned) < 0) {
        fprintf(stderr, "shm_tensor_create: ftruncate failed: %s\n", strerror(errno));
        close(fd);
        shm_unlink(shm_name.c_str());
        return false;
    }

    void *addr = mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        fprintf(stderr, "shm_tensor_create: mmap failed: %s\n", strerror(errno));
        close(fd);
        shm_unlink(shm_name.c_str());
        return false;
    }

    close(fd);

    out.shm_name  = shm_name;
    out.addr      = addr;
    out.byte_size = byte_size; // original requested size
    out.fd        = -1;        // already closed after mmap
    out.refcount  = 1;
    return true;
}

bool shm_tensor_open(const std::string &shm_name, size_t byte_size, ShmTensor &out) {
    size_t aligned = shm_page_align(byte_size);

    int fd = shm_open(shm_name.c_str(), O_RDWR, 0600);
    if (fd < 0) {
        fprintf(stderr, "shm_tensor_open: shm_open(%s) failed: %s\n",
                shm_name.c_str(), strerror(errno));
        return false;
    }

    void *addr = mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        fprintf(stderr, "shm_tensor_open: mmap failed: %s\n", strerror(errno));
        close(fd);
        return false;
    }

    close(fd);

    out.shm_name  = shm_name;
    out.addr      = addr;
    out.byte_size = byte_size;
    out.fd        = -1;
    out.refcount  = 1;
    return true;
}

void shm_tensor_close(ShmTensor &t) {
    if (t.addr && t.byte_size > 0) {
        munmap(t.addr, shm_page_align(t.byte_size));
        t.addr = nullptr;
    }
    t.byte_size = 0;
    t.refcount  = 0;
}

void shm_tensor_unlink(const std::string &shm_name) {
    shm_unlink(shm_name.c_str());
}

} // namespace deepx::shmem
