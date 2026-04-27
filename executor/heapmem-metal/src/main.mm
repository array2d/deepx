#include <iostream>
#include <string>
#include <sstream>
#include <thread>
#include <chrono>
#include <atomic>
#include <unistd.h>

#include "registry/registry_file.h"
#include "lifecycle/lifecycle.h"

using namespace deepx::heapmem;

// 简单 stdin 交互式 demo（生产环境替换为 Redis list 消费）
static void print_help() {
    std::cout << "heapmem-metal commands:\n"
              << "  create <name> <dtype> <dim1> [dim2] [dim3]\n"
              << "  get    <name>\n"
              << "  delete <name>\n"
              << "  list\n"
              << "  quit\n";
}

int main(int argc, char **argv) {
    const char *registry_path = "/tmp/deepx_heapmem_registry.txt";
    if (argc > 1) registry_path = argv[1];

    FileRegistry reg(registry_path);
    LifecycleManager mgr(&reg);

    std::cout << "heapmem-metal started. Registry: " << registry_path << "\n";
    print_help();

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty() || line[0] == '#') continue;

        // parse
        std::istringstream iss(line);
        std::string op;
        iss >> op;

        if (op == "quit" || op == "exit") {
            break;
        }
        if (op == "help") {
            print_help();
            continue;
        }
        if (op == "list") {
            // 简单遍历文件
            std::cout << "Listing registry entries:\n";
            // FileRegistry doesn't expose iteration, we'd add it in production
            continue;
        }

        LifecycleCommand cmd;
        cmd.op = op;
        iss >> cmd.name;

        if (op == "create") {
            std::string dtype;
            iss >> dtype;
            cmd.dtype = dtype;

            // parse shape
            int64_t total = 1;
            int d;
            std::string shape_str = "[";
            bool first = true;
            while (iss >> d) {
                if (!first) shape_str += ",";
                shape_str += std::to_string(d);
                total *= d;
                first = false;
            }
            shape_str += "]";
            cmd.shape = shape_str;
            cmd.element_count = total;
            cmd.device = 0;
            cmd.pid = getpid();
        }

        std::string error;
        std::string shm_name = mgr.handle(cmd, error);

        if (!error.empty()) {
            std::cerr << "ERROR: " << error << "\n";
        } else if (!shm_name.empty()) {
            std::cout << "  → shm: " << shm_name << "\n";
        } else if (op == "delete") {
            std::cout << "  → deleted\n";
        } else if (op == "get") {
            std::cout << "  → shm: " << shm_name << "\n";
        }
    }

    mgr.shutdown();
    std::cout << "heapmem-metal stopped.\n";
    return 0;
}
