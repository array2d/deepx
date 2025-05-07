#include <mutex>
#include <thread>
#include <cstdlib>

#include <deepx/tensorfunc/init.hpp>
#include "deepx/tf/tf.hpp"
#include "deepx/tf/tffactory.hpp"
#include "client/tfs.hpp"
#include "deepx/mem/mem.hpp"
#include "deepx/mem/mem_ompsimd.hpp"
#include "client/udpserver.hpp"

using namespace deepx::tensorfunc;
using namespace deepx::mem;

// 从环境变量读取IR日志配置
bool kIrLog = []()
{
    const char *env = std::getenv("DEEPX_IR_LOG");
    return env != nullptr && (strcmp(env, "1") == 0 || strcasecmp(env, "true") == 0);
}();

int main()
{
    shared_ptr<MemBase> mem = make_shared<Mem>();
    std::mutex memmutex;

    client::udpserver server(8080);
    deepx::tf::TfFactory tf_factory;
    register_all(tf_factory);

    // 将op table输出到markdown文件
    string docdir = "../../../doc/excuter/op-mem-ompsimd/";
    std::ofstream md_file(docdir + "list.md");
    if (md_file.is_open())
    {
        md_file << tf_factory.print_markdown("op-mem-ompsimd");
        md_file.close();
    }

    queue<deepx::tf::TF> tasks;
    // 启动一个新线程来运行UDP服务器
    std::thread server_thread([&server, &tasks]()
                              { server.start(tasks); });
    // 分离线程，让它在后台运行
    server_thread.detach();

    while (true)
    {
        if (!tasks.empty())
        {
            deepx::tf::TF op = tasks.front();
            tasks.pop();

            // 根据kIrLog标志决定是否打印op信息
            if (kIrLog)
            {
                cout << "~" << op.to_string() << endl;
            }

            deepx::tf::OpResp opresp;
            opresp.id = op.metadata.id;
            opresp.recv_at = op.metadata.recv_at;

            auto src = tf_factory.get_tf(op);
            if (src == nullptr)
            {
                opresp.error("op" + op.name + " not found");
                server.resp(opresp.to_string());
                continue;
            }
            (*src).init(op.name, op.args, op.returns);
            memmutex.lock();
            opresp.start_at = chrono::system_clock::now();
            int ret = 0;
            if ((*src).metadata.benchmark.repeat > 1)
            {
                for (int i = 0; i < (*src).metadata.benchmark.repeat; i++)
                {
                    ret = (*src).run(mem, opresp.message);
                    if (ret != 0)
                    {
                        break;
                    }
                }
            }
            else
            {
                ret = (*src).run(mem, opresp.message);
            }
            memmutex.unlock();
            if (ret != 0)
            {
                opresp.error(opresp.message);
                server.resp(opresp.to_string());
                cerr << opresp.message << endl;
                continue;
            }
            opresp.finish("");
            server.resp(opresp.to_string());
        }
    }
    return 0;
}
