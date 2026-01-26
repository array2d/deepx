#pragma once

namespace memcuda {

class StreamSync {
public:
    StreamSync();
    void Record();
    void Wait();
};

}
