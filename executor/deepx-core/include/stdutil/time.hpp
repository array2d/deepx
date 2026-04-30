#ifndef STDUTIL_TIME_HPP
#define STDUTIL_TIME_HPP

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

namespace stdutil{
    using namespace std::chrono;    
    static std::string format_time(const system_clock::time_point &tp)
    {
        auto ms = duration_cast<microseconds>(tp.time_since_epoch());
        auto sec = duration_cast<seconds>(ms);
        ms -= sec;

        std::time_t t = sec.count();
        std::tm tm;
        localtime_r(&t, &tm); // 线程安全版本

        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S")
            << '.' << std::setfill('0') << std::setw(6) << ms.count();
        return oss.str();
    }
}
#endif