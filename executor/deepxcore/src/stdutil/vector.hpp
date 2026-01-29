#ifndef STDUTIL_VECTOR_HPP
#define STDUTIL_VECTOR_HPP

#include <vector>
#include <ostream>

// 全局重载 operator<<
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i < vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}
#endif // STDUTIL_VECTOR_HPP