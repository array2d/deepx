
#include <type_traits>

#include "deepx/dtype.hpp"


namespace deepx
{
    // Map C++ scalar type -> Precision (used by tensor constructors and tensorlife helpers)
    template <typename T>
    inline constexpr Precision precision()
    {
        if constexpr (std::is_same_v<T, double>)
            return Precision::Float64;
        else if constexpr (std::is_same_v<T, float>)
            return Precision::Float32;
        else if constexpr (std::is_same_v<T, int64_t>)
            return Precision::Int64;
        else if constexpr (std::is_same_v<T, int32_t>)
            return Precision::Int32;
        else if constexpr (std::is_same_v<T, int16_t>)
            return Precision::Int16;
        else if constexpr (std::is_same_v<T, int8_t>)
            return Precision::Int8;
        else if constexpr (std::is_same_v<T, bool>)
            return Precision::Bool;
        else if constexpr (std::is_same_v<T, std::string>)
            return Precision::String;
        else
            return Precision::Any;
    }
}