#include "deepx/shape_tensorinit.hpp"

namespace deepx
{
     std::pair<int, int> calculateFanInAndFanOut(const Shape &shape)
    {
        int fanIn, fanOut;
        if (shape.dim() < 2)
        {
            fanIn = 1;
            fanOut = 1;
            return std::make_pair(fanIn, fanOut);
        }

        int numInputFmaps = shape[1];  // 输入特征图数量
        int numOutputFmaps = shape[0]; // 输出特征图数量
        int receptiveFieldSize = 1;
        if (shape.dim() > 2)
        {
            for (int i = 2; i < shape.dim(); ++i)
            {
                receptiveFieldSize *= shape[i]; // 计算感受野大小
            }
        }

        fanIn = numInputFmaps * receptiveFieldSize;
        fanOut = numOutputFmaps * receptiveFieldSize;
        return std::make_pair(fanIn, fanOut);
    }

}