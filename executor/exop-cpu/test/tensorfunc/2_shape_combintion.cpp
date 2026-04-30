#include <iostream>
#include <vector>
#include "deepx/vector_combination.hpp"
#include "stdutil/vector.hpp"
using namespace deepx;

void test_combination()
{
    std::vector<std::vector<int>> result = combination(3);
    for (const auto &comb : result)
    {
        std::cout << "Combination:"<<comb<<std::endl;
    }
}
int main()
{
    test_combination();
    return 0;
}