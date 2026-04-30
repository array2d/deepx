#include <functional>

#include "deepx/vector_combination.hpp"

namespace deepx
{
    using namespace std;
    vector<vector<int>> combination(int n, int k)
    {
        if (k > n || k < 0)
        {
            return {};
        }
        if (k == 0)
        {
            return {{}};
        }

        vector<vector<int>> result;
        vector<int> path;

        // 递归函数
        function<void(int)> backtrack = [&](int start)
        {
            if (path.size() == k)
            {
                result.push_back(path);
                return;
            }
            for (int i = start; i < n; i++)
            {
                path.push_back(i);
                backtrack(i + 1);
                path.pop_back(); // 回溯
            }
        };

        backtrack(0);
        return result;
    }
    vector<vector<int>> combination(int n )
    {
        vector<vector<int>> result;
        for (int k = 0; k <= n; k++)
        {
            vector<vector<int>> temp = combination(n, k);
            result.insert(result.end(), temp.begin(), temp.end());
        }
        return result;
    }
    vector<int> arrange(int n)
    {
        vector<int> result;
        for (int i = 0; i < n; i++)
        {
            result.push_back(i);
        }
        return result;
    }
}