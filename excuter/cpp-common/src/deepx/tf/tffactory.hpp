#ifndef DEEPX_TF_TFFACTORY_HPP
#define DEEPX_TF_TFFACTORY_HPP

#include "deepx/tf/tf.hpp"

namespace deepx::tf
{

    struct TypeSignature
    {
        vector<TypeDef> args;
        vector<TypeDef> returns;

        bool is_compatible(const TypeSignature &other) const
        {
            return is_compatible_types(args, other.args) &&
                   is_compatible_types(returns, other.returns);
        }

    private:
        static bool is_compatible_types(const vector<TypeDef> &def, const vector<TypeDef> &other)
        {
            if (def.size() != other.size())
                return false;
            for (size_t i = 0; i < def.size(); i++)
            {
                if (!def[i].match(other[i]))
                    return false;
            }
            return true;
        }
    };
    // tf，包括不同dtypes的实现
    struct TFAuthor
    {
        vector<std::shared_ptr<TF>> tfs;

        // 获取匹配的TF实现
        std::shared_ptr<TF> get_matching_tf(const vector<TypeDef> &arg_types,
                                            const vector<TypeDef> &return_types) const
        {
            TypeSignature target{arg_types, return_types};

            for (const auto &tf : tfs)
            {
                vector<TypeDef> tf_arg_types;
                for (const auto &arg : tf->args)
                {
                    tf_arg_types.push_back(arg.dtype);
                }

                vector<TypeDef> tf_return_types;
                for (const auto &ret : tf->returns)
                {
                    tf_return_types.push_back(ret.dtype);
                }

                TypeSignature current{tf_arg_types, tf_return_types};
                if (target.is_compatible(current))
                {
                    return tf;
                }
            }
            return nullptr;
        }
    };

    // 同前缀的op，但是不同作者的实现
    struct TFFamily
    {
        std::string _default;
        std::unordered_map<std::string, std::shared_ptr<TFAuthor>> tf_authors;
    };

    class TfFactory
    {
    public:
        std::unordered_map<std::string, std::shared_ptr<TFFamily>> tf_families;
        void add_tf(std::shared_ptr<TF> tf)
        {
            // 检查是否存在该op名称，不存在则创建
            if (tf_families.find(tf->name) == tf_families.end())
            {
                tf_families[tf->name] = std::make_shared<TFFamily>();
            }

            // 检查是否存在该作者的实现，不存在则创建
            if (tf_families[tf->name]->tf_authors.find(tf->author) ==
                tf_families[tf->name]->tf_authors.end())
            {
                tf_families[tf->name]->tf_authors[tf->author] = std::make_shared<TFAuthor>();
            }

            // 直接添加到vector中
            tf_families[tf->name]->tf_authors[tf->author]->tfs.push_back(tf);
        }
        shared_ptr<TF> get_tf(const TF &other) const;
        // 输出为markdown表格格式
        string print_markdown(string excuter_name) const;
    };
}

#endif
