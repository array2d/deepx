#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <sstream>

#include "deepx/tf/tffactory.hpp"
#include "deepx/dtype.hpp"

namespace deepx::tf
{   
    using namespace std;

    shared_ptr<TF> TfFactory::get_tf(const TF &other) const
    {
        // 检查操作名是否存在
        auto family_it = tf_families.find(other.name);
        if (family_it == tf_families.end())
        {
            cerr << "<op> " << other.name << " not found" << endl;
            return nullptr;
        }

        // 检查作者是否存在
        auto author_it = family_it->second->tf_authors.find(other.metadata.author);
        if (author_it == family_it->second->tf_authors.end())
        {
            cerr << "<op> " << other.name << " author:" << other.metadata.author << " not found" << endl;
            //使用第一个作者
            author_it = family_it->second->tf_authors.begin();
            cerr << "<op> " << other.name << " use first author:" << author_it->first << endl;
            if (author_it == family_it->second->tf_authors.end())
            {
                cerr << "<op> " << other.name << " default author:" << author_it->first << " not found" << endl;
                return nullptr;
            }
        }

        // 提取参数和返回值类型
        vector<TypeDef> arg_types;
        for (const auto &arg : other.args)
        {
            arg_types.push_back(arg.dtype);
        }

        vector<TypeDef> return_types;
        for (const auto &ret : other.returns)
        {
            return_types.push_back(ret.dtype);
        }

        // 尝试找到匹配的实现
        auto tf = author_it->second->get_matching_tf(arg_types, return_types);
        if (!tf)
        {
            cerr << "<op> " << other.name << " " << other.to_string(false, false) << " not found" << endl;
            cerr << "supported dtypes: " << endl;
            // 遍历所有已注册的实现
            for (const auto &registered_tf : author_it->second->tfs)
            {
                cerr << "(";
                for (size_t i = 0; i < registered_tf->args.size(); i++)
                {
                    if (i > 0)
                        cerr << ", ";
                    cerr << dtype_str(registered_tf->args[i].dtype);
                }
                cerr << ")->(";
                for (size_t i = 0; i < registered_tf->returns.size(); i++)
                {
                    if (i > 0)
                        cerr << ", ";
                    cerr << dtype_str(registered_tf->returns[i].dtype);
                }
                cerr << ")" << endl;
            }
            return nullptr;
        }

        // 使用clone()方法创建新实例，而不是直接复制构造
        auto cloned = tf->clone();
        cloned->metadata=other.metadata;
        return cloned;
    }
    string TfFactory::print_markdown(string excuter_name) const
    {
        std::stringstream ss;
        ss << "## " << excuter_name << " 支持算子列表 \n\n";
        ss << "本页面由 `excuter/" << excuter_name << " 生成，请勿手动修改 \n\n";
 
        // 首先按tftype分组
        unordered_map<string, vector<shared_ptr<TF>>> tf_by_type;
        
        // 收集所有TF并按tftype分组
        for (const auto &[name, tf_family] : tf_families) {
            for (const auto &[author, tf_author] : tf_family->tf_authors) {
                for (const auto &tf : tf_author->tfs) {
                    tf_by_type[tf->tftype].push_back(tf);
                }
            }
        }
        
        // 为每个tftype生成一个表格
        for (const auto &[tftype, tfs] : tf_by_type) {
            ss << "### " << tftype << "\n\n";
            ss << "| Operation | Author |  Math Formula | IR Instruction |\n";
            ss << "|-----------|--------|--------------|----------------|\n";
            
            for (const auto &tf : tfs) {
                ss << "| " << tf->name << " | ";
                ss << (tf->metadata.author.empty() ? " none " : tf->metadata.author) << " | ";
                ss << tf->math_formula() << " | ";
                ss << stdutil::escape_markdown(tf->to_string(false, true)) << " |\n";
            }
            
            ss << "\n";
        }
        
        return ss.str();
    }
}