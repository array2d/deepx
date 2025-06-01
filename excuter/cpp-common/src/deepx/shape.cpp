
#include <iostream>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "tensor.hpp"
#include "deepx/dtype.hpp"
namespace deepx
{
    Shape::Shape(const int *shape, int dim)
    {
        setshape(shape, dim);
    }
    int Shape::dim() const{
        return shape.size();
    }
    int64_t Shape::bytes() const{
        return size * (precision_bits(dtype) / 8);
    }
    void Shape::setshape(const int *shape, int dim)
    {
        this->shape.resize(dim);
        std::copy(shape, shape + dim, this->shape.begin());
        strides.resize(dim);
        strides[dim - 1] = 1;
        for (int i = dim - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        size = strides[0] * shape[0];
    }
    Shape::Shape(const std::vector<int> &shape)
    {
        setshape(shape.data(), shape.size());
    }
    Shape::Shape(const std::initializer_list<int> &shape)
    {
        setshape(shape.begin(), shape.size());
    }
    int Shape::operator[](int index) const
    {
        if (index < 0)
        {
            index += shape.size();
        }
        return shape[index];
    }
    int &Shape::operator[](int index)
    {
        if (index < 0)
        {
            index += shape.size();
        }
        return shape[index];
    }
   
    void Shape::print() const
    {
        std::cout << "shape:[";
        for (int i = 0; i < dim(); ++i)
        {
            std::cout << shape[i];
            if (i < dim() - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    //linearat
    //linearat 支持和strides不同dim的indice索引
    //当strides.size() < indices.size()时，indices的前面部分会默认都是unsqueeze的维度，从而被忽略。在repeat方法中，用到。
    //当strides.size() > indices.size()时，strides的后面部分会被忽略不计算
    int Shape::linearat(const std::vector<int> &indices) const{
        int idx=0;
        int stride_i=0,indices_i=0;
        if (indices.size()> shape.size()) {
            indices_i=indices.size()-shape.size();
        }
        for(;stride_i<strides.size()&&stride_i<indices.size();){
            idx+=indices[indices_i++]*strides[stride_i++];
        }       
        return idx;
    }
    std::vector<int> Shape::linearto(int idx_linear) const{
        std::vector<int> indices(dim(),0);
        for(int i=0;i<dim();i++){
            indices[i]=idx_linear/strides[i];
            idx_linear%=strides[i];
        }
        return indices;
    }

    std::string Shape::toYaml() const{
        YAML::Node node;
        node["dtype"] = precision_str(dtype);
        node["dim"] = dim();
        node["shape"] = shape;
        node["stride"] = strides;
        node["size"] = size;
        return YAML::Dump(node);
    }
    void Shape::fromYaml(const std::string &yaml){
        YAML::Node node = YAML::Load(yaml);
        dtype = precision(node["dtype"].as<std::string>());
        shape = node["shape"].as<std::vector<int>>();
        strides=node["stride"].as<std::vector<int>>();
        size=node["size"].as<int>();
        
        //check
        Shape checkedshape(shape);
        if(checkedshape.shape!=shape){
            throw std::runtime_error("Shape::fromYaml: shape mismatch");
        }
        if(checkedshape.strides!=strides){
            throw std::runtime_error("Shape::fromYaml: strides mismatch");
        }
         if(checkedshape.size!=size){
            throw std::runtime_error("Shape::fromYaml: size mismatch");
        }
    }

    void Shape::saveShape( const std::string &tensorPath) const{
            std::string shapedata = toYaml();
            std::ofstream shape_fs(tensorPath + ".shape", std::ios::binary);
            shape_fs.write(shapedata.c_str(), shapedata.size());
            shape_fs.close();
        }

    pair<std::string,Shape> Shape::loadShape(const std::string &path)   
    {
        std::string shapepath = path + ".shape";
        std::ifstream shape_fs(shapepath, std::ios::binary);
        if (!shape_fs.is_open())
        {
                throw std::runtime_error("Failed to open shape file: " + shapepath);
            }
            std::string shapedata((std::istreambuf_iterator<char>(shape_fs)), std::istreambuf_iterator<char>());
            Shape shape;
            shape.fromYaml(shapedata);
            std::string filename = stdutil::filename(path);
            std::string tensor_name = filename.substr(0, filename.find_last_of('.'));
            return std::make_pair(tensor_name, shape);
        }
}