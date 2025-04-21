
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
    void Shape::setshape(const int *shape, int dim)
    {
        this->shape.resize(dim);
        this->dim = dim;
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
        for (int i = 0; i < dim; ++i)
        {
            std::cout << shape[i];
            if (i < dim - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    int Shape::linearat(const std::vector<int> &indices) const{
        int idx=0;
        for(int i=0;i<indices.size();i++){
            idx+=indices[i]*strides[i];
        }
        return idx;
    }
    std::vector<int> Shape::linearto(int idx_linear) const{
        std::vector<int> indices(dim,0);
        for(int i=0;i<dim;i++){
            indices[i]=idx_linear/strides[i];
            idx_linear%=strides[i];
        }
        return indices;
    }

    std::string Shape::toYaml() const{
        YAML::Node node;
        node["dtype"] = precision_str(dtype);
        node["dim"] = dim;
        node["shape"] = shape;
        node["stride"] = strides;
        node["size"] = size;
        return YAML::Dump(node);
    }
    void Shape::fromYaml(const std::string &yaml){
        YAML::Node node = YAML::Load(yaml);
        dtype = precision(node["dtype"].as<std::string>());
        dim = node["dim"].as<int>();
        shape = node["shape"].as<std::vector<int>>();
        strides=node["stride"].as<std::vector<int>>();
        size=node["size"].as<int>();
    }
}