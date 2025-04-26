 
#include <deepx/tensor.hpp>
#include <iostream>

#include <stdutil/vector.hpp>
using namespace deepx;
 
void test_tensor_shape() {
    Shape shape({2, 3, 4});
    std::cout << "print shape: " << shape.size << std::endl;
    std::string yaml=shape.toYaml();
    std::cout<<"yaml:"<<std::endl<<yaml<<std::endl<<std::endl<<std::endl;
    
    Shape shape2;
    shape2.fromYaml(yaml);
    std::cout<<"shape2: "<<shape2.size<<" "<<shape2.shape<<" "<<shape2.dim()<<" "<<shape2.strides<<std::endl;
    
}

int main() {
    test_tensor_shape();
    return 0;
}
 