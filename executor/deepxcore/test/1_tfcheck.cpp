#include "deepx/tf/tf.hpp"
#include <iostream>
using namespace std;
using namespace deepx::tf;
 

int main(int argc,char **argv){
    TF t("matmul(float16 a,float16 b)->(float32 c)");
    cout<<t.to_string(false,false)<<endl;
 
    TF other("matmul(float16 a,float16 b)->(float32 c)");
    
    cout<<"checkdtype:"<<t.check_dtype(other)<<endl;
    return 0;
}