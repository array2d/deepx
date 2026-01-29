#include "deepx/tf/tf.hpp"
#include <iostream>
using namespace std;
using namespace deepx::tf;

unordered_map<string,pair<TF,TF>> op_name_map = {
    {"matmul",{
        TF("matmul(float16,float16)->(float32)"),
        TF("matmul(a,b)->(c)")}},
    {"matmul2",{
        TF("matmul(float16|float32 a,float16|float32 b)->(float16|float32 c)"),
        TF("matmul(a,b)->(c)")}},
    {"newtensor",{
        TF("newtensor(shape)->(float16 A)"),
        TF("newtensor([3 4 5])->(A)")}},
    {"argset",{
        TF("argset(vector)->(int32 A)"),
        TF("argset([3 4 5])->(A)")}},
    {"argset2",{
        TF("argset(int32 a)->(int32 shape)"),
        TF("argset(a)->(shape)")}},
    {"argset3",{
        TF("argset(float32 1.34)->(float32 var1)"),
        TF("argset(a)->(var1)")}},
    {"argset4",{
        TF("argset(1.34,2.34)->(float32 v1)"),
        TF("argset(a,b)->(v1)")}},
};

int main(int argc,char **argv){
    if(argc!=2){
        cout<<"usage: "<<argv[0]<<" opname"<<endl;
        return 1;
    }
    string opname=argv[1];
    if(op_name_map.find(opname)==op_name_map.end()){
        cout<<"opname not found"<<endl;
        return 1;
    }
    bool show_name=true;
    cout<<"deffunc"<<op_name_map[opname].first.to_string(false,show_name)<<endl;
    cout<<"funccall"<<op_name_map[opname].second.to_string(false,show_name)<<endl;
    cout<<"funcmap_key:"<<op_name_map[opname].first.dtypes()<<endl;

    
    return 0;
}