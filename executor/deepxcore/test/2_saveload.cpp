#include "stdutil/fs.hpp"
#include <iostream>
using namespace stdutil;
void test_save(int total_size){
    stdutil::byte *data = new stdutil::byte[total_size];
    for(int i=0;i<total_size;i++){
        data[i] =32+ i;
    }
    save(data,total_size,"test.bin");
    delete[] data;
}   

void test_load(int total_size ){
 
    auto [size,dataptr]=load("test.bin");
    stdutil::byte *data = dataptr.get();
    if (size != total_size){
        cout<<"load failed"<<endl;
    }
    for (int i=0;i<total_size;i++){
        cout<<data[i]<<" ";
    }
    cout<<endl;
}

int main(int argc,char **argv){
    int total_size = 96;
    test_save(total_size);
    test_load(total_size);
    return 0;
}
