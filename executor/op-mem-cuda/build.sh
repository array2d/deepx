cd build    
rm -rf *
cmake ..
make -j$(nproc)
