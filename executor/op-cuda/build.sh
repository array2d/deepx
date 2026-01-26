mkdir -p build && cd build    
rm -rf build/*
cmake ..
make -j$(nproc)
