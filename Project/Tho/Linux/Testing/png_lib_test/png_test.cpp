#include <libpng16/png.h>
#include <zlib/zlib.h>
#include <iostream>


int main(){
    std::cout << "Hello world" << PNG_LIBPNG_VER_STRING;
    std::cout << "Hello world" << PNG_IMAGE_VERSION;
}