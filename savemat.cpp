#include <cstdio>
#include <iostream>
#include <fstream>

int main(int argn, char** args) {
  int myints[5] = {2017, 3, 16, 3, 11};
  char mychars[sizeof(int)*5];
  std::fstream file;


  // file.open("asdf.bin", std::ios::out|std::ios::binary)
  // file.write(reinterpret_cast<char*>(myints), sizeof(int)*5);
  // file.close();

  file.open("asdf.bin", std::ios::in|std::ios::binary);
  if (file.good()) {
    std::cout << "a okay!\n";
  }
  file.read(mychars, sizeof(int)*5);
  int* readints = reinterpret_cast<int*>(mychars);
  int i = 0;
  for (i=0;i < 5; i++) {
    std::cout << readints[i] << "\n";
  }
  file.close();

  // int mynums[5] = {2017, 3, 16, 3, 11};
  // file.write(mynums, sizeof(int) * 5);
  // std::cout << "hello world!\n";
}
