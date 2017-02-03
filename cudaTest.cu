#include <iostream>
using namespace std;

__global__ void doSomething(int* outdata) {
  *outdata = 69;
}


int main(int argn, char** args) {
  int a = 5;
  cout << a << endl;
  int* kOut = 0;
  cudaMalloc((void**) &kOut, sizeof(int));
  if (kOut == 0) {
    cerr << "crap, malloc failed" << endl;
    return 1;
  }
  doSomething<<<1,1>>>(kOut);
  cudaMemcpy(&a, kOut, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(kOut);
  cout << a << endl;
  return 0;
}