#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

void pv(Vec3b v) {
  cout << "(" << (int)v[0] << ", " << (int)v[1] << ", " << (int)v[2] << ")" << endl;
}

void pv(Vec3f v) {
  cout << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")" << endl;

}

void pi(float* data, int  x, int  y, int  z, int  H, int  W, int  C) {
  cout << data[(((y * W) + x) * 3) + z] << endl;
}




int main(int argn, char** args) {
  Vec3f check = Vec3f(255, 255, 255);
  cout << check[0] << endl;
  cout << check[1] << endl;
  cout << check[2] << endl;
  Mat im = Mat(2,2,CV_32FC3);
  float* dataPointer = reinterpret_cast<float*>(im.data);
  for (int i = 0; i < 12; i++) {
    dataPointer[i] = 0;
  }
  for (int i = 0; i < 6; i++) {
    dataPointer[i] = 69;
  }

  cout << im.isContinuous() << endl;

  //im.at is row, column (dim0, dim1, dim2)
  //                      row , col , chan...
  pv(im.at<Vec3f>(0,0));
  pv(im.at<Vec3f>(0,1));
  pv(im.at<Vec3f>(1,0));
  pv(im.at<Vec3f>(1,1));

  
  int H = 2;
  int W = 2;
  int C = 3;
  pi(dataPointer, 0, 0, 0, 2, 2, 3);
  pi(dataPointer, 0, 0, 1, 2, 2, 3);
  pi(dataPointer, 0, 0, 2, 2, 2, 3);
  pi(dataPointer, 1, 0, 0, 2, 2, 3);
  pi(dataPointer, 1, 0, 1, 2, 2, 3);
  pi(dataPointer, 1, 0, 2, 2, 2, 3);

}
