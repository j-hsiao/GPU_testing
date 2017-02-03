#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

__global__ void convolveKernelBordered(uchar* indata, uchar* outdata, float* kernel, int IW,
                                       int KH, int KW, int BW) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int channel = threadIdx.z;
  float val = 0.0f;
  for (int i = 0; i < KH; i++) {
    for (int j = 0; j < KW; j++) {
      int imx = idx + j;
      int imy = idy + i;
      val += indata[(((imy * BW) + imx) * 3) + channel] * kernel[(i * KW) + j];
    }
  }
  val = val < 0 ? 0:val;
  outdata[(((idy * IW) + idx) * 3) + channel] = val > 255 ? 255: (uchar) val;

  //outdata[(((idy * IW) + idx) * 3) + channel] = indata[((((idy + (KH / 2)) * BW) + (idx + (KW / 2))) * 3) + channel];
  //outdata[(((idy * IW) + idx) * 3) + channel] = 0;
}




int main(int argn, char** args){
  cv::VideoCapture cap = cv::VideoCapture(0);
  cv::Mat imframe;
  cv::Mat cvProcessed;
  cv::Mat filtdisp;
  cv::Mat myProcessed;
  cv::Mat bordered;
  int cvswitch;
  int fh = 5;

  cudaEvent_t start, stop;



  cv::Mat blurfilt = cv::Mat(fh,fh,CV_32FC1);
  cv::Size fdispsize = cv::Size(500,500);
  for(int i = 0; i < fh; i++) {
    for (int j = 0; j < fh; j++) {
      blurfilt.at<float>(i,j) = 1.0 / (fh*fh);
      // std::cout << ((float*) blurfilt.data)[blurfilt.step[0]*i + blurfilt.step[1] * j] << std::endl;
      // blurfilt.data<float>[(blurfilt.step[0]*i) + blurfilt.step[1]*j] = 1.0;
      // std::cout << blurfilt.data<float>[(blurfilt.step[0]*i) + blurfilt.step[1]*j];
    }
  }
  cv::resize(blurfilt, filtdisp, fdispsize);
  int KW = blurfilt.size().width;
  int KH = blurfilt.size().height;

  //640x480x3 total
  dim3 grid(160, 120);

  dim3 block(4, 4, 3);
  uchar* kernelInput = 0;
  uchar* kernelOutput = 0;
  float* kernelFilter = 0;

  cudaMalloc((void**) &kernelInput, sizeof(uchar) * (640 + (2 * (KH / 2))) * (480 + (2 * (KW / 2))) * 3);
  // printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  cudaMalloc((void**) &kernelOutput, sizeof(uchar) * 640 * 480 * 3);
  // printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  cudaMalloc((void**) &kernelFilter, sizeof(float) * fh * fh);
  // printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  if (!blurfilt.isContinuous()) {
    std::cout << "blurfilt is not continuous" << std::endl;
    return 1;
  }




  cudaMemcpy((void*) kernelFilter, (void*) blurfilt.data, sizeof(float)*fh*fh, cudaMemcpyHostToDevice);
  // printf("%s\n", cudaGetErrorString(cudaGetLastError()));




  if ((kernelInput == 0)
      || (kernelOutput == 0)
      || (kernelFilter == 0)) {
    std::cout << "failed to allocate" << std::endl;
    return 0;
  }




  while(true) {
    if(cap.read(imframe)) {
      cv::flip(imframe, imframe, 2);
      cv::filter2D(imframe, cvProcessed, -1, blurfilt);
      myProcessed = imframe.clone() * 0;

      // std::cout << myProcessed.size().width << "," << myProcessed.size().height << std::endl;


      if (!myProcessed.isContinuous()) {
        std::cout << "myProcessed not continuous" << std::endl;
        return 1;
      }
      cv::copyMakeBorder(imframe, bordered, fh / 2, fh / 2, fh / 2, fh / 2, cv::BORDER_DEFAULT);
      if (!bordered.isContinuous()) {
        std::cout << "bordered not continuous" << std::endl;
        return 1;
      }

      int IW = imframe.size().width;
      int IH = imframe.size().height;

      int BW = bordered.size().width;


      cudaMemcpy((void*) kernelInput, (void*) bordered.data, sizeof(uchar)*(bordered.size[0]*bordered.size[1]*3), cudaMemcpyHostToDevice);
      printf("copy input: %s\n", cudaGetErrorString(cudaGetLastError()));

      // std::cout << "size of input: " << sizeof(uchar)*(bordered.size[0]*bordered.size[1]*3) << std::endl;
      // std::cout.flush();

      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start, 0);

      convolveKernelBordered<<<grid,block>>>(kernelInput, kernelOutput, kernelFilter, IW, KW, KH, BW);
      
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      float et;
      cudaEventElapsedTime(&et, start, stop);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);

      printf("kernel runtime: %8.7f\n", et);

      cudaMemcpy((void*) myProcessed.data, (void*) kernelOutput, (sizeof(uchar) * IW * IH * 3), cudaMemcpyDeviceToHost);
      printf("copy output: %s\n", cudaGetErrorString(cudaGetLastError()));

      // std::cout << "size of output: " << sizeof(uchar) * IW * IH * 3 << std::endl;
      // std::cout.flush();

      cv::imshow("hello", imframe);
      cv::imshow("cvProcessed", cvProcessed);
      cv::imshow("myProcessed", myProcessed);


      cv::imshow("filt", filtdisp);
    }
    cvswitch = cv::waitKey(1);
    if (cvswitch >= 0) {
      break;
    }
  }
  imframe.release();
  cvProcessed.release();
  cap.release();
  myProcessed.release();
  bordered.release();
}
