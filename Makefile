LIB:= -lopencv_core -lopencv_highgui -lopencv_imgproc
all: conv.cpp
	g++ -o conv conv.cpp $(LIB)
	touch all
test: cvtest.cpp
	g++ -o cvtest cvtest.cpp $(LIB)
gpu: conv.cu
	nvcc -o convgpu conv.cu $(LIB)
	touch gpu
t: cudaTest.cu
	nvcc -o t cudaTest.cu $(LIB)
