# Build script for project

NVCC		:= /usr/local/cuda-10.0/bin/nvcc
LD_LIBRARY_PATH	:= /usr/local/cuda-10.0/lib64/
FLAGS		:= -O3 -ccbin g++ -m64 --gpu-architecture=sm_61

# Add source files here
EXECUTABLE	:= vector_dot_product 
# Cuda source files (compiled with cudacc)
CUFILES_sm_13		:= vector_dot_product.cu
CCFILES		:= \
		   vector_dot_product_gold.cpp

SOURCE		:= vector_dot_product.cu vector_dot_product_gold.cpp 
OBJ		:= vector_dot_product.o vector_dot_product_gold.o 
PROGRAM		:= vector_dot_product


all:$(PROGRAM)

$(PROGRAM): $(SOURCE)
	 $(NVCC) $(FLAGS) --device-c $(SOURCE)
	 $(NVCC) $(FLAGS) $(OBJ) --output-file $(PROGRAM)

clean:
	rm *.o
# Rules and targets
# NVCCFLAGS := -arch sm_13
# include ../../common/common.mk
