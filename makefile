CC = clang
CXX = clang++
NVCC = nvcc
FLAGS = -Wall -Wextra -march=native -Ofast
CXXFLAGS = -fopenmp $(FLAGS)
CCFLAGS = -fopenmp -lm $(FLAGS)
CFLAGS = -lm $(FLAGS)

NVFLAGS := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
NVFLAGS += -Xcompiler "-fopenmp -pthread -Wall -Wextra -march=native"
LDFLAGS  := -lm

EXES := asm_cpu_seq asm_cpu asm_gpu asm_blocked test

all: $(EXES)

clean:
	rm -f $(EXES)

test: test.c
	$(CC) $(CFLAGS) -o $@ $?

asm_cpu_seq: asm_cpu_seq.c
	$(CC) $(CFLAGS) -o $@ $?

asm_cpu: asm_cpu.cc
	$(CC) $(CCFLAGS) -o $@ $?

asm_gpu: asm_gpu.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

asm_blocked: asm_blocked.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?