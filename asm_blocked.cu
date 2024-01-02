#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#define alphabet_size 26
#define num_thread 16

/*
Optimized from GPU to blocked version
*/

__device__ char POSSIBLE_CHAR[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

__global__ void compute_X(int *X, int *T, int n) {
    int row = threadIdx.x;
    for (int j = 0; j <= n; j++) {
        if (j == 0)
            X[row*(n+1)+j] = 0;
        else if (T[j-1] == int(POSSIBLE_CHAR[row]))
            X[row*(n+1)+j] = j;
        else
            X[row*(n+1)+j] = X[row*(n+1)+(j-1)];
    }
}

__global__ void compute_Dist(int *Dist, int *X, int *T, int *P, int n, int m, int rd) {
    int col = blockIdx.x * num_thread + threadIdx.x;
    if (col > n)
        return;
    if (rd == 0) 
        Dist[rd*(n+1)+col] = 0;
    else if (col == 0)
        Dist[rd*(n+1)+col] = rd;
    else if (T[col-1] == P[rd-1])
        Dist[rd*(n+1)+col] = 1 + min(Dist[(rd-1)*(n+1)+col], min(Dist[(rd-1)*(n+1)+(col-1)], rd+col-1));
    else if (X[(P[rd-1]-int('A'))*(n+1) + col] == 0)
        Dist[rd*(n+1)+col] = 1 + min(Dist[(rd-1)*(n+1)+col], Dist[(rd-1)*(n+1)+(col-1)]);
    else
        Dist[rd*(n+1)+col] = Dist[(rd-1)*(n+1) + X[(P[rd-1]-int('A'))*(n+1) + col]] + (col-1-X[(P[rd-1]-int('A'))*(n+1) + col]);
    __syncthreads();
}

int main(int argc, char **argv) {

    assert(argc == 4);

    char *in_T = argv[1], *in_P = argv[2], *out = argv[3];
    FILE *input_T, *input_P, *output;
    input_T = fopen(in_T, "rb");
    input_P = fopen(in_P, "rb");

    int n, m;
    int *T, *P;

    fread(&n, 1, sizeof(int), input_T);
    fread(&m, 1, sizeof(int), input_P);

    // T = (int*)malloc(sizeof(int)*n);
    // P = (int*)malloc(sizeof(int)*m);
    cudaMallocHost(&T, sizeof(int)*n);
    cudaMallocHost(&P, sizeof(int)*m);

    fread(T, n, sizeof(int), input_T);
    fread(P, m, sizeof(int), input_P);

    fclose(input_T);
    fclose(input_P);

    int *host_Dist, *device_Dist, *host_X, *device_X;

    cudaMallocHost(&host_X, sizeof(int)*(n+1)*alphabet_size);
    cudaMallocHost(&host_Dist, sizeof(int)*(n+1)*(m+1));
    // host_Dist = (int*)malloc(sizeof(int)*(n+1)*(m+1));

    cudaMalloc((void**)&device_Dist, sizeof(int)*(n+1)*(m+1));
    cudaMalloc((void**)&device_X, sizeof(int)*(n+1)*alphabet_size);
    // cudaMemcpy(device_Dist, host_Dist, sizeof(int)*n*m, cudaMemcpyHostToDevice);

    compute_X <<< 1, alphabet_size >>> (device_X, T, n);

    int nblocks = (n+1)/num_thread+1;
    for (int i = 0; i <= m; i++)
        compute_Dist <<< nblocks, num_thread >>> (device_Dist, device_X, T, P, n, m, i);

    cudaMemcpy(host_Dist, device_Dist, sizeof(int)*(n+1)*(m+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_X, device_X, sizeof(int)*(n+1)*alphabet_size, cudaMemcpyDeviceToHost);

    output = fopen(out, "wb");
    fwrite(host_Dist, (n+1)*(m+1), sizeof(int), output);
    fclose(output);

    return 0;
}