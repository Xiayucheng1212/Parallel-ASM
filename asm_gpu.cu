#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#define block_size 16
#define alphabet_size 26

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

__global__ void compute_Dist(int *Dist, int *X, int *T, int *P, int n, int m) {
    int col = threadIdx.x;
    for (int i = 0; i <= m; i++) {
        if (i == 0) 
            Dist[i*(n+1)+col] = 0;
        else if (col == 0)
            Dist[i*(n+1)+col] = i;
        else if (T[col-1] == P[i-1])
            Dist[i*(n+1)+col] = 1 + min(Dist[(i-1)*(n+1)+col], min(Dist[(i-1)*(n+1)+(col-1)], i+col-1));
        else if (X[(P[i-1]-int('A'))*(n+1) + col] == 0)
            Dist[i*(n+1)+col] = 1 + min(Dist[(i-1)*(n+1)+col], Dist[(i-1)*(n+1)+(col-1)]);
        else
            Dist[i*(n+1)+col] = Dist[(i-1)*(n+1) + X[(P[i-1]-int('A'))*(n+1) + col]] + (col-1-X[(P[i-1]-int('A'))*(n+1) + col]);
        __syncthreads();
    }
}

int main(int argc, char **argv) {

    assert(argc == 2);

    char* name = argv[1];
    char *in_T, *in_P, *out;
    in_T = (char*)malloc(sizeof(char)*(strlen(name)+3));
    in_P = (char*)malloc(sizeof(char)*(strlen(name)+3));
    out = (char*)malloc(sizeof(char)*(strlen(name)+5));
    strcpy(in_T, name);
    strcpy(in_P, name);
    strcpy(out, name);
    strcat(in_T, "_T");
    strcat(in_P, "_P");
    strcat(out, "_out");
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

    // dim3 compute_X_nblocks((n+1)/block_size+1, m/block_size+1);
    // dim3 compute_X_nthreads(block_size, block_size);

    compute_X <<< 1, alphabet_size >>> (device_X, T, n);
    compute_Dist <<< 1, n+1 >>> (device_Dist, device_X, T, P, n, m);

    cudaMemcpy(host_Dist, device_Dist, sizeof(int)*(n+1)*(m+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_X, device_X, sizeof(int)*(n+1)*alphabet_size, cudaMemcpyDeviceToHost);

    output = fopen(out, "wb");
    fwrite(host_Dist, (n+1)*(m+1), sizeof(int), output);
    fclose(output);

    return 0;
}