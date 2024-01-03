#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#define block_size 16
#define alphabet_size 26
#define DEBUG 0
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
            Dist[i*(n+1)+col] = Dist[(i-1)*(n+1)+(col-1)];
        else if (X[(P[i-1]-int('A'))*(n+1) + col] == 0)
            Dist[i*(n+1)+col] = 1 + min(Dist[(i-1)*(n+1)+col], min(Dist[(i-1)*(n+1)+(col-1)], i+col-1));
        else                                                                                   // D[i-1, X[l,j]-1] + (j - 1 - X[l,j])
            Dist[i*(n+1)+col] = 1 + min(min(Dist[(i-1)*(n+1)+col], Dist[(i-1)*(n+1)+(col-1)]), Dist[(i-1)*(n+1) + X[(P[i-1]-int('A'))*(n+1) + col] - 1] + (col-1-X[(P[i-1]-int('A'))*(n+1) + col]));
        __syncthreads();
    }
}

int main(int argc, char **argv) {

    assert(argc == 4);

    char *in_T = argv[1], *in_P = argv[2], *out = argv[3];
    FILE *input_T, *input_P, *output;
    input_T = fopen(in_T, "rb");
    input_P = fopen(in_P, "rb");

    int n, m;
    int *T, *P, *device_T, *device_P;

    fread(&n, 1, sizeof(int), input_T);
    fread(&m, 1, sizeof(int), input_P);

    T = (int*)malloc(sizeof(int)*n);
    P = (int*)malloc(sizeof(int)*m);
    // cudaMallocHost(&T, sizeof(int)*n);
    // cudaMallocHost(&P, sizeof(int)*m);

    fread(T, n, sizeof(int), input_T);
    fread(P, m, sizeof(int), input_P);
    # if DEBUG
    printf("n = %d, m = %d\n", n, m);
    for (int i = 0; i < n; i++)
        printf("%d ", T[i]);
    printf("\n");
    for (int i = 0; i < m; i++)
        printf("%d ", P[i]);
    printf("\n");
    # endif
    fclose(input_T);
    fclose(input_P);

    int *host_Dist, *device_Dist, *host_X, *device_X;

    // cudaMallocHost(&host_X, sizeof(int)*(n+1)*alphabet_size);
    // cudaMallocHost(&host_Dist, sizeof(int)*(n+1)*(m+1));
    host_X = (int*)malloc(sizeof(int)*(n+1)*alphabet_size);
    host_Dist = (int*)malloc(sizeof(int)*(n+1)*(m+1));

    cudaMalloc((void**)&device_Dist, sizeof(int)*(n+1)*(m+1));
    cudaMalloc((void**)&device_X, sizeof(int)*(n+1)*alphabet_size);
    cudaMalloc((void**)&device_T, sizeof(int)*n);
    cudaMalloc((void**)&device_P, sizeof(int)*m);
    cudaMemcpy(device_T, T, sizeof(int)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_P, P, sizeof(int)*m, cudaMemcpyHostToDevice);
    // cudaMemcpy(device_Dist, host_Dist, sizeof(int)*n*m, cudaMemcpyHostToDevice);

    // dim3 compute_X_nblocks((n+1)/block_size+1, m/block_size+1);
    // dim3 compute_X_nthreads(block_size, block_size);

    compute_X <<< 1, alphabet_size >>> (device_X, device_T, n);
    compute_Dist <<< 1, n+1 >>> (device_Dist, device_X, device_T, device_P, n, m);

    cudaMemcpy(host_Dist, device_Dist, sizeof(int)*(n+1)*(m+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_X, device_X, sizeof(int)*(n+1)*alphabet_size, cudaMemcpyDeviceToHost);

    # if DEBUG
    for (int i = 0; i < alphabet_size; i++) {
        for (int j = 0; j < n+1; j++)
            printf("%d ", host_X[i*(n+1)+j]);
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < m+1; i++) {
        for (int j = 0; j < n+1; j++)
            printf("%d ", host_Dist[i*(n+1)+j]);
        printf("\n");
    }
    # endif

    output = fopen(out, "wb");
    fwrite(host_Dist, (n+1)*(m+1), sizeof(int), output);
    fclose(output);

    return 0;
}