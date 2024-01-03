#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#define alphabet_size 26
#define num_block 256
#define num_thread 64
#define WARPSIZE 32
#define DEBUG 0

/*
Optimized from GPU to blocked version
*/

__device__ char POSSIBLE_CHAR[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

// TODO: Consider using openMP with vectorization to parallelize this loop.
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
    int num_tile = 1;
    if (num_thread * num_block < (n+1))
        num_tile = (n+1) / (num_block*num_thread) + 1;
    int s_col = blockIdx.x * num_thread * num_tile + threadIdx.x, e_col = s_col + num_tile;
    for (int col = s_col; col < e_col; col++) {
        if (col > n)
            return;
        if (rd == 0) 
            Dist[rd*(n+1)+col] = 0;
        else if (col == 0)
            Dist[rd*(n+1)+col] = rd;
        else if (T[col-1] == P[rd-1])
            Dist[rd*(n+1)+col] = Dist[(rd-1)*(n+1)+(col-1)];
        else if (X[(P[rd-1]-int('A'))*(n+1) + col] == 0)
            Dist[rd*(n+1)+col] = 1 + min(Dist[(rd-1)*(n+1)+col], min(Dist[(rd-1)*(n+1)+(col-1)], rd + col - 1));
        else
            Dist[rd*(n+1)+col] = 1 + min(min(Dist[(rd-1)*(n+1)+col], Dist[(rd-1)*(n+1)+(col-1)]), Dist[(rd-1)*(n+1) + X[(P[rd-1]-int('A'))*(n+1) + col] - 1] + (col-1-X[(P[rd-1]-int('A'))*(n+1) + col]));
    }
}

__global__ void compute_Dist_with_shuffle(int *Dist, int *X, int *T, int *P, int n, int m, int rd) {
    // int col = blockIdx.x * num_thread + threadIdx.x;
    // tile up
    int num_tile = 1;
    if (num_thread * num_block < (n+1))
        num_tile = (n+1) / (num_block*num_thread) + 1;
    int s_col = blockIdx.x * num_thread + threadIdx.x;
    for (int i = 0; i < num_tile; i++) {
        int col = s_col + num_block*num_thread*i;
        if (col > n || rd == 0)
            return;

        int Dvar = Dist[(rd-1)*(n+1) + col], Avar, Bvar, Cvar;

        if (col % WARPSIZE == 0) // edge between two warps, cannot use shuffle across warps
            Avar = Dist[(rd-1)*(n+1) + col - 1];
        else {
            int test = __shfl_up_sync(0xffffffff, Dvar, 1);
            // TODO: need explain why this is needed
            if (col % WARPSIZE == 1) {
                Avar = Dist[(rd-1)*(n+1) + col - 1];
            }
            else Avar = test;
        }

        Bvar = Dvar; // D[i-1][j]
        Cvar = Dist[(rd-1)*(n+1) + X[(P[rd-1]-int('A'))*(n+1) + col] - 1]; // D[i-1][X[l][j]-1]

        // compute D[i][j] in local memory
        if (col == 0) Dvar = rd;
        else if (T[col-1] == P[rd-1]) Dvar = Avar;
        else if (X[(P[rd-1]-int('A'))*(n+1) + col] == 0) Dvar = 1 + min(Avar, min(Bvar, rd + col - 1));
        else Dvar = 1 + min( min(Avar, Bvar), Cvar + (col-1-X[(P[rd-1]-int('A'))*(n+1) + col]));
        
        Dist[rd*(n+1)+col] = Dvar; // write back to global memory
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

    compute_X <<< 1, alphabet_size >>> (device_X, device_T, n);

    int nblocks = min((n+1)/num_thread+1, num_block);
    for (int i = 0; i <= m; i++){
        compute_Dist_with_shuffle <<< nblocks, num_thread >>> (device_Dist, device_X, device_T, device_P, n, m, i);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(host_Dist, device_Dist, sizeof(int)*(n+1)*(m+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_X, device_X, sizeof(int)*(n+1)*alphabet_size, cudaMemcpyDeviceToHost);

    # if DEBUG
    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++)
            printf("%d ", host_Dist[i*(n+1)+j]);
        printf("\n");
    }
    # endif

    output = fopen(out, "wb");
    fwrite(host_Dist, (n+1)*(m+1), sizeof(int), output);
    fclose(output);

    return 0;
}