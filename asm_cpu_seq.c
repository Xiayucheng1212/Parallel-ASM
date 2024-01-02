#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// ASM algorithm DP
int** Dist;
int* T, *P;
int m, n;

int min(int a, int b) {
    if (a < b) return a;
    else return b;
}

void asm_seq() {
    for( int i = 0; i <= m ; i++ ) {
        for( int j = 0 ; j <= n; j++ ) {
            if (i == 0) continue;
            if (j == 0) Dist[i][j] = i;
            else if (T[j-1] == P[i-1]) Dist[i][j] = Dist[i-1][j-1];
            else {
                Dist[i][j] = 1 + min(min(Dist[i-1][j-1], Dist[i-1][j]),  Dist[i][j-1]);
            }
        }
    }
}


int main (int argc, char *argv[]) {

    char *in_T = argv[1], *in_P = argv[2], *out = argv[3];
    FILE *input_T, *input_P, *output;
    input_T = fopen(in_T, "rb");
    input_P = fopen(in_P, "rb");

    fread(&n, 1, sizeof(int), input_T);
    fread(&m, 1, sizeof(int), input_P);

    T = (int*)malloc(sizeof(int)*n);
    P = (int*)malloc(sizeof(int)*m);

    fread(T, n, sizeof(int), input_T);
    fread(P, m, sizeof(int), input_P);

    for(int i = 0; i < n; i++){
        printf("%d ", T[i]);
    }
    printf("\n");
    for(int i = 0; i < m; i++){
        printf("%d ", P[i]);
    }
    printf("\n");
    

    fclose(input_T);
    fclose(input_P);

    Dist = (int**)malloc(sizeof(int*)*(m+1));
    for (int i = 0; i <= m; i++){
        Dist[i] = (int*)malloc(sizeof(int)*(n+1));
        memset(Dist[i], 0, sizeof(int)*(n+1));
    }

    asm_seq();

    output = fopen(out, "wb");
    for (int i = 0; i <= m; i++)
        fwrite(Dist[i], n+1, sizeof(int), output);

    for(int i = 0; i <= m; i++){
        for(int j = 0; j <= n; j++){
            printf("%d ", Dist[i][j]);
        }
        printf("\n");
    }

    fclose(output);

    return 0;
}