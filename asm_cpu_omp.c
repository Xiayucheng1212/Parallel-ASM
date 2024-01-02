#include <stdlib.h>
#include <stdio.h>
#include <string.h>
# define DEBUG 0

// ASM algorithm DP
const int POSSIBLE_NUM = 26;
char * POSSIBLE_CHAR = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
int** D_matrix, **X_matrix;
int* T, *P;
int m, n;

int min(int a, int b) {
    if (a < b) return a;
    else return b;
}

void asm_parallel() {
    # pragma omp parallel for
    for (int i = 0; i < POSSIBLE_NUM; i++) {
        for (int j = 0; j <= n; j++) {
            if (j == 0) X_matrix[i][j] = 0;
            else if(T[j-1] == (int)POSSIBLE_CHAR[i]) X_matrix[i][j] = j;
            else X_matrix[i][j] = X_matrix[i][j-1];
        }
    }

    for (int i = 0; i <= m; i++){
        # pragma omp parallel for
        for (int j = 0; j <= n; j++){
            int l = P[i-1] - (int)'A'; // need check
            int tmp = min(D_matrix[i-1][j-1], D_matrix[i-1][j]);
            
            if (i == 0) D_matrix[i][j] = 0; 
            else if (j == 0) D_matrix[i][j] = i;
            else if (T[j-1] == P[i-1]) D_matrix[i][j] = D_matrix[i-1][j-1];
            else if (X_matrix[l][j] == 0) D_matrix[i][j] = 1 + min(tmp, i+j-1);
            else D_matrix[i][j] = 1 + min(tmp, D_matrix[i-1][X_matrix[l][j]-1] + j - 1 - X_matrix[l][j]);
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

    fclose(input_T);
    fclose(input_P);

    D_matrix = (int**)malloc(sizeof(int*)*(m+1));
    for (int i = 0; i <= m; i++){
        D_matrix[i] = (int*)malloc(sizeof(int)*(n+1));
        memset(D_matrix[i], 0, sizeof(int)*(n+1));
    }
    
    X_matrix = (int**)malloc(sizeof(int*)*(POSSIBLE_NUM));
    for (int i = 0; i < POSSIBLE_NUM; i++){
        X_matrix[i] = (int*)malloc(sizeof(int)*(n+1));
        memset(X_matrix[i], 0, sizeof(int)*(n+1));
    }

    asm_parallel();

    output = fopen(out, "wb");
    for (int i = 0; i <= m; i++)
        fwrite(D_matrix[i], n+1, sizeof(int), output);

    # if DEBUG
    for (int i = 0; i < POSSIBLE_NUM; i++){
        for (int j = 0; j <= n; j++){
            printf("%d ", X_matrix[i][j]);
        }
        printf("\n");
    }

    for (int i = 0; i <= m; i++){
        for (int j = 0; j <= n; j++){
            printf("%d ", D_matrix[i][j]);
        }
        printf("\n");
    }
    # endif

    fclose(output);

    return 0;
}