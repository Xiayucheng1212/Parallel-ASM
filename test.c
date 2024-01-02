#include <stdlib.h>
#include <stdio.h>
#include <string.h>

char POSSIBLE_CHAR[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

// Test case generator for ASM algorithm

int main (int argc, char *argv[]) {
    char *name = argv[1];
    int T_size = atoi(argv[2]);
    int P_size = atoi(argv[3]);
    char *T_name, *P_name;
    T_name = (char*)malloc(sizeof(char)*(strlen(name)+3));
    P_name = (char*)malloc(sizeof(char)*(strlen(name)+3));
    strcpy(T_name, name);
    strcpy(P_name, name);
    strcat(T_name, "_T");
    strcat(P_name, "_P");
    int* T = (int*)malloc(sizeof(int)*T_size);
    int* P = (int*)malloc(sizeof(int)*P_size);
    
    for (int i =0 ; i < T_size ; i++) {
        int rand_idx = rand() % 26;
        T[i] = (int)POSSIBLE_CHAR[rand_idx];
    }

    for (int i = 0; i < P_size; i++) {
        int rand_idx = rand() % 26;
        P[i] = (int)POSSIBLE_CHAR[rand_idx];
    }

    FILE *output;
    output = fopen(T_name, "wb");
    fwrite(&T_size, 1, sizeof(int), output);
    fwrite(T, T_size, sizeof(int), output);
    fclose(output);
    FILE *output2;
    output2 = fopen(P_name, "wb");
    fwrite(&P_size, 1, sizeof(int), output2);
    fwrite(P, P_size, sizeof(int), output2);
    fclose(output2);

    return 0;
}