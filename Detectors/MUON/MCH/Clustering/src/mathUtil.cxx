#include <stdio.h>

#include "MCHClustering/mathUtil.h"

void vectorPrint(const char* str, const double* x, int K)
{
  printf("%s", str);
  for (int k = 0; k < K; k++) {
    printf(" %9.6g", x[k]);
  }
  printf("\n");
}

void vectorPrintInt(const char* str, const int* x, int K)
{
  printf("%s", str);
  for (int k = 0; k < K; k++) {
    printf(" %3d", x[k]);
  }
  printf("\n");
}

void vectorPrintShort(const char* str, const short* x, int K)
{
  printf("%s", str);
  for (int k = 0; k < K; k++) {
    printf(" %3d", x[k]);
  }
  printf("\n");
}

void vectorPrint2Columns(const char* str, const double* x, const double* y, int K)
{
  for (int k = 0; k < K; k++) {
    printf("%s", str);
    printf(" %6.3g %6.3g\n", x[k], y[k]);
    printf("\n");
  }
}
