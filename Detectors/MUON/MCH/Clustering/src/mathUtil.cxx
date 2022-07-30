// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <cstdio>

#include "mathUtil.h"

namespace o2
{
namespace mch
{
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

void vectorPrint2Columns(const char* str, const double* x, const double* y,
                         int K)
{
  for (int k = 0; k < K; k++) {
    printf("%s", str);
    printf(" %6.3g %6.3g\n", x[k], y[k]);
    printf("\n");
  }
}

void printMatrixInt(const char* str, const int* matrix, int N, int M)
{
  printf("%s\n", str);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      printf(" %2d", matrix[i * M + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void printMatrixShort(const char* str, const short* matrix, int N, int M)
{
  printf("%s\n", str);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      printf(" %2d", matrix[i * M + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void printMatrixChar(const char* str, const char* matrix, int N, int M)
{
  printf("%s\n", str);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      printf(" %2d", matrix[i * M + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void printInterMap(const char* str, const PadIdx_t* inter, int N)
{
  const PadIdx_t* ij_ptr = inter;
  printf("%s\n", str);
  for (PadIdx_t i = 0; i < N; i++) {
    printf("row/col %d:", i);
    for (int k = 0; *ij_ptr != -1; k++, ij_ptr++) {
      printf(" %2d", *ij_ptr);
    }
    // skip -1, row/col change
    ij_ptr++;
    printf("\n");
  }
  printf("\n");
}

} // namespace mch
} // namespace o2