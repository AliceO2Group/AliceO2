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
