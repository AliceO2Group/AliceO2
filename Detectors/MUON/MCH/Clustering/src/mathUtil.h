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

#ifndef O2_MCH_MATHUTIL_H
#define O2_MCH_MATHUTIL_H

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>

#include "MCHClustering/PadsPEM.h"

namespace o2
{
namespace mch
{

inline static void vectorSetZero(double* u, int N)
{
  for (int i = 0; i < N; i++) {
    u[i] = 0;
  }
  return;
}

inline static void vectorSetZeroInt(int* u, int N)
{
  for (int i = 0; i < N; i++) {
    u[i] = 0;
  }
  return;
}

inline static void vectorSetZeroShort(short* u, int N)
{
  for (int i = 0; i < N; i++) {
    u[i] = 0;
  }
  return;
}

inline static void vectorSetZeroChar(char* u, int N)
{
  for (int i = 0; i < N; i++) {
    u[i] = 0;
  }
  return;
}

inline static void vectorSet(double* u, double value, int N)
{
  for (int i = 0; i < N; i++) {
    u[i] = value;
  }
  return;
}

inline static void vectorSetInt(int* u, int value, int N)
{
  for (int i = 0; i < N; i++) {
    u[i] = value;
  }
  return;
}

inline static void vectorSetShort(short* u, short value, int N)
{
  for (int i = 0; i < N; i++) {
    u[i] = value;
  }
  return;
}

inline static void vectorCopy(const double* src, int N, double* dest)
{
  for (int i = 0; i < N; i++) {
    dest[i] = src[i];
  }
  return;
}

inline static void vectorCopyShort(const short* src, int N, short* dest)
{
  for (int i = 0; i < N; i++) {
    dest[i] = src[i];
  }
  return;
}

inline static void vectorAddVector(const double* u, double cst, const double* v,
                                   int N, double* res)
{
  for (int i = 0; i < N; i++) {
    res[i] = u[i] + cst * v[i];
  }
  return;
}

inline static void vectorAbs(const double* u, int N, double* res)
{
  for (int i = 0; i < N; i++) {
    res[i] = fabs(u[i]);
  }
  return;
}

inline static void vectorAddScalar(const double* u, double cst, int N,
                                   double* res)
{
  for (int i = 0; i < N; i++) {
    res[i] = u[i] + cst;
  }
  return;
}

inline static void vectorMultVector(const double* u, const double* v, int N,
                                    double* res)
{
  for (int i = 0; i < N; i++) {
    res[i] = u[i] * v[i];
  }
  return;
}

inline static void vectorMultVectorShort(const short* u, const short* v, int N,
                                         short* res)
{
  for (int i = 0; i < N; i++) {
    res[i] = u[i] * v[i];
  }
  return;
}

inline static void vectorMultScalar(const double* u, double cst, int N,
                                    double* res)
{
  for (int i = 0; i < N; i++) {
    res[i] = u[i] * cst;
  }
  return;
}

inline static double vectorNorm(const double* u, int N)
{
  double res = 0;
  for (int i = 0; i < N; i++) {
    res += u[i] * u[i];
  }
  return sqrt(res);
}

inline static double vectorSum(const double* u, int N)
{
  double res = 0;
  for (int i = 0; i < N; i++) {
    res += u[i];
  }
  return res;
}

inline static int vectorSumInt(const int* u, int N)
{
  int res = 0;
  for (int i = 0; i < N; i++) {
    res += u[i];
  }
  return res;
}

inline static int vectorSumShort(const short* u, int N)
{
  int res = 0;
  for (int i = 0; i < N; i++) {
    res += u[i];
  }
  return res;
}

inline static int vectorSumChar(const char* u, int N)
{
  int res = 0;
  for (int i = 0; i < N; i++) {
    res += u[i];
  }
  return res;
}

inline static int vectorSumRowInt(const int* matrix, int N, int M)
{
  int res = 0;
  for (int j = 0; j < M; j++) {
    res += matrix[j];
  }
  return res;
}

inline static int vectorSumColumnInt(const int* matrix, int N, int M)
{
  int res = 0;
  for (int i = 0; i < N; i++) {
    res += matrix[i * M];
  }
  return res;
}

inline static int vectorSumRowChar(const char* matrix, int N, int M)
{
  int res = 0;
  for (int j = 0; j < M; j++) {
    res += matrix[j];
  }
  return res;
}

inline static int vectorSumColumnChar(const char* matrix, int N, int M)
{
  int res = 0;
  for (int i = 0; i < N; i++) {
    res += matrix[i * M];
  }
  return res;
}

inline static double vectorMin(const double* u, int N)
{
  double res = DBL_MAX;
  for (int i = 0; i < N; i++) {
    res = fmin(res, u[i]);
  }
  return res;
}

inline static double vectorMax(const double* u, int N)
{
  double res = -DBL_MAX;
  for (int i = 0; i < N; i++) {
    res = fmax(res, u[i]);
  }
  return res;
}

inline static short vectorMaxShort(const short* u, int N)
{
  short res = SHRT_MIN;
  for (int i = 0; i < N; i++) {
    res = std::max(res, u[i]);
  }
  return res;
}
//
// Logical operations
//
inline static void vectorNotShort(const short* src, int N, short* dest)
{
  for (int i = 0; i < N; i++) {
    dest[i] = !src[i];
  }
  return;
}

//
// Compare oparations
//
inline static int vectorSumOfGreater(const double* src, double cmpValue,
                                     int N)
{
  int count = 0;
  for (int i = 0; i < N; i++) {
    count += ((src[i] > cmpValue) ? 1 : 0);
  }
  return count;
}

inline static int vectorArgMax(const double* u, int N)
{
  int idx = -1;
  double maxVal = DBL_MIN;
  for (int i = 0; i < N; i++) {
    if (u[i] > maxVal) {
      maxVal = u[i];
      idx = i;
    }
  }
  return idx;
}

inline static int vectorArgMin(const double* u, int N)
{
  int idx = -1;
  double minVal = DBL_MAX;
  for (int i = 0; i < N; i++) {
    if (u[i] < minVal) {
      minVal = u[i];
      idx = i;
    }
  }
  return idx;
}

//
// Mask operations
//
inline static int vectorBuildMaskEqualShort(const short* src, short value,
                                            int N, short* mask)
{
  int count = 0;
  for (int i = 0; i < N; i++) {
    mask[i] = (src[i] == value);
    count += ((src[i] == value));
  }
  return count;
}

inline static void vectorBuildMaskGreater(const double* src, double value,
                                          int N, short* mask)
{
  for (int i = 0; i < N; i++) {
    mask[i] = (src[i] > value);
  }
  return;
}

inline static void vectorBuildMaskEqual(const double* src, double value, int N,
                                        short* mask)
{
  for (int i = 0; i < N; i++) {
    mask[i] = (src[i] == value);
  }
  return;
}

inline static int vectorGetIndexFromMaskInt(const Mask_t* mask, int N,
                                            int* indexVector)
{
  int k = 0;
  for (int i = 0; i < N; i++) {
    if (mask[i] == 1) {
      indexVector[i] = k;
      k++;
    }
  }
  return k;
}
// int k=0;for (int i=0; i < N; i++) { indexVector[i] = ( mask[i] == 1) ? k :
// -1; k++;} return k;}

inline static void vectorAppyMapIdxInt(const int* vect, const int* map, int N,
                                       int* mappedVect)
{
  for (int i = 0; i < N; i++) {
    mappedVect[i] = map[vect[i]];
  }
  return;
}

inline static int vectorGather(const double* v, const Mask_t* mask, int N,
                               double* gatherVector)
{
  int k = 0;
  for (int i = 0; i < N; i++) {
    if (mask[i]) {
      gatherVector[k++] = v[i];
    }
  }
  return k;
}

inline static int vectorScatter(const double* v, const Mask_t* mask, int N,
                                double* scatterVec)
{
  int k = 0;
  for (int i = 0; i < N; i++) {
    if (mask[i]) {
      scatterVec[i] = v[k++];
    }
  }
  return k;
}
inline static int vectorScatterShort(const short* v, const Mask_t* mask, int N,
                                     short* scatterVec)
{
  int k = 0;
  for (int i = 0; i < N; i++) {
    if (mask[i]) {
      scatterVec[i] = v[k++];
    }
  }
  return k;
}

inline static int vectorGatherShort(const short* v, const Mask_t* mask, int N,
                                    short* gatherVector)
{
  int k = 0;
  for (int i = 0; i < N; i++) {
    if (mask[i]) {
      gatherVector[k++] = v[i];
    }
  }
  return k;
}

inline static int vectorGetIndexFromMaskShort(const Mask_t* mask, int N,
                                              short* index)
{
  int k = 0;
  for (int i = 0; i < N; i++) {
    if (mask[i]) {
      index[k++] = i;
    }
  }
  return k;
}

inline static int vectorGetIndexFromMask(const Mask_t* mask, int N,
                                         int* index)
{
  int k = 0;
  for (int i = 0; i < N; i++) {
    if (mask[i]) {
      index[k++] = i;
    }
  }
  return k;
}

inline static void vectorMaskedSet(const Mask_t* mask, const double* vTrue,
                                   const double* vFalse, int N, double* vR)
{
  for (int i = 0; i < N; i++) {
    vR[i] = (mask[i]) ? vTrue[i] : vFalse[i];
  };
  return;
}

inline static void vectorMaskedUpdate(const Mask_t* mask, const double* vTrue,
                                      int N, double* vR)
{
  for (int i = 0; i < N; i++) {
    vR[i] = (mask[i]) ? vTrue[i] : vR[i];
  };
  return;
}

inline static double vectorMaskedSum(const double* v, const Mask_t* mask,
                                     int N)
{
  double sum = 0;
  for (int i = 0; i < N; i++) {
    sum += v[i] * mask[i];
  }
  return sum;
}

inline static void vectorMaskedMult(const double* v, const Mask_t* mask, int N,
                                    double* res)
{
  for (int i = 0; i < N; i++) {
    res[i] = v[i] * mask[i];
  }
  return;
}

inline static void vectorMaskedMultScalar(double* v, const Mask_t* mask,
                                          double trueVal, double falseVal,
                                          int N)
{
  for (int i = 0; i < N; i++) {
    v[i] = (mask[i]) ? v[i] * trueVal : v[i] * falseVal;
  }
  return;
}

inline static void vectorMapShort(short* array, const short* map, int N)
{
  for (int i = 0; i < N; i++) {
    array[i] = map[array[i]];
  }
}

inline void deleteDouble(double* ptr)
{
  if (ptr != nullptr) {
    delete[] ptr;
    ptr = nullptr;
  }
}

inline void deleteInt(int* ptr)
{
  if (ptr != nullptr) {
    delete[] ptr;
    ptr = nullptr;
  }
}

inline void deleteShort(short* ptr)
{
  if (ptr != nullptr) {
    delete[] ptr;
    ptr = nullptr;
  }
}

void vectorPrint(const char* str, const double* x, int N);
void vectorPrintInt(const char* str, const int* x, int N);
void vectorPrintShort(const char* str, const short* x, int N);
void vectorPrint2Columns(const char* str, const double* x, const double* y,
                         int N);

void printMatrixInt(const char* str, const int* matrix, int N, int M);
void printMatrixShort(const char* str, const short* matrix, int N, int M);
void printMatrixChar(const char* str, const char* matrix, int N, int M);

void printInterMap(const char* str, const PadIdx_t* inter, int N);

} // namespace mch
} // namespace o2

#endif // O2_MCH_MATHUTIL_H
