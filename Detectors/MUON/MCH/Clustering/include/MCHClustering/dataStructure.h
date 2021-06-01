#ifndef _DATASTRUCT_H
#define _DATASTRUCT_H

#include "mathUtil.h"

typedef int PadIdx_t;

// theta
double* getVarX(double* theta, int K);
double* getVarY(double* theta, int K);
double* getMuX(double* theta, int K);
double* getMuY(double* theta, int K);
double* getW(double* theta, int K);
double* getMuAndW(double* theta, int K);
//
const double* getConstVarX(const double* theta, int K);
const double* getConstVarY(const double* theta, int K);
const double* getConstMuX(const double* theta, int K);
const double* getConstMuY(const double* theta, int K);
const double* getConstW(const double* theta, int K);
const double* getConstMuAndW(const double* theta, int K);
// xyDxy
double* getX(double* xyDxy, int N);
double* getY(double* xyDxy, int N);
double* getDX(double* xyDxy, int N);
double* getDY(double* xyDxy, int N);
//
const double* getConstX(const double* xyDxy, int N);
const double* getConstY(const double* xyDxy, int N);
const double* getConstDX(const double* xyDxy, int N);
const double* getConstDY(const double* xyDxy, int N);

// xySupInf
double* getXInf(double* xyInfSup, int N);
double* getYInf(double* xyInfSup, int N);
double* getXSup(double* xyInfSup, int N);
double* getYSup(double* xyInfSup, int N);
const double* getConstXInf(const double* xyInfSup, int N);
const double* getConstYInf(const double* xyInfSup, int N);
const double* getConstXSup(const double* xyInfSup, int N);
const double* getConstYSup(const double* xyInfSup, int N);

// copy
void copyTheta(const double* theta0, int K0, double* theta, int K1, int K);
void copyXYdXY(const double* xyDxy0, int N0, double* xyDxy, int N1, int N);

// Transformations
void xyDxyToxyInfSup(const double* xyDxy, int nxyDxy,
                     double* xyInfSup);
// Mask operations
void maskedCopyXYdXY(const double* xyDxy, int nxyDxy, const Mask_t* mask, int nMask,
                     double* xyDxyMasked, int nxyDxyMasked);

void maskedCopyToXYInfSup(const double* xyDxy, int ndxyDxy, const Mask_t* mask, int nMask,
                          double* xyDxyMasked, int ndxyDxyMasked);

void maskedCopyTheta(const double* theta, int K, const Mask_t* mask, int nMask, double* maskedTheta, int maskedK);

void printTheta(const char* str, const double* theta, int K);

void printXYdXY(const char* str, const double* xyDxy, int NMax, int N, const double* val1, const double* val2);

#endif // _DATASTRUCT_H