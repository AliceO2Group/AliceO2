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

// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright
// holders. All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_MATHIESON_H
#define O2_MCH_MATHIESON_H

#include <cmath>
#include <cstddef>

#include "MCHClustering/PadsPEM.h"

namespace o2
{
namespace mch
{
typedef struct SplineCoef {
  double* a;
  double* b;
  double* c;
  double* d;
  SplineCoef(int N);
  ~SplineCoef();
} SplineCoef_t;

typedef struct {
  // Compressed values along X and Y
  int nYc;
  int nXc;
  // Mapping : original array -> compressed array
  int* mapXInf;
  int* mapXSup;
  int* mapYInf;
  int* mapYSup;
  double* xCompressed;
  double* yCompressed;
} CompressedPads_t;

void initMathieson(int useSpline_, int useCache_);
void mathiesonPrimitive(const double* xy, int N,
                        int axe, int chamberId, double mPrimitive[]);
void initSplineMathiesonPrimitive();
void computeSplineCoef(const double* xy, double xyStep, const double* f, int N,
                       double leftDerivative, double rightDerivative, SplineCoef* splineCoef);
void splineMathiesonPrimitive(const double* x, int N, int axe, int chamberId, double* mPrimitive);

void compute1DMathieson(const double* xy, int N,
                        int axe, int chamberId, double mathieson[]);
void compute1DPadIntegrals(const double* xyInf, const double* xySup, int N,
                           double xy0, int axe, int chamberId, double* integrals);
void compute1DPadIntegrals(const double* xInf, const double* xSup, int N,
                           int axe, int chamberId, double Integrals[]);

void compute2DPadIntegrals(const double* xInf, const double* xSup,
                           const double* yInf, const double* ySup, int N,
                           int chamberId, double Integrals[]);

CompressedPads_t* compressPads(const double* xInf, const double* xSup,
                               const double* yInf, const double* ySup, int N);
void computeCompressed2DPadIntegrals(CompressedPads_t* compressedPads, double xShift, double yShift, int N,
                                     int chamberId, double Integrals[]);
void deleteCompressedPads(CompressedPads_t* compressedPads);

void compute2DMathiesonMixturePadIntegrals(const double* xyInfSup0,
                                           const double* theta, int N, int K,
                                           int chamberId, double Integrals[]);

void computeCij(const Pads& pads, const Pads& theta, double Cij[]);
void computeFastCij(const Pads& pads, const Pads& theta, double Cij[]);
void checkCij(const Pads& pads, const Pads& pixels, const double* checkCij, int mode);

bool checkIntegrals(const double* xInf, const double* xSup, const double* yInf, const double* ySup,
                    const double* integralsToCheck, int chId, int N);

// Utilities to handle mixture of parameter theta
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
void xyDxyToxyInfSup(const double* xyDxy, int nxyDxy, double* xyInfSup);
// Mask operations
void maskedCopyXYdXY(const double* xyDxy, int nxyDxy, const Mask_t* mask,
                     int nMask, double* xyDxyMasked, int nxyDxyMasked);

void maskedCopyToXYInfSup(const double* xyDxy, int ndxyDxy, const Mask_t* mask,
                          int nMask, double* xyDxyMasked, int ndxyDxyMasked);

void maskedCopyTheta(const double* theta, int K, const Mask_t* mask, int nMask,
                     double* maskedTheta, int maskedK);

void printTheta(const char* str, double meanCharge, const double* theta, int K);

void printXYdXY(const char* str, const double* xyDxy, int NMax, int N,
                const double* val1, const double* val2);

} // namespace mch
} // namespace o2

// For InspectModel (if required)
extern "C" {
void o2_mch_initMathieson();
void o2_mch_compute2DPadIntegrals(const double* xInf, const double* xSup,
                                  const double* yInf, const double* ySup, int N,
                                  int chamberId, double Integrals[]);
void o2_mch_compute2DMathiesonMixturePadIntegrals(const double* xyInfSup0,
                                                  const double* theta, int N,
                                                  int K, int chamberId,
                                                  double Integrals[]);
void o2_mch_computeCij(const double* xyInfSup0, const double* theta, int N,
                       int K, int chamberId, double Cij[]);
}

#endif // O2_MCH_MATHIESON_H_
