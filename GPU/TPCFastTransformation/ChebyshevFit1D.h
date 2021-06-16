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

/// \file  ChebyshevFit1D.h
/// \brief Definition of ChebyshevFit1D class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_CHEBYSHEVFIT1D_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_CHEBYSHEVFIT1D_H

#include "GPUCommonDef.h"
#include <vector>

namespace GPUCA_NAMESPACE
{
namespace gpu
{
///
///  The class ChebyshevFit1D allows one to fit a function with chebyshev polynomials
///  with given measurements m_i
///
class ChebyshevFit1D
{
 public:
  ChebyshevFit1D()
  {
    reset(0, -1., 1.);
  }

  ChebyshevFit1D(int order, double xMin, double xMax)
  {
    reset(order, xMin, xMax);
  }

  ~ChebyshevFit1D() CON_DEFAULT;

  void reset(int order, double xMin, double xMax);

  void reset();

  void addMeasurement(double x, double m);

  void fit();

  double eval(double x);

  int getNmeasurements() const { return mM; }

  const std::vector<double>& getCoefficients() const { return mC; }

  void print();

 private:
  int mN = 0;             // n coefficients == polynom order + 1
  int mM = 0;             // number of measurenents
  double mXmin = -1.;     // min of X segment
  double mXscale = 1;     // scaling factor (x-mXmin) to [-1,1]
  std::vector<double> mA; // fit matiix
  std::vector<double> mB; // fit vector
  std::vector<double> mC; // Chebyshev coefficients
  std::vector<double> mT; // Chebyshev coefficients
};

inline void ChebyshevFit1D::addMeasurement(double x, double m)
{
  x = -1. + (x - mXmin) * mXscale;
  mT[0] = 1;
  mT[1] = x;
  x *= 2.;
  for (int i = 2; i < mN; i++) {
    mT[i] = x * mT[i - 1] - mT[i - 2];
  }
  double* Ai = mA.data();
  for (int i = 0; i < mN; i++, Ai += mN) {
    for (int j = i; j < mN; j++) {
      Ai[j] += mT[i] * mT[j];
    }
    mB[i] += m * mT[i];
  }
  mM++;
}

inline double ChebyshevFit1D::eval(double x)
{
  x = -1. + (x - mXmin) * mXscale;
  double y = mC[0] + mC[1] * x;
  double f0 = 1.;
  double f1 = x;
  x *= 2;
  for (int i = 2; i < mN; i++) {
    double f = x * f1 - f0;
    y += mC[i] * f;
    f0 = f1;
    f1 = f;
  }
  return y;
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
