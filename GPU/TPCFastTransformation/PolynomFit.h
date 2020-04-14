// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  PolynomFit.h
/// \brief Definition of PolynomFit class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_POLYNOMFIT_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_POLYNOMFIT_H

#include "GPUCommonDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
///
///  The class PolynomFit allows one to fit polynomial coefficients c0..cN
///
///  with given measurements m_i, i=0, ... :
///
///  m_i = c0*f0_i + c1*f1_i + .. + cN*mN_i + measurement_error
///
class PolynomFit
{
 public:
  PolynomFit() : mN(0), mA(0), mB(0)
  {
  }

  PolynomFit(int nCoefficients) : mN(0), mA(0), mB(0)
  {
    reset(nCoefficients);
  }

  ~PolynomFit()
  {
    delete[] mA;
    delete[] mAi;
    delete[] mB;
  }

  static int invS(long double M[], int N);

  void reset(int nCoefficients = -1);

  template <typename T>
  void addMeasurement(T f[], T m);

  template <typename T>
  int fit(T Coefficients[], int nCoefficients = -1);

  int getNmeasurements() const { return mNmeasurements; }

 private:
  int mN;
  int mNmeasurements;
  long double* mA;
  long double* mAi;
  long double* mB;
};

template <typename T>
void PolynomFit::addMeasurement(T f[], T m)
{
  for (int i = 0; i < mN; i++) {
    for (int j = 0; j <= i; j++) {
      mA[i * (i + 1) / 2 + j] += ((long double)f[i]) * ((long double)f[j]);
    }
    mB[i] += m * (long double)f[i];
  }
  mNmeasurements++;
}

template <typename T>
int PolynomFit::fit(T Coefficients[], int nCoefficients)
{
  for (int i = 0; i < mN * (mN + 1) / 2; i++) {
    mAi[i] = mA[i];
  }
  int n = mN;
  if (nCoefficients >= 0 && nCoefficients < mN) {
    n = nCoefficients;
  }
  int ret = invS(mAi, n);
  for (int i = 0; i < n; i++) {
    long double s = 0;
    for (int j = 0; j <= i; j++)
      s += mAi[i * (i + 1) / 2 + j] * mB[j];
    for (int j = i + 1; j < n; j++)
      s += mAi[j * (j + 1) / 2 + i] * mB[j];

    Coefficients[i] = (T)s;
  }
  return ret;
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
