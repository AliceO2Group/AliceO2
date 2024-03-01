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

/// \file GPUTPCGMPolynomialField.cxx
/// \author Sergey Gorbunov, David Rohr

#include "GPUTPCGMPolynomialField.h"
using namespace GPUCA_NAMESPACE::gpu;

#if defined(GPUCA_ALIROOT_LIB) & !defined(GPUCA_GPUCODE)

#include "GPUCommonConstants.h"
#include <iostream>
#include <iomanip>
#include <limits>

using namespace std;

void GPUTPCGMPolynomialField::Print() const
{
  const double kCLight = gpu_common_constants::kCLight;
  typedef std::numeric_limits<float> flt;
  cout << std::scientific;
#if __cplusplus >= 201103L
  cout << std::setprecision(flt::max_digits10 + 2);
#endif
  cout << " nominal field " << mNominalBz << " [kG * (2.99792458E-4 GeV/c/kG/cm)]"
       << " == " << mNominalBz / kCLight << " [kG]" << endl;

  cout << " TpcBx[NTPCM] = { ";
  for (int i = 0; i < NTPCM; i++) {
    cout << mTpcBx[i];
    if (i < NTPCM - 1) {
      cout << ", ";
    } else {
      cout << " };" << endl;
    }
  }

  cout << " TpcBy[NTPCM] = { ";
  for (int i = 0; i < NTPCM; i++) {
    cout << mTpcBy[i];
    if (i < NTPCM - 1) {
      cout << ", ";
    } else {
      cout << " };" << endl;
    }
  }

  cout << " TpcBz[NTPCM] = { ";
  for (int i = 0; i < NTPCM; i++) {
    cout << mTpcBz[i];
    if (i < NTPCM - 1) {
      cout << ", ";
    } else {
      cout << " };" << endl;
    }
  }

  cout << "TRD field: \n"
       << endl;

  cout << " TrdBx[NTRDM] = { ";
  for (int i = 0; i < NTRDM; i++) {
    cout << mTrdBx[i];
    if (i < NTRDM - 1) {
      cout << ", ";
    } else {
      cout << " };" << endl;
    }
  }

  cout << " TrdBy[NTRDM] = { ";
  for (int i = 0; i < NTRDM; i++) {
    cout << mTrdBy[i];
    if (i < NTRDM - 1) {
      cout << ", ";
    } else {
      cout << " };" << endl;
    }
  }

  cout << " TrdBz[NTRDM] = { ";
  for (int i = 0; i < NTRDM; i++) {
    cout << mTrdBz[i];
    if (i < NTRDM - 1) {
      cout << ", ";
    } else {
      cout << " };" << endl;
    }
  }

  cout << "ITS field: \n"
       << endl;

  cout << " ItsBx[NITSM] = { ";
  for (int i = 0; i < NITSM; i++) {
    cout << mItsBx[i];
    if (i < NITSM - 1) {
      cout << ", ";
    } else {
      cout << " };" << endl;
    }
  }

  cout << " ItsBy[NITSM] = { ";
  for (int i = 0; i < NITSM; i++) {
    cout << mItsBy[i];
    if (i < NITSM - 1) {
      cout << ", ";
    } else {
      cout << " };" << endl;
    }
  }

  cout << " ItsBz[NITSM] = { ";
  for (int i = 0; i < NITSM; i++) {
    cout << mItsBz[i];
    if (i < NITSM - 1) {
      cout << ", ";
    } else {
      cout << " };" << endl;
    }
  }
}

#else

void GPUTPCGMPolynomialField::Print() const
{
  // do nothing
}

#endif
