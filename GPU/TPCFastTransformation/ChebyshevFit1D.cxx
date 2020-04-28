// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  ChebyshevFit1D.cxx
/// \brief Implementation of ChebyshevFit1D class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#include "ChebyshevFit1D.h"

using namespace GPUCA_NAMESPACE::gpu;

void ChebyshevFit1D::reset(int order, double xMin, double xMax)
{
  if (order < 0) {
    order = 0;
  }
  mN = order + 1;
  mXmin = xMin;
  mXscale = (xMax - xMin > 1.e-8) ? 2. / (xMax - mXmin) : 2;

  mA.resize(mN * mN);
  mB.resize(mN + 1);
  mC.resize(mN + 1);
  mT.resize(mN + 1);
  reset();
}

void ChebyshevFit1D::reset()
{
  for (int i = 0; i <= mN; i++) {
    mB[i] = 0.;
    mC[i] = 0.;
    mT[i] = 0.;
  }

  for (int i = 0; i < mN * mN; i++) {
    mA[i] = 0.;
  }
  mM = 0;
}
