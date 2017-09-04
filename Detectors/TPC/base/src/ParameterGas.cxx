// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ParameterGas.cxx
/// \brief Implementation of the parameter class for the detector gas
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCBase/ParameterGas.h"

using namespace o2::TPC;

ParameterGas::ParameterGas()
  : mWion(0.f),
    mIpot(0.f),
    mAttCoeff(0.f),
    mOxyCont(0.f),
    mDriftV(0.f),
    mSigmaOverMu(0.f),
    mDiffT(0.f),
    mDiffL(0.f),
    mNprim(0.f),
    mScaleFactorG4(0.f),
    mFanoFactorG4(0.f),
    mBetheBlochParam()
{}

void ParameterGas::setDefaultValues()
{
  mWion = 37.3e-9f;
  mIpot = 20.77e-9f;
  mAttCoeff = 250.f;
  mOxyCont = 5.e-6f;
  mDriftV = 2.58f;
  mSigmaOverMu = 0.78f;
  mDiffT = 0.0209f;
  mDiffL = 0.0221f;
  mNprim = 14.f;
  mScaleFactorG4 = 0.85f;
  mFanoFactorG4 = 0.7f;
  mBetheBlochParam = {{0.76176e-1f, 10.632f, 0.13279e-4f, 1.8631f, 1.9479f}};
}
