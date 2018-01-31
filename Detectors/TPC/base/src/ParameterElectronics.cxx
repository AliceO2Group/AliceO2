// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ParameterElectronics.cxx
/// \brief Implementation of the parameter class for the detector electronics
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCBase/ParameterElectronics.h"

using namespace o2::TPC;

ParameterElectronics::ParameterElectronics()
  : mNShapedPoints(0),
    mPeakingTime(0.f),
    mChipGain(0.f),
    mADCdynamicRange(0.f),
    mADCsaturation(0.f),
    mZbinWidth(0.f),
    mElectronCharge(0.f),
    mDigitizationMode(DigitzationMode::FullMode)
{}

void ParameterElectronics::setDefaultValues()
{
  mNShapedPoints = 8; ///< to fit into SSE registers
  mPeakingTime = 160e-3f;
  mChipGain = 20.f;
  mADCdynamicRange = 2200.f;
  mADCsaturation = 1024.f;
  mZbinWidth = 0.2f;
  mElectronCharge = 1.602e-19f;
  mDigitizationMode = DigitzationMode::FullMode;
}
