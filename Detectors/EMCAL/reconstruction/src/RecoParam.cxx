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

#include <iostream>
#include "EMCALReconstruction/RecoParam.h"

O2ParamImpl(o2::emcal::RecoParam);

using namespace o2::emcal;

std::ostream& operator<<(std::ostream& stream, const o2::emcal::RecoParam& s)
{
  s.PrintStream(stream);
  return stream;
}

void RecoParam::PrintStream(std::ostream& stream) const
{
  stream << "EMCAL Reconstruction parameters:"
         << "\n============================================="
         << "\nTime offset (ns):                 " << mCellTimeShiftNanoSec
         << "\nNoise threshold HGLG suppression: " << mNoiseThresholdLGnoHG
         << "\nPhase in BC mod 4 correction:     " << mPhaseBCmod4;
}