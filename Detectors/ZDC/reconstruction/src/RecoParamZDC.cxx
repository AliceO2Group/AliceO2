// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/Logger.h"
#include "ZDCReconstruction/RecoParamZDC.h"

O2ParamImpl(o2::zdc::RecoParamZDC);

void o2::zdc::RecoParamZDC::setBit(uint32_t ibit, bool val)
{
  if (ibit >= 0 && ibit < NTDCChannels) {
    bitset[ibit] = val;
  } else {
    LOG(FATAL) << __func__ << " bit " << ibit << " not in allowed range";
  }
}
