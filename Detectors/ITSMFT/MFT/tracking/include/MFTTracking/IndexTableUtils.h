// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file IndexTableUtils.h
/// \brief May collect the functions to place clusters into R-Phi bins (to be fixed)
///

#ifndef O2_MFT_INDEXTABLEUTILS_H_
#define O2_MFT_INDEXTABLEUTILS_H_

#include <array>
#include <utility>
#include <vector>

#include "MFTTracking/Constants.h"

namespace o2
{
namespace MFT
{

namespace IndexTableUtils
{
Int_t getRBinIndex(const Int_t, const Float_t);
Int_t getPhiBinIndex(const Float_t);
Int_t getBinIndex(const Int_t, const Int_t);
} // namespace IndexTableUtils

inline Int_t IndexTableUtils::getRBinIndex(const Int_t layerIndex, const Float_t rCoordinate)
{
  return -1;
}

inline Int_t IndexTableUtils::getPhiBinIndex(const Float_t currentPhi)
{
  return -1;
}

inline Int_t IndexTableUtils::getBinIndex(const Int_t rIndex, const Int_t phiIndex)
{
  return -1;
}
} // namespace MFT
} // namespace o2

#endif /* O2_MFT_INDEXTABLEUTILS_H_ */
