// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file RUDecodeData.cxx
/// \brief Definition of the Readout Unite decoder class

///======================================================================
///                 REDOUT UNIT data decoding class
///======================================================================

#include "Framework/Logger.h"
#include "ITSMFTReconstruction/GBTWord.h"
#include "ITSMFTReconstruction/GBTLink.h"
#include "ITSMFTReconstruction/RUDecodeData.h"
#include "ITSMFTReconstruction/RUInfo.h"
#include "ITSMFTReconstruction/RUDecodeData.h"

namespace o2
{
namespace itsmft
{

///_________________________________________________________________
/// reset RU and its links
void RUDecodeData::clear()
{
  for (int i = nCables; i--;) {
    cableData[i].clear();
  }
  nCables = 0;
  nChipsFired = 0;
  calibData.clear();
}

///_________________________________________________________________
/// attach ROF-related info to fired chip. This method has to be in the cxx to avoid explicit
/// dependence of the RUDecodData on GBTLink class
void RUDecodeData::setROFInfo(ChipPixelData* chipData, const GBTLink* lnk)
{
  chipData->setTrigger(lnk->trigger);
  chipData->setInteractionRecord(lnk->ir);
}

///_________________________________________________________________
/// fill chip decoding statistics
void RUDecodeData::fillChipStatistics(int icab, const ChipPixelData* chipData)
{
  cableLinkPtr[icab]->chipStat.nHits += chipData->getData().size();
  cableLinkPtr[icab]->chipStat.addErrors(chipData->getErrorFlags(), chipData->getChipID());
}

} // namespace itsmft
} // namespace o2
