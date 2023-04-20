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

/// \file RUDecodeData.cxx
/// \brief Definition of the Readout Unite decoder class

///======================================================================
///                 REDOUT UNIT data decoding class
///======================================================================

#include "Framework/Logger.h"
#include "ITSMFTReconstruction/GBTWord.h"
#include "ITSMFTReconstruction/GBTLink.h"
#include "ITSMFTReconstruction/DecodingStat.h"
#include "ITSMFTReconstruction/RUDecodeData.h"

namespace o2
{
namespace itsmft
{

///_________________________________________________________________
/// reset RU and its links
void RUDecodeData::clear()
{
  for (int i = ruInfo->nCables; i--;) {
    if (cableLinkPtr[i] && !cableLinkPtr[i]->rofJumpWasSeen) {
      cableData[i].clear();
    }
  }
  nChipsFired = 0;
  nNonEmptyLinks = 0;
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
  uint32_t action = 0;
  if (chipData->getErrorFlags()) {
    cableLinkPtr[icab]->chipStat.addErrors(*chipData, verbosity);
    auto compid = ChipError::composeID(cableLinkPtr[icab]->feeID, chipData->getChipID());
    auto& chErr = chipErrorsTF[compid];
    chErr.first++;
    chErr.second |= chipData->getErrorFlags();
  }
  if (action & ChipStat::ErrActDump) {
    linkHBFToDump[(uint64_t(cableLinkPtr[icab]->subSpec) << 32) + cableLinkPtr[icab]->hbfEntry] = cableLinkPtr[icab]->irHBF.orbit;
  }
}

///_________________________________________________________________
/// dump cable data for debugging
void RUDecodeData::dumpcabledata(int icab)
{
  const auto* cdat = cableData[icab].data();
  size_t cdats = cableData[icab].getUnusedSize(), cdi = 0;
  const auto lll = cableLinkPtr[icab];
  LOGP(info, "RU#{} Cab{} Nbytes: {} IR: {} | {}", ruSWID, icab, cdats, lll->ir.asString(), lll->describe());
  std::string dmp = "";
  while (cdi < cdats) {
    dmp += fmt::format(" {:#04x}", cdat[cdi++]);
    if (cdi && (cdi % 9) == 0) {
      LOGP(info, "wrd#{}: {}", cdi / 9 - 1, dmp);
      dmp = "";
    }
  }
  if (!dmp.empty()) {
    LOGP(info, "wrd#{}: {}", cdi / 9 - 1, dmp);
  }
}

///_________________________________________________________________
/// check if the link data is in sync with majority
bool RUDecodeData::checkLinkInSync(int icab, const o2::InteractionRecord ir)
{
  auto* link = cableLinkPtr[icab];
  if (link->ir == ir) {
    link->rofJumpWasSeen = false;
    return true;
  }
  if (link->statusInTF == GBTLink::CollectedDataStatus::None || link->statusInTF == GBTLink::CollectedDataStatus::StoppedOnEndOfData) { // link was not seen in this TF or data was over
    return true;
  }
  // apparently there was desynchronization
  if (link->ir > ir) {
    link->rofJumpWasSeen = true;
#ifdef _RAW_READER_ERROR_CHECKS_
    link->statistics.errorCounts[GBTLinkDecodingStat::ErrMissingROF]++;
    linkHBFToDump[(uint64_t(link->subSpec) << 32) + link->hbfEntry] = link->irHBF.orbit;
    if (link->needToPrintError(link->statistics.errorCounts[GBTLinkDecodingStat::ErrMissingROF])) {
      LOGP(info, "{} (cable {}) has IR={} > current majority IR={} -> {}", link->describe(),
           cableHWID[icab], link->ir.asString(), ir.asString(), link->statistics.ErrNames[GBTLinkDecodingStat::ErrMissingROF]);
    }
#endif
  } else { // link IR is behind the majority IR? In principle, this should never happen
#ifdef _RAW_READER_ERROR_CHECKS_
    link->statistics.errorCounts[GBTLinkDecodingStat::ErrOldROF]++;
    linkHBFToDump[(uint64_t(link->subSpec) << 32) + link->hbfEntry] = link->irHBF.orbit;
    if (link->needToPrintError(link->statistics.errorCounts[GBTLinkDecodingStat::ErrOldROF])) {
      LOGP(error, "{} (cable {}) has IR={} for current majority IR={} -> {}", link->describe(),
           cableHWID[icab], link->ir.asString(), ir.asString(), link->statistics.ErrNames[GBTLinkDecodingStat::ErrOldROF]);
    }
#endif
    // clean data of cables of this link
    for (int i = 0; i < ruInfo->nCables; i++) {
      if (cableLinkPtr[i] == link) {
        cableData[i].clear();
      }
    }
    link->rofJumpWasSeen = false;
  }
  return false;
}

} // namespace itsmft
} // namespace o2
