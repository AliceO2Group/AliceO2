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

/// \file FEEConfig.cxx
/// \brief Frontend electronics configuration values
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include <numeric>
#include <string_view>
#include <bitset>
#include "fmt/format.h"
#include "Framework/Logger.h"
#include "CommonUtils/StringUtils.h"
#include "TPCBase/FEEConfig.h"
#include "TPCBase/CRU.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/CRUCalibHelpers.h"

using namespace o2::utils;
using namespace o2::tpc;

bool CRUConfig::setValues(std::string_view cruData)
{
  const auto cruDataV = Str::tokenize(cruData.data(), ',');
  if (cruDataV.size() != CRUConfig::NConfigValues) {
    LOGP(warning, "Wrong number of CRU config values {}/{} in line {}", cruDataV.size(), NConfigValues, cruData);
    return false;
  }

  linkOn = static_cast<decltype(linkOn)>(std::stol(cruDataV[0]));
  cmcEnabled = static_cast<decltype(cmcEnabled)>(std::stol(cruDataV[1]));
  zsOffset = static_cast<decltype(zsOffset)>(std::stol(cruDataV[4]));
  itCorr0 = std::stof(cruDataV[5]);
  itfEnabled = static_cast<decltype(itfEnabled)>(std::stol(cruDataV[2]));
  zsEnabled = static_cast<decltype(zsEnabled)>(std::stol(cruDataV[3]));
  resyncEnabled = static_cast<decltype(resyncEnabled)>(std::stol(cruDataV[6]));

  return true;
}

const std::unordered_map<FEEConfig::Tags, const std::string> FEEConfig::TagNames{
  {Tags::Unspecified, "Unspecified"},
  {Tags::TestWithZS, "TestWithZS"},
  {Tags::Pedestals, "Pedestals"},
  {Tags::Pulser, "Pulser"},
  {Tags::Laser, "Laser"},
  {Tags::Cosmics, "Cosmics"},
  {Tags::Physics35sigma, "Physics35sigma"},
  {Tags::Physics30sigma, "Physics30sigma"},
  {Tags::Physics25sigma, "Physics25sigma"},
  {Tags::Laser10ADCoff, "Laser10ADCoff"},
};

const std::unordered_map<FEEConfig::PadConfig, const std::string> FEEConfig::PadConfigNames{
  {PadConfig::ITfraction, "ITfraction"},
  {PadConfig::ITexpLambda, "ITexpLambda"},
  {PadConfig::CMkValues, "CMkValues"},
  {PadConfig::ThresholdMap, "ThresholdMap"},
  {PadConfig::Pedestals, "Pedestals"},
};

size_t FEEConfig::getNumberActiveLinks() const
{
  return std::accumulate(cruConfig.begin(), cruConfig.end(), size_t(0),
                         [](size_t sum, const auto& c) {
                           return sum + std::bitset<32>(c.linkOn).count();
                         });
}

bool FEEConfig::isCMCEnabled() const
{
  const auto nEnabled = std::accumulate(cruConfig.begin(), cruConfig.end(), size_t(0), [](size_t b, const auto& c) { return b + (c.cmcEnabled > 0); });
  if ((nEnabled > 0) && (nEnabled != cruConfig.size())) {
    LOGP(warning, "CMC not enabled for all CRUs: {} < {}", nEnabled, cruConfig.size());
  }

  return nEnabled > (cruConfig.size() / 2);
}

bool FEEConfig::isITFEnabled() const
{
  const auto nEnabled = std::accumulate(cruConfig.begin(), cruConfig.end(), size_t(0), [](size_t b, const auto& c) { return b + (c.itfEnabled); });
  if ((nEnabled > 0) && (nEnabled != cruConfig.size())) {
    LOGP(warning, "ITF not enabled for all CRUs: {} < {}", nEnabled, cruConfig.size());
  }

  return nEnabled > (cruConfig.size() / 2);
}

bool FEEConfig::isZSEnabled() const
{
  const auto nEnabled = std::accumulate(cruConfig.begin(), cruConfig.end(), size_t(0), [](size_t b, const auto& c) { return b + (c.zsEnabled); });
  if ((nEnabled > 0) && (nEnabled != cruConfig.size())) {
    LOGP(warning, "ZS not enabled for all CRUs: {} < {}", nEnabled, cruConfig.size());
  }

  return nEnabled > (cruConfig.size() / 2);
}

bool FEEConfig::isResyncEnabled() const
{
  const auto nEnabled = std::accumulate(cruConfig.begin(), cruConfig.end(), size_t(0), [](size_t b, const auto& c) { return b + (c.resyncEnabled); });
  if ((nEnabled > 0) && (nEnabled != cruConfig.size())) {
    LOGP(warning, "Resync not enabled for all CRUs: {} < {}", nEnabled, cruConfig.size());
  }

  return nEnabled > (cruConfig.size() / 2);
}

void FEEConfig::setAllLinksOn()
{
  const auto& mapper = Mapper::instance();

  // ===| Check active link map |===
  for (int iCRU = 0; iCRU < cruConfig.size(); ++iCRU) {
    const CRU cru(iCRU);
    const PartitionInfo& partInfo = mapper.getMapPartitionInfo()[cru.partition()];
    const int nFECs = partInfo.getNumberOfFECs();
    const int fecOffset = (nFECs + 1) / 2;
    cruConfig.at(iCRU).linkOn = 0;
    for (int iFEC = 0; iFEC < nFECs; ++iFEC) {
      const int fecTest = (iFEC < fecOffset) ? iFEC : 10 + (iFEC - fecOffset);
      const int fecBIT = 1 << fecTest;
      cruConfig.at(iCRU).linkOn |= fecBIT;
    }
  }
}

void FEEConfig::print() const
{
  fmt::print("\n");
  const auto& mapper = Mapper::instance();

  auto message = fmt::format("Printing tag {} ({})", int(tag), TagNames.at(tag));
  const size_t boxWidth = 80;
  fmt::print(
    "┌{0:─^{2}}┐\n"
    "│{1: ^{2}}│\n"
    "└{0:─^{2}}┘\n",
    "", message, boxWidth);
  fmt::print("\n");

  // ===| CRU summary |=========================================================
  message = "| CRU summary |";
  fmt::print("{0:=^{1}}\n", message, boxWidth);

  if (cruConfig.size() != CRU::MaxCRU) {
    LOGP(error, "Unexpected size of cru config:{} != {}", cruConfig.size(), (int)CRU::MaxCRU);
  } else {
    for (int iCRU = 0; iCRU < cruConfig.size(); ++iCRU) {
      const auto& c = cruConfig.at(iCRU);
      const CRU cru(iCRU);
      const PartitionInfo& partInfo = mapper.getMapPartitionInfo()[cru.partition()];
      const int nLinks = partInfo.getNumberOfFECs();
      const auto nLinkOn = std::bitset<32>(c.linkOn).count();

      fmt::print("CRU {:3d}: linkOn = {:5x} ({:2}/{:2}), cmcEn = {:5x}, zsOffset = {:5x}, itCorr0 = {:4.2f}, itfEn = {:b}, zsEn = {:b}, resyncEn = {:b}\n",
                 iCRU, c.linkOn, nLinkOn, nLinks, c.cmcEnabled, c.zsOffset, c.itCorr0, c.itfEnabled, c.zsEnabled, c.resyncEnabled);
    }
  }

  fmt::print("\n");
  // ===| CRU summary |=========================================================
  message = "| Pad maps summary |";
  fmt::print("{0:=^{1}}\n", message, boxWidth);

  for ([[maybe_unused]] auto& [key, val] : padMaps) {
    fmt::print("{}\n", key);
  }
}

void FEEConfig::printShort() const
{
  LOGP(info, "FEEConfig: tag: {}, #active links: {}, CMC enabled: {}, ITF enabled: {}, ZS enabled: {}, resync enabled: {}",
       (int)tag, getNumberActiveLinks(), isCMCEnabled(), isITFEnabled(), isZSEnabled(), isResyncEnabled());
}

CalDet<bool> FEEConfig::getDeadChannelMap() const
{
  const auto& mapper = Mapper::instance();
  CalDet<bool> deadMap("DeadChannelMap");

  // ===| Check active link map |===
  for (int iCRU = 0; iCRU < cruConfig.size(); ++iCRU) {
    const CRU cru(iCRU);
    const PartitionInfo& partInfo = mapper.getMapPartitionInfo()[cru.partition()];
    const int nFECs = partInfo.getNumberOfFECs();
    const int fecOffset = (nFECs + 1) / 2;
    for (int iFEC = 0; iFEC < nFECs; ++iFEC) {
      // const int fecInPartition = (iFEC < fecOffset) ? iFEC : 12 + (iFEC % 10);
      const int fecTest = (iFEC < fecOffset) ? iFEC : 10 + (iFEC - fecOffset);
      const int fecBIT = 1 << fecTest;
      if (cruConfig.at(iCRU).linkOn & fecBIT) {
        continue; // FEC is active
      }

      // all channels of the FEC are masked
      for (int iChannel = 0; iChannel < 80; ++iChannel) {
        const auto& [sampaOnFEC, channelOnSAMPA] = cru_calib_helpers::getSampaInfo(iChannel, cru);
        const PadROCPos padROCPos = mapper.padROCPos(cru, iFEC, sampaOnFEC, channelOnSAMPA);
        deadMap.getCalArray(padROCPos.getROC()).setValue(padROCPos.getRow(), padROCPos.getPad(), true);
      }
    }
  }

  // ===| Check pedestal values |===============================================
  const auto& pedestals = padMaps.at(PadConfigNames.at(PadConfig::Pedestals));

  for (int iROC = 0; iROC < pedestals.getData().size(); ++iROC) {
    const auto& rocPed = pedestals.getCalArray(iROC);
    auto& rocDead = deadMap.getCalArray(iROC);

    for (int iPad = 0; iPad < rocPed.getData().size(); ++iPad) {
      if (rocPed.getValue(iPad) > 1022) {
        rocDead.setValue(iPad, true);
      }
    }
  }

  return deadMap;
}
