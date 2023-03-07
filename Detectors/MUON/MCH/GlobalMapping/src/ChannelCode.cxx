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

#include "MCHGlobalMapping/ChannelCode.h"

#include "MCHConstants/DetectionElements.h"
#include "MCHGlobalMapping/DsIndex.h"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHRawElecMap/Mapper.h"
#include <fmt/format.h>

namespace o2::mch
{
ChannelCode::ChannelCode(uint16_t deId, uint16_t dePadIndex)
{
  auto deIndexOpt = constants::deId2DeIndex(deId);
  if (deIndexOpt == std::nullopt) {
    throw std::runtime_error(fmt::format("invalid deId {}", deId));
  }
  auto deIndex = deIndexOpt.value();
  const auto& seg = o2::mch::mapping::segmentation(deId);
  if (!seg.isValid(dePadIndex)) {
    throw std::runtime_error(fmt::format("invalid dePadIndex {} for deId {}",
                                         dePadIndex, deId));
  }
  auto dsId = seg.padDualSampaId(dePadIndex);
  o2::mch::raw::DsDetId dsDetId(deId, dsId);
  auto dsIndex = o2::mch::getDsIndex(dsDetId);
  uint8_t channel = seg.padDualSampaChannel(dePadIndex);

  static auto det2elec = raw::createDet2ElecMapper<raw::ElectronicMapperGenerated>();

  auto elec = det2elec(dsDetId);
  if (elec == std::nullopt) {
    throw std::runtime_error(fmt::format("could not get solar,elink for {}",
                                         raw::asString(dsDetId)));
  }
  uint16_t solarId = elec->solarId();
  auto solarIndex = raw::solarId2Index<raw::ElectronicMapperGenerated>(solarId);
  if (solarIndex == std::nullopt) {
    throw std::runtime_error(fmt::format("could not get index from solarId {}",
                                         solarId));
  }
  uint8_t elinkIndex = elec->elinkId();
  set(deIndex, dePadIndex, dsIndex, solarIndex.value(), elinkIndex, channel);
}

ChannelCode::ChannelCode(uint16_t solarId, uint8_t elinkId, uint8_t channel)
{
  auto solarIndexOpt = raw::solarId2Index<raw::ElectronicMapperGenerated>(solarId);
  if (solarIndexOpt == std::nullopt) {
    throw std::runtime_error(fmt::format("invalid solarId {}", solarId));
  }
  static auto elec2det = raw::createElec2DetMapper<raw::ElectronicMapperGenerated>();
  auto group = raw::groupFromElinkId(elinkId);
  auto index = raw::indexFromElinkId(elinkId);
  raw::DsElecId dsElecId{solarId, group.value(), index.value()};
  auto dsDetIdOpt = elec2det(dsElecId);
  if (dsDetIdOpt == std::nullopt) {
    throw std::runtime_error(fmt::format("invalid solarid {} elinkid {}",
                                         solarId, elinkId));
  }
  auto deId = dsDetIdOpt->deId();
  auto deIndexOpt = constants::deId2DeIndex(deId);
  if (deIndexOpt == std::nullopt) {
    throw std::runtime_error(fmt::format("invalid deId {}", deId));
  }
  const auto& seg = o2::mch::mapping::segmentation(deId);
  auto dsId = dsDetIdOpt->dsId();
  int dePadIndex = seg.findPadByFEE(dsId, channel);
  if (!seg.isValid(dePadIndex)) {
    throw std::runtime_error(fmt::format("invalid dePadIndex {} for deId {}",
                                         dePadIndex, deId));
  }
  auto dsIndex = o2::mch::getDsIndex(dsDetIdOpt.value());
  auto solarIndex = solarIndexOpt.value();
  set(deIndexOpt.value(), dePadIndex, dsIndex, solarIndex, elinkId, channel);
}

/** build a 64 bits integer from the various indices.
 *
 * - deIndex (0..155) -----------  8 bits -- left shift 52
 * - dePadIndex (0..28671) ------ 15 bits --            37
 * - DsIndex (0..16819) --------- 15 bits --            22
 * - solarIndex (0..623) -------- 10 bits --            12
 * - elinkId (0..39) ------------  6 bits --             6
 * - channel number (0..63) -----  6 bits --             0
 */
void ChannelCode::set(uint8_t deIndex,
                      uint16_t dePadIndex,
                      uint16_t dsIndex,
                      uint16_t solarIndex,
                      uint8_t elinkIndex,
                      uint8_t channel)
{
  mValue = (static_cast<uint64_t>(deIndex & 0xFF) << 52) +
           (static_cast<uint64_t>(dePadIndex & 0x7FFF) << 37) +
           (static_cast<uint64_t>(dsIndex & 0x7FFF) << 22) +
           (static_cast<uint64_t>(solarIndex & 0x3FF) << 12) +
           (static_cast<uint64_t>(elinkIndex & 0x3F) << 6) +
           (static_cast<uint64_t>(channel & 0x3F));
}

uint8_t ChannelCode::getDeIndex() const
{
  return static_cast<uint8_t>((mValue >> 52) & 0xFF);
}

uint16_t ChannelCode::getDePadIndex() const
{
  return static_cast<uint16_t>((mValue >> 37) & 0x7FFF);
}

uint16_t ChannelCode::getDsIndex() const
{
  return static_cast<uint16_t>((mValue >> 22) & 0x7FFF);
}

uint16_t ChannelCode::getSolarIndex() const
{
  return static_cast<uint16_t>((mValue >> 12) & 0x3FF);
}

uint8_t ChannelCode::getElinkId() const
{
  return static_cast<uint8_t>((mValue >> 6) & 0x3F);
}

uint8_t ChannelCode::getChannel() const
{
  return static_cast<uint8_t>(mValue & 0x3F);
}

uint16_t ChannelCode::getDeId() const
{
  auto deIndex = getDeIndex();
  return o2::mch::constants::deIdsForAllMCH[deIndex];
}

uint16_t ChannelCode::getDsId() const
{
  auto dsIndex = getDsIndex();
  o2::mch::raw::DsDetId dsDetId = getDsDetId(dsIndex);
  return dsDetId.dsId();
}

uint16_t ChannelCode::getSolarId() const
{
  auto solarIndex = getSolarIndex();
  return raw::solarIndex2Id<raw::ElectronicMapperGenerated>(solarIndex).value();
}

std::string asString(const ChannelCode& cc)
{
  return fmt::format("deid {:4d} dsid {:4d} ch {:2d} depadindex {:5d} solarid {:4d} elink {:2d}",
                     cc.getDeId(),
                     cc.getDsId(),
                     cc.getChannel(),
                     cc.getDePadIndex(),
                     cc.getSolarId(),
                     cc.getElinkId());
}
} // namespace o2::mch
