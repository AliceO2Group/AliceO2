// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/FEEIdConfig.h
/// \brief  Hardware Id to FeeId mapper
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   11 March 2020
#ifndef O2_MID_FEEIDCONFIG_H
#define O2_MID_FEEIDCONFIG_H

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace o2
{
namespace mid
{
class FEEIdConfig
{
 public:
  FEEIdConfig();
  FEEIdConfig(const char* filename);
  ~FEEIdConfig() = default;

  uint16_t getGBTUniqueId(uint32_t linkUniqueId) const;

  inline const std::vector<uint16_t>& getGBTUniqueIdsInLink(uint16_t feeId) const { return mGBTUniqueIdsInLink.find(feeId)->second; }

  /// Gets the FEE ID from the physical ID of the link
  uint16_t getGBTUniqueId(uint8_t linkId, uint8_t endPointId, uint16_t cruId) const { return getGBTUniqueId(getLinkUniqueId(linkId, endPointId, cruId)); }

  /// Gets a uniqueID from the combination of linkId, endPointId and cruId;
  inline uint32_t getLinkUniqueId(uint8_t linkId, uint8_t endPointId, uint16_t cruId) const { return (linkId + 1) << ((endPointId == 1) ? 8U : 0U) | (cruId << 16U); }

  /// Gets the CRU ID
  inline uint16_t getCRUId(uint32_t linkUniqueId) const { return linkUniqueId >> 16; }
  /// Gets the end point id
  inline uint8_t getEndPointId(uint32_t linkUniqueId) const { return (linkUniqueId & 0xFF00) ? 1 : 0; }
  /// Gets the Link ID
  inline uint8_t getLinkId(uint32_t linkUniqueId) const { return ((linkUniqueId >> (8U * getEndPointId(linkUniqueId))) & 0xFF) - 1; }

  std::vector<uint16_t> getConfiguredGBTUniqueIDs() const;
  std::vector<uint32_t> getConfiguredLinkUniqueIDs() const;
  std::vector<uint16_t> getConfiguredFEEIDs() const;

  void write(const char* filename) const;

 private:
  bool load(const char* filename);

  std::unordered_map<uint32_t, uint16_t> mLinkUniqueIdToGBTUniqueId{};       /// Correspondence between link unique ID and GBT unique Id
  std::unordered_map<uint16_t, uint16_t> mGBTUniqueIdToFeeId{};              /// Correspondence between GBT unique ID and FEE ID
  std::unordered_map<uint16_t, std::vector<uint16_t>> mGBTUniqueIdsInLink{}; /// Input GBT links in output link
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_FEEIDCONFIG_H */
