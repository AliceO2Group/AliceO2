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

#ifndef O2_QUALITYCONTROL_FLAGTYPE_H
#define O2_QUALITYCONTROL_FLAGTYPE_H

// STL
#include <iosfwd>
#include <cstdint>

// ROOT includes
#include <Rtypes.h>

namespace o2
{
namespace quality_control
{

class FlagTypeFactory;
class QualityControlFlagCollection;

class FlagType
{
 private:
  uint16_t mId;
  std::string mName;
  bool mBad; // if true, data should become bad by default

  // By making the constructor private and FlagTypes available only in the FlagTypeFactory
  // we forbid to declare any flags in the user code. If you need a new FlagType, please add it FlagTypeFactory.
 private:
  FlagType(uint16_t id, const char* name, bool bad) : mId(id), mName(name), mBad(bad) {}

 public:
  FlagType();
  FlagType& operator=(const FlagType&) = default;
  FlagType(const FlagType&) = default;
  bool operator==(const FlagType& rhs) const;
  bool operator!=(const FlagType& rhs) const;
  bool operator<(const FlagType& rhs) const;
  bool operator>(const FlagType& rhs) const;

  uint16_t getID() const { return mId; }
  const std::string& getName() const { return mName; }
  bool getBad() const { return mBad; }

  friend std::ostream& operator<<(std::ostream& os, FlagType const& me);
  friend class FlagTypeFactory;
  friend class QualityControlFlagCollection;

  ClassDefNV(FlagType, 1);
};

} // namespace quality_control
} // namespace o2
#endif // O2_QUALITYCONTROL_FLAGTYPE_H
