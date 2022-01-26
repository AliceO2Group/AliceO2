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

#ifndef ALICEO2_ANALYSIS_FLAGREASONS
#define ALICEO2_ANALYSIS_FLAGREASONS

/// \file FlagReasons.h
/// \brief classes keeping reasons for flagging time ranges
/// \author Jens Wiechula, jens.wiechula@ikf.uni-frankfurt.de
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

// STL
#include <iosfwd>
#include <cstdint>

// ROOT includes
#include <Rtypes.h>

namespace o2
{
namespace quality_control
{

class FlagReasonFactory;

class FlagReason
{
 private:
  uint16_t mId;
  const char* mName;
  bool mBad; // if true, data should become bad by default

  // By making the constructor private and FlagReasons available only in the FlagReasonFactory
  // we forbid to declare any flags in the user code. If you need a new FlagReason, please add it FlagReasonFactory.
 private:
  constexpr FlagReason(uint16_t id, const char* name, bool bad) : mId(id), mName(name), mBad(bad) {}

 public:
  FlagReason();
  FlagReason& operator=(const FlagReason&) = default;
  FlagReason(const FlagReason&) = default;
  bool operator==(const FlagReason& rhs) const;
  bool operator!=(const FlagReason& rhs) const;
  bool operator<(const FlagReason& rhs) const;
  bool operator>(const FlagReason& rhs) const;

  uint16_t getID() const { return mId; }
  const char* getName() const { return mName; }
  bool getBad() const { return mBad; }

  friend std::ostream& operator<<(std::ostream& os, FlagReason const& me);
  friend class FlagReasonFactory;

  ClassDefNV(FlagReason, 1);
};

} // namespace quality_control
} // namespace o2

#endif