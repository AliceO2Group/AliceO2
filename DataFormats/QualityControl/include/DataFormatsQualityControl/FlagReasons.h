// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

struct FlagReasonFactory;

class FlagReason
{
 private:
  uint16_t mId;
  std::string mName;
  bool mBad; // if true, data should become bad by default

  // By making the constructor private and FlagReasons available only in the FlagReasonFactory
  // we forbid to declare any flags in the user code. If you need a new FlagReason, please add it FlagReasonFactory.
 private:
  FlagReason(uint16_t id, const char* name, bool bad) : mId(id), mName(name), mBad(bad) {}

 public:
  FlagReason& operator=(const FlagReason&) = default;
  FlagReason(const FlagReason&) = default;
  bool operator==(const FlagReason& rhs) const;
  bool operator!=(const FlagReason& rhs) const;
  bool operator<(const FlagReason& rhs) const;
  bool operator>(const FlagReason& rhs) const;

  uint16_t getID() { return mId; }
  const std::string& getName() { return mName; }
  bool getBad() { return mBad; }

  friend std::ostream& operator<<(std::ostream& os, FlagReason const& me);
  friend FlagReasonFactory;

  ClassDefNV(FlagReason, 1);
};

struct FlagReasonFactory {
  FlagReasonFactory() = delete;

  // TODO: migrate the flag list from RCT
  // TODO: find a way to have a nicely formatted list of reasons.

  // !!! NEVER MODIFY OR DELETE EXISTING FLAGS AFTER RUN 3 STARTS !!!
  static FlagReason Invalid() { return {static_cast<uint16_t>(-1), "Invalid", true}; }

  static FlagReason Unknown() { return {1, "Unknown", true}; }
  static FlagReason ProcessingError() { return {2, "Processing error", true}; }
  // it can be used when there are no required Quality Objects in QCDB in certain time range.
  static FlagReason MissingQualityObject() { return {3, "Missing Quality Object", true}; }
  // Quality Object is there, but it has Quality::Null
  static FlagReason MissingQuality() { return {4, "Missing Quality", true}; }

  // TODO: to be seen if we should actively do anything when a detector was off.
  static FlagReason DetectorOff() { return {10, "Detector off", true}; }
  static FlagReason LimitedAcceptance() { return {11, "Limited acceptance", true}; }
};

} // namespace quality_control
} // namespace o2
#endif
