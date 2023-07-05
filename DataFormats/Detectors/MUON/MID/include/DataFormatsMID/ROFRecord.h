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

/// \file   DataFormatsMID/ROFRecord.h
/// \brief  Definition of the MID event record
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   13 October 2019

#ifndef ALICEO2_MID_ROFRECORD_H
#define ALICEO2_MID_ROFRECORD_H

#include <utility>

#include "CommonConstants/LHCConstants.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/TimeStamp.h"
#include "Framework/Logger.h"

namespace o2
{
namespace mid
{

enum class EventType {
  Standard = 0,
  Calib = 1,
  FET = 2
};
constexpr uint32_t NEvTypes = 3;

/// ROFRecord class encodes the trigger interaction record of given ROF and
/// the reference on the 1st object (digit, cluster etc) of this ROF in the data tree
struct ROFRecord {
  using Time = o2::dataformats::TimeStampWithError<float, float>;

  o2::InteractionRecord interactionRecord{}; //< Interaction record
  EventType eventType{EventType::Standard};  //< Event type
  size_t firstEntry{0};                      //< First associated entry
  size_t nEntries{0};                        //< Number of associated entries

  ROFRecord() = default;
  ROFRecord(const o2::InteractionRecord& intRecord, const EventType& evtType, size_t first, size_t nElements) : interactionRecord(intRecord), eventType(evtType), firstEntry(first), nEntries(nElements) {}
  ROFRecord(const ROFRecord& other, size_t first, size_t nElements) : interactionRecord(other.interactionRecord), eventType(other.eventType), firstEntry(first), nEntries(nElements) {}
  size_t getEndIndex() const { return firstEntry + nEntries; }

  //__________________________________________________________________________
  /// return a pair consisting of the ROF time with error (in mus) relative to the reference IR 'startIR'
  /// and a flag telling if it is inside the TF starting at 'startIR' and containing 'nOrbits' orbits.
  /// if printError = true, print an error message in case the ROF is outside the TF
  std::pair<Time, bool> getTimeMUS(const InteractionRecord& startIR, uint32_t nOrbits = 128,
                                   bool printError = false) const
  {
    auto bcDiff = interactionRecord.differenceInBC(startIR);
    float tMean = (bcDiff + 0.5) * o2::constants::lhc::LHCBunchSpacingMUS;
    float tErr = 0.5 * o2::constants::lhc::LHCBunchSpacingMUS;
    bool isInTF = bcDiff >= 0 && bcDiff < nOrbits * o2::constants::lhc::LHCMaxBunches;
    if (printError && !isInTF) {
      LOGP(alarm, "ATTENTION: wrong bunches diff. {} for current IR {} wrt 1st TF orbit {}, source:MID",
           bcDiff, interactionRecord.asString(), startIR.asString());
    }
    return std::make_pair(Time(tMean, tErr), isInTF);
  }

  ClassDefNV(ROFRecord, 1);
};

} // namespace mid
} // namespace o2

#endif
