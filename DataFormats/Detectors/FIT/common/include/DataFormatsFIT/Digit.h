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

// \file Digit.h
/// \brief FIT event entity
/// \author Artur Furs afurs@cern.ch

#ifndef O2_FIT_DIGIT_H_
#define O2_FIT_DIGIT_H_

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsFIT/ChannelData.h"
#include "DataFormatsFIT/Triggers.h"

#include <gsl/span>
#include <Rtypes.h>

namespace o2
{
namespace fit
{
using Triggers = o2::fit::Triggers;

template <o2::detectors::DetID::ID DetID>
struct DigitBase {
  static constexpr o2::detectors::DetID sDetID = o2::detectors::DetID(DetID);
  typedef ChannelData<DetID> ChannelData_t; // related ChannelData entity
  o2::InteractionRecord mIR{};              // Interaction record (orbit, bc)
  o2::dataformats::RangeReference<int, int> mReference{};
  DigitBase() = default;
  DigitBase(int first, int nEntries, const o2::InteractionRecord& ir) : mIR(ir)
  {
    mReference.setFirstEntry(first);
    mReference.setEntries(nEntries);
  }
  uint32_t getOrbit() const { return mIR.orbit; }
  uint16_t getBC() const { return mIR.bc; }
  const o2::InteractionRecord& getIntRecord() const { return mIR; };
  gsl::span<const ChannelData_t> getBunchChannelData(const gsl::span<const ChannelData_t> channelData) const
  {
    return mReference.getEntries() ? gsl::span<const ChannelData_t>(&channelData[mReference.getFirstEntry()], mReference.getEntries()) : gsl::span<const ChannelData_t>();
  }
  ClassDefNV(DigitBase, 1);
};

template <o2::detectors::DetID::ID DetID>
struct Digit : public DigitBase<DetID> {
  uint8_t mTriggerWord{};
  Digit() = default;
  Digit(int first, int nEntries, const o2::InteractionRecord& ir, uint8_t trgWord) : DigitBase<DetID>(first, nEntries, ir), mTriggerWord(trgWord)
  {
  }
  ClassDefNV(Digit, 1);
};

template <o2::detectors::DetID::ID DetID>
struct DigitExt : public DigitBase<DetID> {
  Triggers mTriggers{};
  DigitExt() = default;
  DigitExt(int first, int nEntries, const o2::InteractionRecord& ir, const Triggers& triggers) : DigitBase<DetID>(first, nEntries, ir), mTriggers(triggers)
  {
  }
  ClassDefNV(DigitExt, 1);
};

} // namespace fit
} // namespace o2

#endif
