// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FDD_DIGIT_H
#define ALICEO2_FDD_DIGIT_H

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/TimeStamp.h"
#include <iosfwd>
#include "Rtypes.h"

namespace o2
{
namespace fdd
{

struct ChannelData {
  Int_t mPMNumber;    // PhotoMultiplier number (0 to 16)
  Float_t mTime;      // Time of Flight
  Float_t mWidth;     // Width of the time distribution
  Short_t mChargeADC; // ADC sample as present in raw data
  ClassDefNV(ChannelData, 1);
};

using DigitBase = o2::dataformats::TimeStamp<double>;
class Digit : public DigitBase
{
 public:
  Digit() = default;

  Digit(std::vector<ChannelData> const& channelData, Double_t time, uint16_t bc, uint32_t orbit, std::vector<Bool_t> const& triggers)
  {
    SetChannelData(std::move(channelData));
    SetTime(time);
    SetInteractionRecord(bc, orbit);
    SetTriggers(std::move(triggers));
  }

  ~Digit() = default;

  Double_t GetTime() const { return mTime; }
  void SetTime(Double_t time) { mTime = time; }

  void SetInteractionRecord(uint16_t bc, uint32_t orbit)
  {
    mIntRecord.bc = bc;
    mIntRecord.orbit = orbit;
  }
  const o2::InteractionRecord& GetInteractionRecord() const { return mIntRecord; }
  o2::InteractionRecord& GetInteractionRecord(o2::InteractionRecord& src) { return mIntRecord; }
  void SetInteractionRecord(const o2::InteractionRecord& src) { mIntRecord = src; }
  uint32_t GetOrbit() const { return mIntRecord.orbit; }
  uint16_t GetBC() const { return mIntRecord.bc; }

  const std::vector<Bool_t>& GetTriggers() const { return mTriggers; }
  std::vector<Bool_t>& GetTriggers() { return mTriggers; }
  void SetTriggers(const std::vector<Bool_t>& triggers) { mTriggers = triggers; }
  void SetTriggers(std::vector<Bool_t>&& triggers) { mTriggers = std::move(triggers); }

  const std::vector<ChannelData>& GetChannelData() const { return mChannelData; }
  std::vector<ChannelData>& GetChannelData() { return mChannelData; }
  void SetChannelData(const std::vector<ChannelData>& channelData) { mChannelData = channelData; }
  void SetChannelData(std::vector<ChannelData>&& channelData) { mChannelData = std::move(channelData); }

  void ClearDigits()
  {
    mTriggers.clear();
    mChannelData.clear();
  }

 private:
  Double_t mTime;                   // time stamp
  o2::InteractionRecord mIntRecord; // Interaction record (orbit, bc)

  std::vector<Bool_t> mTriggers;
  std::vector<ChannelData> mChannelData;

  ClassDefNV(Digit, 1);
};

std::ostream& operator<<(std::ostream& stream, const Digit& digi);
} // namespace fdd
} // namespace o2
#endif
