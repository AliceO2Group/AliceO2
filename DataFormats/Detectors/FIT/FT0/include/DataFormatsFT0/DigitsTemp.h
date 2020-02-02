// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FIT_DIGIT_H
#define ALICEO2_FIT_DIGIT_H

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/TimeStamp.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"
#include <iosfwd>
#include "Rtypes.h"

namespace o2
{
namespace ft0
{

/// \class Digit
/// \brief FIT digit implementation
using DigitBase = o2::dataformats::TimeStamp<double>;
class DigitsTemp : public DigitBase
{
 public:
  DigitsTemp() = default;

  DigitsTemp(std::vector<ChannelData> ChDgDataArr, Double_t time, uint16_t bc, uint32_t orbit, uint64_t triggerWord)
    : mChDgDataArr(std::move(ChDgDataArr))
  {
    setTime(time);
    setInteractionRecord(bc, orbit);
    setTriggers(triggerWord);
  }
  DigitsTemp(std::vector<ChannelData> ChDgDataArr, Double_t time, uint16_t bc, uint32_t orbit, o2::ft0::Triggers trigger)
    : mChDgDataArr(std::move(ChDgDataArr)),
      mTrigger(trigger)
  {
    setTime(time);
    setInteractionRecord(bc, orbit);
  }
  ~DigitsTemp() = default;

  Double_t getTime() const { return mTime; }
  void setTime(Double_t time) { mTime = time; }

  void setInteractionRecord(uint16_t bc, uint32_t orbit)
  {
    mIntRecord.bc = bc;
    mIntRecord.orbit = orbit;
  }
  const o2::InteractionRecord& getInteractionRecord() const { return mIntRecord; }
  o2::InteractionRecord& getInteractionRecord(o2::InteractionRecord& src) { return mIntRecord; }
  void setInteractionRecord(const o2::InteractionRecord& src) { mIntRecord = src; }
  uint32_t getOrbit() const { return mIntRecord.orbit; }
  uint16_t getBC() const { return mIntRecord.bc; }

  o2::ft0::Triggers mTrigger; //online triggers processed on TCM
  void setTriggers(uint64_t trig) { mTrigger.word = trig; }

  const std::vector<ChannelData>& getChDgData() const { return mChDgDataArr; }
  std::vector<ChannelData>& getChDgData() { return mChDgDataArr; }
  void setChDgData(const std::vector<ChannelData>& ChDgDataArr) { mChDgDataArr = ChDgDataArr; }
  void setChDgData(std::vector<ChannelData>&& ChDgDataArr) { mChDgDataArr = std::move(ChDgDataArr); }

  void printStream(std::ostream& stream) const;
  void cleardigits()
  {
    mChDgDataArr.clear();
  }

 private:
  Double_t mTime;                   // time stamp
  o2::InteractionRecord mIntRecord; // Interaction record (orbit, bc)

  std::vector<ChannelData> mChDgDataArr;

  ClassDefNV(DigitsTemp, 1);
};

std::ostream& operator<<(std::ostream& stream, const DigitsTemp& digi);
} // namespace ft0
} // namespace o2
#endif
