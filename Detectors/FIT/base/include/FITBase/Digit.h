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

#include "CommonDataFormat/TimeStamp.h"
#include <iosfwd>
#include "Rtypes.h"

namespace o2
{
namespace fit
{

struct ChannelData {
  Int_t ChId;       //channel Id
  Double_t CFDTime; //time in ns, 0 at lhc clk center
  Double_t QTCAmpl; // Amplitude in mips
  ClassDefNV(ChannelData, 1);
};

/// \class Digit
/// \brief FIT digit implementation
using DigitBase = o2::dataformats::TimeStamp<double>;
class Digit : public DigitBase
{
 public:
  Digit() = default;

  Digit(std::vector<ChannelData> ChDgDataArr, Double_t time, Int_t bc, Bool_t isA, Bool_t isC, Bool_t isCnt, Bool_t isSCnt, Bool_t isVrtx)
  {
    setChDgData(std::move(ChDgDataArr));
    setTime(time);
    setBC(bc);
    setTriggers(isA, isC, isCnt, isSCnt, isVrtx);
  }

  ~Digit() = default;

  Double_t getTime() const { return mTime; }
  void setTime(Double_t time) { mTime = time; }

  Int_t getBC() const { return mBC; }
  void setBC(Int_t bc) { mBC = bc; }

  Bool_t getisA() const { return mIsA; }
  Bool_t getisC() const { return mIsC; }
  Bool_t getisCnt() const { return mIsCentral; }
  Bool_t getisSCnt() const { return mIsSemiCentral; }
  Bool_t getisVrtx() const { return mIsVertex; }

  void setTriggers(Bool_t isA, Bool_t isC, Bool_t isCnt, Bool_t isSCnt, Bool_t isVrtx)
  {
    mIsA = isA;
    mIsC = isC;
    mIsCentral = isCnt;
    mIsSemiCentral = isSCnt;
    mIsVertex = isVrtx;
  }

  const std::vector<ChannelData>& getChDgData() const { return mChDgDataArr; }
  void setChDgData(const std::vector<ChannelData>& ChDgDataArr) { mChDgDataArr = ChDgDataArr; }
  void setChDgData(std::vector<ChannelData>&& ChDgDataArr) { mChDgDataArr = std::move(ChDgDataArr); }

  void printStream(std::ostream& stream) const;

 private:
  //  friend class boost::serialization::access;

  Double_t mTime; /// time stamp
  Int_t mBC;      ///< Bunch Crossing

  //online triggers processed on TCM
  Bool_t mIsA, mIsC;
  Bool_t mIsCentral;
  Bool_t mIsSemiCentral;
  Bool_t mIsVertex;

  std::vector<ChannelData> mChDgDataArr;

  ClassDefNV(Digit, 1);
};

std::ostream& operator<<(std::ostream& stream, const Digit& digi);
} // namespace fit
} // namespace o2
#endif
