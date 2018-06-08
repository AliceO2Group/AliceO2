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
#include <iostream>

#include "Rtypes.h"

namespace o2
{
namespace fit
{

struct ChannelDigitData {
//    ChannelDigitData(Int_t chid, Float_t cfdtime, Float_t qtcampl) :
//        ChId(chid), CFDTime(cfdtime), QTCAmpl(qtcampl) {}
//~ChannelDigitData() {}

  Int_t ChId; //channel Id
  Float_t CFDTime; //time in ns, 0 at lhc clk center
  Float_t QTCAmpl; // Amplitude in mips
  ClassDefNV(ChannelDigitData, 1);
};

/// \class Digit
/// \brief FIT digit implementation
using DigitBase = o2::dataformats::TimeStamp<double>;
class Digit : public DigitBase
{
 public:
  Digit() = default;

  Digit(std::vector<ChannelDigitData> ChDgDataArr, Double_t time, Int_t bc, Bool_t IsA, Bool_t IsC, Bool_t IsCnt, Bool_t IsSCnt, Bool_t IsVrtx)
  {setChDgData(ChDgDataArr); setTime(time); setBC(bc); setTriggers(IsA, IsC, IsCnt, IsSCnt, IsVrtx); }

  ~Digit() = default;

  Double_t getTime() const { return mTime; }
  void setTime(Double_t time) { mTime = time; }

  Int_t getBC() const { return mBC; }
  void setBC(Int_t bc) { mBC = bc; }

  Bool_t getIsA() const {return mIsA;}
  Bool_t getIsC() const {return mIsC;}
  Bool_t getIsCnt() const {return mIsCentral;}
  Bool_t getIsSCnt() const {return mIsSemiCentral;}
  Bool_t getIsVrtx() const {return mIsVertex;}

  void setTriggers(Bool_t IsA, Bool_t IsC, Bool_t IsCnt, Bool_t IsSCnt, Bool_t IsVrtx)
  {mIsA = IsA; mIsC = IsC; mIsCentral = IsCnt; mIsSemiCentral = IsSCnt; mIsVertex = IsVrtx;}

  std::vector<ChannelDigitData> getChDgData() const {return mChDgDataArr; }
  void setChDgData(std::vector<ChannelDigitData> ChDgDataArr) {mChDgDataArr = ChDgDataArr;}


  void printStream(std::ostream& stream) const
  {
      stream << "FIT Digit: event time " << mTime << " BC " << mBC << std::endl;
      stream << "IS A " << mIsA << " IS C " << mIsC << " Is Central " << mIsCentral
                 << " Is SemiCentral " << mIsSemiCentral << " Is Vertex " << mIsVertex << std::endl;

      for (auto& chdata : mChDgDataArr)
          stream << "CH " << chdata.ChId << " TIME " << chdata.CFDTime << " MIP " << chdata.QTCAmpl << std::endl;
  }

 private:
  //  friend class boost::serialization::access;

  Double_t mTime; /// time stamp
  Int_t mBC;      ///< Bunch Crossing

  //online triggers processed on TCM
  Bool_t mIsA, mIsC;
  Bool_t mIsCentral;
  Bool_t mIsSemiCentral;
  Bool_t mIsVertex;

  std::vector<ChannelDigitData> mChDgDataArr;


  ClassDefNV(Digit, 1);
};

std::ostream& operator<<(std::ostream& stream, const Digit& digi);
} // namespace fit
} // namespace o2
#endif
