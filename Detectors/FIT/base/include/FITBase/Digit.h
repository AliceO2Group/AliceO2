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
/// \class Digit
/// \brief FIT digit implementation
using DigitBase = o2::dataformats::TimeStamp<double>;
class Digit : public DigitBase
{
 public:
  Digit() = default;

  Digit(Double_t time, Int_t channel, Double_t cfd, Float_t amp, Int_t bc);
  ~Digit() = default;

  Int_t getChannel() const { return mChannel; }
  void setChannel(Int_t channel) { mChannel = channel; }

  Double_t getTime() const { return mTime; }
  void setTime(Double_t time) { mTime = time; }

  Double_t getCFD() const { return mCFD; }
  void setCFD(Double_t time) { mCFD = time; }

  Float_t getQTC() const { return mQTC; }
  void setQTC(Float_t amp) { mQTC = amp; }

  Int_t getBC() const { return mBC; }
  void setBC(Int_t bc) { mBC = bc; }

  void printStream(std::ostream& stream) const;

 private:
  //  friend class boost::serialization::access;

  Double_t mTime; /// time stamp
  Int_t mChannel; ///< FIT channel index
  Double_t mCFD;  ///< CFD time value
  Float_t mQTC;   ///< QTC time value
  Int_t mBC;      ///< Bunch Crossing

  ClassDefNV(Digit, 1);
};

std::ostream& operator<<(std::ostream& stream, const Digit& dig);
} // namespace fit
} // namespace o2
#endif
