// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TOF_DIGIT_H_
#define ALICEO2_TOF_DIGIT_H_

#include "FairTimeStamp.h"
#include "Rtypes.h"
#include <iosfwd>

#ifndef __CINT__
#include <boost/serialization/base_object.hpp> // for base_object
#endif

namespace o2 {
namespace tof {
/// \class Digit
/// \brief TOF digit implementation
class Digit : public FairTimeStamp {
public:
  Digit() = default;

  Digit(Double_t time, Int_t channel, Int_t tdc, Int_t tot, Int_t bc);
  ~Digit() override = default;

  Int_t getChannel() const { return mChannel; }
  void setChannel(Int_t channel) { mChannel = channel; }

  Int_t getTDC() const { return mTDC; }
  void setTDC(Int_t tdc) { mTDC = tdc; }

  Int_t getTOT() const { return mTOT; }
  void setTOT(Int_t tot) { mTOT = tot; }

  Int_t getBC() const { return mBC; }
  void setBC(Int_t bc) { mBC = bc; }

  void printStream(std::ostream &stream) const;

private:
#ifndef __CINT__
  friend class boost::serialization::access;
#endif

  Int_t mChannel;       ///< TOF channel index
  Int_t mTDC;           ///< TDC bin number
  Int_t mTOT;           ///< TOT bin number
  Int_t mBC;            ///< Bunch Crossing

  ClassDefOverride(Digit, 1);
};

std::ostream &operator<<(std::ostream &stream, const Digit &dig);
} // namespace TOF
} // namespace o2
#endif
