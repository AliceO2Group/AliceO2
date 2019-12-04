// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/** @file Digit.h
 * C++ simple Muon MCH digit.
 * @author  Michael Winn
 */

#ifndef ALICEO2_MCH_BASE_DIGIT_H_
#define ALICEO2_MCH_BASE_DIGIT_H_

namespace o2
{
namespace mch
{

// \class Digit
/// \brief MCH digit implementation
class Digit //: public DigitBase
{
 public:
  Digit() = default;

  Digit(double time, int detid, int pad, unsigned long adc);
  ~Digit() = default;

  double getTimeStamp() const { return mTime; }
  void setTimeStamp(double time) { mTime = time; }
  
  int getDetID() const { return mDetID; }
  void setDetID(int detid) { mDetID = detid; }

  int getPadID() const { return mPadID; }
  void setPadID(int pad) { mPadID = pad; }

  unsigned long getADC() const { return mADC; }
  void setADC(unsigned long adc) { mADC = adc; }
  
  bool operator==(const Digit & d) const { return mTime == d.getTimeStamp(); }
  
 private:
  double mTime;
  int mDetID;
  int mPadID;  /// PadIndex to which the digit corresponds to
  unsigned long mADC; /// Amplitude of signal
  //ClassDefNV(Digit, 1);
}; //class Digit

} //namespace mch
} //namespace o2
#endif // ALICEO2_MCH_DIGIT_H_
