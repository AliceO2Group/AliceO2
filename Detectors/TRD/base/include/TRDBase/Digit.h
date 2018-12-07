// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_DIGIT_H_
#define ALICEO2_TRD_DIGIT_H_

namespace o2 {
namespace trd {
class Digit {
 public:
  Digit();
  ~Digit();
  int GetRow() const { return mRow; }
  int GetCol() const { return mCol; }
  int GetTime() const { return mTime; }
  int GetAmp() const { return mAmp; }
  int GetEventID() const { return mEventID; }
  int GetSourceID() const { return mSourceID; }

 private:
  int mRow; // pad row number
  int mCol; // pad col number
  int mTime; // time stamp
  int mAmp; // digitalized energy
  int mEventID;
  int mSourceID;
};
}  // namespace trd
}  // namespace o2

#endif
