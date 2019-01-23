// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CalibInfoTOF.h
/// \brief Class to store the output of the matching to TOF for calibration

#ifndef ALICEO2_CALIBINFOTOF_H
#define ALICEO2_CALIBINFOTOF_H

namespace o2
{
namespace dataformats
{
class CalibInfoTOF
{
 public:
  CalibInfoTOF(int indexTOFCh, float DeltaTimePi, float tot, int flags=0) : mTOFChIndex(indexTOFCh), mDeltaTimePi(DeltaTimePi), mTot(tot), mFlags(flags){};
  CalibInfoTOF() = default;
  void setTOFChIndex(int index) { mTOFChIndex = index; }
  int getTOFChIndex() const { return mTOFChIndex; }

  void setDeltaTimePi(int time) { mDeltaTimePi = time; }
  float getDeltaTimePi() const { return mDeltaTimePi; }

  void setTot(int tot) { mTot = tot; }
  float getTot() const { return mTot; }

 private:
  int mTOFChIndex; // index of the TOF channel
  float mDeltaTimePi;   // raw tof time - expected time for pi hypotesis
  float mTot;      // time-over-threshold
  unsigned char mFlags;   // bit mask with quality flags (to be defined)
  //  ClassDefNV(CalibInfoTOF, 1);
};
}
}
#endif
