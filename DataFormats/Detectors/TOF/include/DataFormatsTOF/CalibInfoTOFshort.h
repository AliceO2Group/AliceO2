// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CalibInfoTOFshort.h
/// \brief Class to store the output of the matching to TOF for calibration (no channel info, available in CalibInfoTOF.h)

#ifndef ALICEO2_CALIBINFOTOFSHORT_H
#define ALICEO2_CALIBINFOTOFSHORT_H

namespace o2
{
namespace dataformats
{
class CalibInfoTOFshort
{
 public:
  CalibInfoTOFshort(int timestamp, float DeltaTimePi, float tot, int flags = 0) : mTimestamp(timestamp), mDeltaTimePi(DeltaTimePi), mTot(tot), mFlags(flags){};
  CalibInfoTOFshort() = default;
  void setTOFChIndex(int index) {}
  int getTOFChIndex() const { return 0; }

  void setTimestamp(int ts) { mTimestamp = ts; }
  int getTimestamp() const { return mTimestamp; }

  void setDeltaTimePi(float time) { mDeltaTimePi = time; }
  float getDeltaTimePi() const { return mDeltaTimePi; }

  void setTot(int tot) { mTot = tot; }
  float getTot() const { return mTot; }

  void setFlags(int flags) { mFlags = flags; }
  float getFlags() const { return mFlags; }

 private:
  int mTimestamp;       // timestamp in seconds
  float mDeltaTimePi;   // raw tof time - expected time for pi hypotesis
  float mTot;           // time-over-threshold
  unsigned char mFlags; // bit mask with quality flags (to be defined)
  //  ClassDefNV(CalibInfoTOFshort, 1);
};
} // namespace dataformats
} // namespace o2
#endif
