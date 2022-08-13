// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CalibInfoTOF.h
/// \brief Class to store the output of the matching to TOF for calibration

#ifndef ALICEO2_CALIBINFOTOF_H
#define ALICEO2_CALIBINFOTOF_H

#include "Rtypes.h"

namespace o2
{
namespace dataformats
{
class CalibInfoTOF
{
 public:
  CalibInfoTOF(int indexTOFCh, int timestamp, float DeltaTimePi, float tot, int mask, int flags = 0) : mTOFChIndex(indexTOFCh), mTimestamp(timestamp), mDeltaTimePi(DeltaTimePi), mTot(tot), mMask(mask), mFlags(flags){};
  CalibInfoTOF() = default;
  ~CalibInfoTOF() = default;

  void setTOFChIndex(int index) { mTOFChIndex = index; }
  int getTOFChIndex() const { return mTOFChIndex; }

  void setTimestamp(int ts) { mTimestamp = ts; }
  int getTimestamp() const { return mTimestamp; }

  void setDeltaTimePi(float time) { mDeltaTimePi = time; }
  float getDeltaTimePi() const { return mDeltaTimePi; }

  void setTot(float tot) { mTot = tot; }
  float getTot() const { return mTot; }

  void setFlags(int flags) { mFlags = flags; }
  float getFlags() const { return mFlags; }

  int getMask() const { return mMask; }

  // for event time maker
  float tofSignal() const { return mDeltaTimePi; }
  float tofExpSignalPi() const { return 0.0; }
  float tofExpSignalKa() const { return 0.0; }
  float tofExpSignalPr() const { return 0.0; }
  float tofExpSigmaPi() const { return 500.0; }
  float tofExpSigmaKa() const { return 500.0; }
  float tofExpSigmaPr() const { return 500.0; }

 private:
  int mTOFChIndex;      // index of the TOF channel
  int mTimestamp;       // timestamp in seconds
  float mDeltaTimePi;   // raw tof time - expected time for pi hypotesis
  float mTot;           // time-over-threshold
  unsigned char mFlags; // bit mask with quality flags (to be defined)
  int mMask;            // mask for int BC used
  ClassDefNV(CalibInfoTOF, 2);
};
} // namespace dataformats
} // namespace o2
#endif
