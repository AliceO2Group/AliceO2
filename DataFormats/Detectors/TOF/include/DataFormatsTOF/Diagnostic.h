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

/// \file Diagnostic.h
/// \brief Definition of the TOF cluster

#ifndef ALICEO2_TOF_DIAGNOSTIC_H
#define ALICEO2_TOF_DIAGNOSTIC_H

#include <map>
#include <TObject.h>
#include <gsl/gsl>
#include "CommonDataFormat/TFIDInfo.h"

namespace o2
{
namespace tof
{
/// \class Diagnostic
/// \brief Diagnostic class for TOF
///

class Diagnostic
{
 public:
  Diagnostic() = default;
  int fill(ULong64_t pattern);
  int fill(ULong64_t pattern, int frequency);
  int getFrequency(ULong64_t pattern) const;                                                    // Get frequency
  int getFrequencyROW() const { return getFrequency(0); }                                       // Readout window frequency
  int getFrequencyEmptyCrate(int crate) const { return getFrequency(getEmptyCrateKey(crate)); } // empty crate frequency
  int getFrequencyEmptyTOF() const { return getFrequency(1); }                                  // empty crate frequency
  int fillNoisy(int channel, int frequency = 1) { return fill(getNoisyChannelKey(channel), frequency); }
  int fillROW() { return fill(0); }
  int fillEmptyCrate(int crate, int frequency = 1) { return fill(getEmptyCrateKey(crate), frequency); }
  int fillEmptyTOF(int frequency = 1) { return fill(1, frequency); }
  static ULong64_t getEmptyCrateKey(int crate);
  static ULong64_t getNoisyChannelKey(int channel);
  static ULong64_t getTRMKey(int crate, int trm);
  void print() const;
  void clear() { mVector.clear(); }
  void fill(const Diagnostic& diag);                       // for calibration
  void fill(const gsl::span<const o2::tof::Diagnostic>){}; // for calibration
  void merge(const Diagnostic* prev);
  void getNoisyMap(Bool_t* output, int noisyThr = 1) const; // set true in output channel array
  void getNoisyLevelMap(Char_t* output) const;              // set true in output channel array
  bool isNoisyChannel(int channel, int thr = 0) const;
  unsigned long size() const { return mVector.size(); }
  ULong64_t getPattern(int i) const
  {
    auto iter = mVector.begin();
    for (; i-- > 0;) {
      iter++;
    }
    return iter->first;
  }
  static int getSlot(ULong64_t pattern) { return (pattern & 68719476735) / 4294967296; }
  static int getCrate(ULong64_t pattern) { return (pattern & 8796093022207) / 68719476736; }
  static int getChannel(ULong64_t pattern)
  {
    if (getSlot(pattern) == 14) {
      return (pattern & 262143);
    }
    return -1;
  }
  static int getNoisyLevel(ULong64_t pattern)
  {
    if (getChannel(pattern)) {
      if (pattern & (1 << 20)) {
        return 3;
      } else if (pattern & (1 << 19)) {
        return 2;
      } else {
        return 1;
      }
    }
    return 0;
  }

  const std::map<ULong64_t, uint32_t>& getVector() const { return mVector; }

  int getTimeStamp() const { return mTimestamp; }
  void setTimeStamp(int val) { mTimestamp = val; }

  void setTFIDInfo(const o2::dataformats::TFIDInfo& val) { mTFinfo = val; }
  const o2::dataformats::TFIDInfo& getTFIDInfo() const { return mTFinfo; }

 private:
  std::map<ULong64_t, uint32_t> mVector; // diagnostic frequency vector (key/pattern , frequency)
  int mTimestamp = 0;                    //! timestamp in seconds
  o2::dataformats::TFIDInfo mTFinfo;     // TF id info

  ClassDefNV(Diagnostic, 3);
};

} // namespace tof
} // namespace o2
#endif
