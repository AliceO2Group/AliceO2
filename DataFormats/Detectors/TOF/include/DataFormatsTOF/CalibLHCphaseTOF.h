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

/// \file CalibLHCphaseTOF.h
/// \brief Class to store the output of the matching to TOF for calibration

#ifndef ALICEO2_CALIBLHCPHASETOF_H
#define ALICEO2_CALIBLHCPHASETOF_H

#include <vector>
#include "Rtypes.h"

namespace o2
{
namespace dataformats
{
class CalibLHCphaseTOF
{
 public:
  CalibLHCphaseTOF() = default;

  float getLHCphase(int timestamp) const;

  void addLHCphase(int timestamp, float phaseLHC);

  int size() const { return mLHCphase.size(); }
  int timestamp(int i) const { return mLHCphase[i].first; }
  float LHCphase(int i) const { return mLHCphase[i].second; }

  CalibLHCphaseTOF& operator+=(const CalibLHCphaseTOF& other);

  long getStartValidity() const { return mStartValidity; }
  long getEndValidity() const { return mEndValidity; }

  void setStartValidity(long validity) { mStartValidity = validity; }
  void setEndValidity(long validity) { mEndValidity = validity; }

 private:
  // LHCphase calibration
  std::vector<std::pair<int, float>> mLHCphase; ///< <timestamp,LHCphase> from which the LHCphase measurement is valid; timestamp in seconds

  long mStartValidity = 0; ///< start validity of the object when put in CCDB
  long mEndValidity = 0;   ///< end validity of the object when put in CCDB

  ClassDefNV(CalibLHCphaseTOF, 2);
};
} // namespace dataformats
} // namespace o2
#endif
