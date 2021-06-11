// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file EMCALChannelData.h
/// \brief

/// \class EMCALChannelCalibrator
/// \brief Class to store the data format for calibraton of the EMCal
/// \author Hannah Bossi, Yale University
/// \ingroup DetectorEMCAL
/// \since Feb 11, 2021

#ifndef ALICEO2_EMCALCHANNELDATA_H
#define ALICEO2_EMCALCHANNELDATA_H

#include "Rtypes.h"

namespace o2
{
namespace dataformats
{
class EMCALChannelData
{
 public:
  EMCALChannelData(int cellID, int timestamp, int flags = 0, int events) : mEMCALCellID(cellID), mTimestamp(timestamp), mFlags(flags){};
  EMCALChannelData() = default;
  ~EMCALChannelData() = default;

  void setEMCALCellID(int index) { mEMCALCellID = index; }
  int getEMCALCellID() const { return mEMCALCellID; }

  void setTimestamp(int ts) { mTimestamp = ts; }
  int getTimestamp() const { return mTimestamp; }

  void setFlags(int flags) { mFlags = flags; }
  float getFlags() const { return mFlags; }

 private:
  int mEMCALCellID;     ///< EMCal Cell ID
  int mTimestamp;       ///< timestamp in seconds
  unsigned char mFlags; ///< bit mask with quality flags (to be defined)

  ClassDefNV(EMCALChannelData, 1);
};
} // namespace dataformats
} // namespace o2
#endif
