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

#ifndef ALICEO2_TRD_KRCLSTRIGGERRECORD_H
#define ALICEO2_TRD_KRCLSTRIGGERRECORD_H

#include "Rtypes.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"

namespace o2
{
namespace trd
{

/// \class KrClusterTriggerRecord
/// \brief Mapping of found Kr clusters to BC information which is taken from the TRD digits
/// \author Ole Schmidt

class KrClusterTriggerRecord
{
  using BCData = o2::InteractionRecord;

 public:
  KrClusterTriggerRecord() = default;
  KrClusterTriggerRecord(BCData bcData, int nEntries) : mBCData(bcData), mNClusters(nEntries) {}

  // setters (currently class members are set at the time of creation, if there is need to change them afterwards setters can be added below)

  // getters
  int getNumberOfClusters() const { return mNClusters; }
  BCData getBCData() const { return mBCData; }

 private:
  BCData mBCData; ///< bunch crossing data
  int mNClusters; ///< number of Kr clusters

  ClassDefNV(KrClusterTriggerRecord, 1);
};
} // namespace trd
} // namespace o2

#endif // ALICEO2_TRD_KRCLSTRIGGERRECORD_H
