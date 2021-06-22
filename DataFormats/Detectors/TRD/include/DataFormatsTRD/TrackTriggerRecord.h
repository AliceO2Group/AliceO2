// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_TRACKTRIGGERRECORD_H
#define ALICEO2_TRD_TRACKTRIGGERRECORD_H

#include "Rtypes.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"

namespace o2
{
namespace trd
{

/// \class TrackTriggerRecord
/// \brief Mapping of reconstructed TRD tracks to BC information which is taken from the TRD tracklets attached to the track
/// \author Ole Schmidt

class TrackTriggerRecord
{
  using BCData = o2::InteractionRecord;
  using DataRange = o2::dataformats::RangeReference<int>;

 public:
  TrackTriggerRecord() = default;
  TrackTriggerRecord(BCData bcData, int firstEntry, int nEntries) : mBCData(bcData), mTrackDataRange(firstEntry, nEntries) {}

  // setters (currently class members are set at the time of creation, if there is need to change them afterwards setters can be added below)

  // getters
  const auto& getTrackRefs() const { return mTrackDataRange; }
  int getFirstTrack() const { return mTrackDataRange.getFirstEntry(); }
  int getNumberOfTracks() const { return mTrackDataRange.getEntries(); }
  BCData getBCData() const { return mBCData; }

 private:
  BCData mBCData;            ///< bunch crossing data
  DataRange mTrackDataRange; ///< range of tracklets for each BC data

  ClassDefNV(TrackTriggerRecord, 1);
};
} // namespace trd
} // namespace o2

#endif // ALICEO2_TRD_TRACKTRIGGERRECORD_H
