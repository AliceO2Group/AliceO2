// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_DATATYPES_H_
#define O2_FRAMEWORK_DATATYPES_H_

#include <cstdint>

namespace o2::aod::track
{
enum TrackTypeEnum : uint8_t {
  GlobalTrack = 0,
  ITSStandalone,
  MFTStandalone,
  Run2GlobalTrack = 254,
  Run2Tracklet = 255
};
enum TrackFlagsRun2Enum {
  ITSrefit = 0x1,
  TPCrefit = 0x2,
  GoldenChi2 = 0x4
};
} // namespace o2::aod::track

#endif // O2_FRAMEWORK_DATATYPES_H_
