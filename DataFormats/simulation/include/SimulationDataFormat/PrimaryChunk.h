// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_DATA_PRIMARYCHUNK_H_
#define ALICEO2_DATA_PRIMARYCHUNK_H_

#include <cstring>
#include <SimulationDataFormat/MCEventHeader.h>

namespace o2
{
namespace data
{

// structure describing an entity of work
// processed by a simulation worker
struct SubEventInfo {
  float eventtime = 0.;
  uint32_t eventID = 0;   // which event ID
  int32_t maxEvents = -1; // the number of events in this run (if known otherwise set to -1)
  int32_t runID = 0;      // the runID of this run
  uint16_t part = 0;      // which part of the eventID
  uint16_t nparts = 0;    // out of how many parts
  uint32_t seed = 0;      // seed for RNG
  uint32_t index = 0;
  int32_t npersistenttracks = -1; // the number of persistent tracks for this SubEvent (might be set to cache it)
  // might add more fields (such as which process treated this chunk etc)

  o2::dataformats::MCEventHeader mMCEventHeader; // associated FairMC header for vertex information

  ClassDefNV(SubEventInfo, 1);
};

inline bool operator<(SubEventInfo const& a, SubEventInfo const& b)
{
  return a.eventID <= b.eventID && (a.part < b.part);
}

// Encapsulating primaries/tracks as well as the event info
// to be processed by the simulation processors.
struct PrimaryChunk {
  SubEventInfo mSubEventInfo;
  std::vector<TParticle> mParticles; // the particles for this chunk
  ClassDefNV(PrimaryChunk, 1);
};
} // namespace data
} // namespace o2

#endif
