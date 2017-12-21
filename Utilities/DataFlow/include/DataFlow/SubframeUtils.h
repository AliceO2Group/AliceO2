// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef DATAFLOW_SUBFRAMEUTILS_H
#define DATAFLOW_SUBFRAMEUTILS_H

#include <tuple>
#include <cstddef>
#include "Headers/HeartbeatFrame.h"

namespace o2 { namespace dataflow {

int64_t extractDetectorPayloadStrip(char **payload, char *buffer, size_t bufferSize) {
  *payload = buffer + sizeof(o2::header::HeartbeatHeader);
  return bufferSize - sizeof(o2::header::HeartbeatHeader) - sizeof(o2::header::HeartbeatTrailer);
}


struct SubframeId {
  size_t timeframeId;
  size_t socketId;

  // operator needed for the equal_range algorithm/ multimap method
  bool operator<(const SubframeId& rhs) const {
    return std::tie(timeframeId, socketId) < std::tie(rhs.timeframeId, rhs.socketId);
  }
};

SubframeId makeIdFromHeartbeatHeader(const header::HeartbeatHeader &header, size_t socketId, size_t orbitsPerTimeframe) {
  SubframeId id = {
    .timeframeId = header.orbit / orbitsPerTimeframe,
    .socketId = socketId
  };
  return id;
}

} /* namespace dataflow */ } /* namespace o2 */

#endif // DATAFLOW_SUBFRAMEUTILS_H
