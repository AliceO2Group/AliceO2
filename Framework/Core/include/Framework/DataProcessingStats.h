// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef o2_framework_DataProcessingStats_H_INCLUDED
#define o2_framework_DataProcessingStats_H_INCLUDED

#include <cstdint>

namespace o2
{
namespace framework
{

/// Helper struct to hold statistics about the data processing happening.
struct DataProcessingStats {
  // We use this to keep track of the latency of the first message we get for a given input record
  // and of the last one.
  struct InputLatency {
    int minLatency = 0;
    int maxLatency = 0;
  };
  int pendingInputs = 0;
  int incomplete = 0;
  int inputParts = 0;
  int lastElapsedTimeMs = 0;
  int lastTotalProcessedSize = 0;
  InputLatency lastLatency = {};
  std::vector<int> relayerState;
};

} // namespace framework
} // namespace o2

#endif // o2_framework_DataProcessingStats_H_INCLUDED
