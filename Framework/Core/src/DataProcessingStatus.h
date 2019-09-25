// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef o2_framework_DataProcessingStatus_H_INCLUDED
#define o2_framework_DataProcessingStatus_H_INCLUDED

#include "Framework/Signpost.h"
#include <cstdint>

/// probes to be used by the DPL
#define O2_PROBE_DATARELAYER 3

namespace o2
{
namespace framework
{

/// Describe the possible states for DataProcessing
enum struct DataProcessingStatus : uint32_t {
  ID = 0,
  IN_DPL_OVERHEAD = O2_SIGNPOST_RED,
  IN_DPL_USER_CALLBACK = O2_SIGNPOST_GREEN,
  IN_DPL_ERROR_CALLBACK = O2_SIGNPOST_PURPLE
};

/// Describe the possible states for Monitoring
enum struct MonitoringStatus : uint32_t {
  ID = 1,
  SEND = 0,
  FLUSH = 1,
};

enum struct DriverStatus : uint32_t {
  ID = 2,
  BYTES_READ = 0,
  BYTES_PROCESSED = 1,
  BUFFER_OVERFLOWS = 2
};

} // namespace framework
} // namespace o2

#endif // o2_framework_DataProcessingStatus_H_INCLUDED
