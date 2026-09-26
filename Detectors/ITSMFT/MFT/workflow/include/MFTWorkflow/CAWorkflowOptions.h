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

#ifndef O2_MFT_CAWORKFLOWOPTIONS_H_
#define O2_MFT_CAWORKFLOWOPTIONS_H_

#include <string>
#include <vector>
#include "ITSMFTTracking/Configuration.h"

namespace o2::framework
{
class ConfigContext;
}
namespace o2::mft::ca
{
enum class WorkflowKind { Reconstruction,
                          TrackerOnly };
enum class InputStage { DigitsFile,
                        UpstreamDigits,
                        UpstreamClusters };
enum class GeometrySource { MFT,
                            Full };
enum class IRFrameSource { None,
                           File,
                           Upstream };
enum class OutputPolicy { All,
                          TracksAndClusterROFs,
                          ClusterROFs,
                          None };

struct TrackerOptions {
  bool useMC = true;
  GeometrySource geometry = GeometrySource::MFT;
  o2::itsmft::TrackingMode::Type mode = o2::itsmft::TrackingMode::Sync;
  int nThreads = 1;
  IRFrameSource irFrames = IRFrameSource::None;
  bool filterIRFrames = false;
};

struct WorkflowOptions {
  WorkflowKind kind = WorkflowKind::Reconstruction;
  InputStage input = InputStage::DigitsFile;
  OutputPolicy output = OutputPolicy::All;
  TrackerOptions tracker;
  bool staggering = false;
  bool runTracking = true; // false removes devices; mode=Off retains an inactive tracker and its consumers.
  bool assessment = false;
  bool processGenerated = true;
  bool tracksToRecords = false;
  std::vector<std::string> diagnostics;
};

// External flags live only at this compatibility boundary. The resolver has
// no singleton, field/geometry, or DPL device dependencies.
struct WorkflowOptionInput {
  WorkflowKind kind = WorkflowKind::Reconstruction;
  bool useMC = true;
  bool staggering = false;
  bool fullGeometry = false;
  bool useIRFrames = false;
  bool upstreamDigits = false;
  bool upstreamClusters = false;
  bool clusterROFsOnly = false;
  bool disableRootOutput = false;
  bool runTracking = true;
  bool assessment = false;
  bool processGenerated = true;
  bool tracksToRecords = false;
  o2::itsmft::TrackingMode::Type mode = o2::itsmft::TrackingMode::Sync;
  int nThreads = 1;
};

struct TrackerOptionAliases {
  int mode = -1;
  int nThreads = 1; // Effective value after applying --nThreads, then configKeyValues.
  bool filterIRFrames = false;
};

// Parameter aliases override CLI values, identically in both entry points.
// Conflicts are reported with both setting names; invalid values throw.
WorkflowOptions resolveWorkflowOptions(const WorkflowOptionInput&, const TrackerOptionAliases&);
WorkflowOptions readWorkflowOptions(const o2::framework::ConfigContext&, WorkflowKind);
} // namespace o2::mft::ca
#endif
