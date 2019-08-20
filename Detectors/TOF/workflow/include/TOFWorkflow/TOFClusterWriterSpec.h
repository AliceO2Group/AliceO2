// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TOFClusterWriterSpec.h

#ifndef STEER_DIGITIZERWORKFLOW_TOFCLUSTERWRITER_H_
#define STEER_DIGITIZERWORKFLOW_TOFCLUSTERWRITER_H_

#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include <string>

using namespace o2::framework;

namespace o2
{
namespace tof
{

class ClusterWriter : public Task
{
 public:
  ClusterWriter(bool useMC = true) : mUseMC(useMC) {}
  ~ClusterWriter() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  bool mFinished = false;
  bool mUseMC = true;
  std::string mOutFileName; // read from workflow
  std::string mOutTreeName; // read from workflow
};

/// create a processor spec
/// write ITS tracks a root file
o2::framework::DataProcessorSpec getTOFClusterWriterSpec(bool useMC);

} // namespace tof
} // namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_TOFCLUSTERWRITER_H_ */
