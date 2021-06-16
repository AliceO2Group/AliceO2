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

/// @file   ClusterReaderSpec.h

#ifndef O2_MFT_CLUSTERREADER_H_
#define O2_MFT_CLUSTERREADER_H_

#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

namespace o2
{
namespace mft
{

class ClusterReader : public o2::framework::Task
{
 public:
  ClusterReader(bool useMC) : mUseMC(useMC) {}
  ~ClusterReader() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 private:
  int mState = 0;
  bool mUseMC = true;
  std::unique_ptr<TFile> mFile = nullptr;
};

/// create a processor spec
/// read simulated MFT digits from a root file
o2::framework::DataProcessorSpec getClusterReaderSpec(bool useMC);

} // namespace mft
} // namespace o2

#endif /* O2_MFT_CLUSTERREADER */
