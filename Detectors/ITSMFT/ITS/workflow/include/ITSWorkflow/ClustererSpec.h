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

/// @file   ClustererSpec.h

#ifndef O2_ITS_CLUSTERERDPL
#define O2_ITS_CLUSTERERDPL

#include <fstream>
#include "DetectorsBase/GRPGeomHelper.h"
#include "ITSMFTReconstruction/Clusterer.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

using namespace o2::framework;

namespace o2
{

namespace itsmft
{
class Clusterer;
}

namespace its
{

class ClustererDPL : public Task
{
 public:
  ClustererDPL(std::shared_ptr<o2::base::GRPGeomRequest> gr, bool useMC) : mGGCCDBRequest(gr), mUseMC(useMC) {}
  ~ClustererDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  void updateTimeDependentParams(ProcessingContext& pc);

  int mState = 0;
  bool mUseMC = true;
  bool mUseClusterDictionary = true;
  int mNThreads = 1;
  std::unique_ptr<std::ifstream> mFile = nullptr;
  std::unique_ptr<o2::itsmft::Clusterer> mClusterer = nullptr;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
};

/// create a processor spec
/// run ITS cluster finder
framework::DataProcessorSpec getClustererSpec(bool useMC);

} // namespace its
} // namespace o2

#endif /* O2_ITS_CLUSTERERDPL */
