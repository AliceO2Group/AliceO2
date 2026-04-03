// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRK_CLUSTERERDPL
#define O2_TRK_CLUSTERERDPL

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "TRKReconstruction/Clusterer.h"
#ifdef O2_WITH_ACTS
#include "TRKReconstruction/ClustererACTS.h"
#endif

namespace o2::trk
{

class ClustererDPL : public o2::framework::Task
{
 public:
  ClustererDPL(bool useMC) : mUseMC(useMC) {}
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 private:
  bool mUseMC = true;
  int mNThreads = 1;
  o2::trk::Clusterer mClusterer;
#ifdef O2_WITH_ACTS
  bool mUseACTS = false;
  o2::trk::ClustererACTS mClustererACTS;
#endif
};

o2::framework::DataProcessorSpec getClustererSpec(bool useMC);

} // namespace o2::trk

#endif
