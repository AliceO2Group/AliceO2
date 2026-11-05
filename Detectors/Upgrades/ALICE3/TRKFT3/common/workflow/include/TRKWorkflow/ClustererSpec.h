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

#include "DetectorsCommonDataFormats/DetID.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "TRKReconstruction/Clusterer.h"
#ifdef O2_WITH_ACTS
#include "TRKReconstruction/ClustererACTS.h"
#endif

namespace o2::trk
{

/// Clusterises the digits of TRK or FT3, layer by layer.
template <int DetID>
class ClustererDPL : public o2::framework::Task
{
 public:
  ClustererDPL(bool useMC) : mUseMC(useMC) {}
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 private:
  bool mUseMC = true;
  int mNThreads = 1;
  Clusterer<DetID> mClusterer;
#ifdef O2_WITH_ACTS
  bool mUseACTS = false; // TRK only
  o2::trk::ClustererACTS mClustererACTS;
#endif
};

/// The clusterer of TRK's digits.
o2::framework::DataProcessorSpec getClustererSpec(bool useMC);
/// The clusterer of FT3's digits.
o2::framework::DataProcessorSpec getFT3ClustererSpec(bool useMC);

} // namespace o2::trk

#endif
