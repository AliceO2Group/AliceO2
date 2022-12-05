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

/// \file   TPCIntegrateClusterCurrent.h
/// \author Matthias Kleiner, mkleiner@ikf.uni-frankfurt.de

#ifndef O2_CALIBRATION_TPCINTEGRATECLUSTERCURRENT_H
#define O2_CALIBRATION_TPCINTEGRATECLUSTERCURRENT_H

#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "TPCWorkflow/ProcessingHelpers.h"
#include "TPCBase/Mapper.h"
#include "DetectorsBase/GRPGeomHelper.h"

using namespace o2::framework;
using o2::header::gDataOriginTPC;
using namespace o2::tpc;

namespace o2
{
namespace tpc
{

class TPCIntegrateClustersDevice : public o2::framework::Task
{
 public:
  TPCIntegrateClustersDevice(std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req) {}

  void init(o2::framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    mNSlicesTF = ic.options().get<int>("nSlicesTF");
    mUseQMax = ic.options().get<bool>("use-qMax");
  }

  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final { o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj); }

  void run(o2::framework::ProcessingContext& pc) final
  {
    // fetch only once
    if (mContinuousMaxTimeBin < 0) {
      o2::base::GRPGeomHelper::instance().checkUpdates(pc);
      mContinuousMaxTimeBin = (base::GRPGeomHelper::instance().getGRPECS()->getNHBFPerTF() * o2::constants::lhc::LHCMaxBunches + 2 * o2::tpc::constants::LHCBCPERTIMEBIN - 2) / o2::tpc::constants::LHCBCPERTIMEBIN;
    }

    const auto& clusters = getWorkflowTPCInput(pc);
    const o2::tpc::ClusterNativeAccess& clIndex = clusters->clusterIndex;
    const auto nCL = clIndex.nClustersTotal;
    LOGP(info, "Processing TF {} with {} clusters", processing_helpers::getCurrentTF(pc), nCL);

    // init only once
    if (mInitICCBuffer) {
      mNTSPerSlice = mContinuousMaxTimeBin / mNSlicesTF;
      for (unsigned int i = 0; i < o2::tpc::CRU::MaxCRU; ++i) {
        mICCS[i].resize(mNSlicesTF * Mapper::PADSPERREGION[CRU(i).region()]);
      }
      mInitICCBuffer = false;
    }

    // loop over clusters and integrate
    for (int isector = 0; isector < constants::MAXSECTOR; ++isector) {
      for (int irow = 0; irow < constants::MAXGLOBALPADROW; ++irow) {
        const int nClusters = clIndex.nClusters[isector][irow];
        if (!nClusters) {
          continue;
        }
        const CRU cru(Sector(isector), Mapper::REGION[irow]);

        for (int icl = 0; icl < nClusters; ++icl) {
          const auto& cl = *(clIndex.clusters[isector][irow] + icl);
          const float time = cl.getTime();
          const unsigned int sliceInTF = time / mNTSPerSlice;
          if (sliceInTF < mNSlicesTF) {
            const float charge = mUseQMax ? cl.getQmax() : cl.getQtot();
            const unsigned int padInCRU = Mapper::OFFSETCRUGLOBAL[irow] + static_cast<int>(cl.getPad() + 0.5f);
            mICCS[cru][sliceInTF * Mapper::PADSPERREGION[cru.region()] + padInCRU] += charge;
          } else {
            LOGP(info, "slice in TF of ICC {} is larger than max slice {} with nTSPerSlice {}", sliceInTF, mNSlicesTF, mNTSPerSlice);
          }
        }
      }
    }
    sendOutput(pc.outputs());
  }

  static constexpr header::DataDescription getDataDescription() { return header::DataDescription{"ICC"}; }

 private:
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;     ///< for accessing the b-field
  int mNSlicesTF{1};                                          ///< number of slices the TFs are divided into for integration of clusters currents
  bool mUseQMax{false};                                       ///< using qMax as cluster current
  int mContinuousMaxTimeBin{-1};                              ///< max time bin of clusters
  std::array<std::vector<float>, o2::tpc::CRU::MaxCRU> mICCS; ///< buffer for ICCs
  bool mInitICCBuffer{true};                                  ///< flag for initializing ICCs only once
  int mNTSPerSlice{1};                                        ///< number of time stamps per slice

  void sendOutput(DataAllocator& output)
  {
    for (unsigned int i = 0; i < o2::tpc::CRU::MaxCRU; ++i) {
      // normalize ICCs
      const float norm = 1. / float(mNTSPerSlice);
      std::transform(mICCS[i].begin(), mICCS[i].end(), mICCS[i].begin(), [norm](float& val) { return val * norm; });

      output.snapshot(Output{gDataOriginTPC, getDataDescription(), o2::header::DataHeader::SubSpecificationType{i << 7}, Lifetime::Timeframe}, mICCS[i]);
      std::fill(mICCS[i].begin(), mICCS[i].end(), 0);
    }
  }
};

DataProcessorSpec getTPCIntegrateClustersSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("clusTPC", ConcreteDataTypeMatcher{gDataOriginTPC, "CLUSTERNATIVE"}, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  for (unsigned int i = 0; i < o2::tpc::CRU::MaxCRU; ++i) {
    outputs.emplace_back(gDataOriginTPC, TPCIntegrateClustersDevice::getDataDescription(), header::DataHeader::SubSpecificationType{i << 7}, o2::framework::Lifetime::Sporadic);
  }

  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);

  return DataProcessorSpec{
    "integrate-tpc-clusters",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCIntegrateClustersDevice>(ccdbRequest)},
    Options{
      {"use-qMax", VariantType::Bool, false, {"Using qMax instead of qTot as cluster current"}},
      {"nSlicesTF", VariantType::Int, 2, {"Divide the TF into n slices"}}}}; // end DataProcessorSpec
}

} // namespace tpc
} // namespace o2

#endif
