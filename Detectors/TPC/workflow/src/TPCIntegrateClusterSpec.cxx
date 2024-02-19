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

/// \file TPCIntegrateClusterSpec.cxx
/// \brief device for integrating the TPC clusters in time slices
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Jan 30, 2023

#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ControlService.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "TPCWorkflow/ProcessingHelpers.h"
#include "TPCBase/Mapper.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "TPCWorkflow/TPCIntegrateClusterSpec.h"
#include "DetectorsBase/TFIDInfoHelper.h"

#include "DetectorsCalibration/IntegratedClusterCalibrator.h"

#include <algorithm>
#include <numeric>

using namespace o2::framework;

namespace o2
{
namespace tpc
{

class TPCIntegrateClusters : public Task
{
 public:
  /// \constructor
  TPCIntegrateClusters(std::shared_ptr<o2::base::GRPGeomRequest> req, const bool disableWriter) : mCCDBRequest(req), mDisableWriter(disableWriter){};

  void init(framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    mNSlicesTF = ic.options().get<int>("nSlicesTF");
    mHBScaling = ic.options().get<float>("heart-beat-scaling");
    mProcess3D = ic.options().get<bool>("process-3D-currents");
    mNBits = ic.options().get<int>("nBits");
  }

  void run(ProcessingContext& pc) final
  {
    // fetch only once
    if (mContinuousMaxTimeBin < 0) {
      o2::base::GRPGeomHelper::instance().checkUpdates(pc);
      mContinuousMaxTimeBin = (mHBScaling * base::GRPGeomHelper::instance().getGRPECS()->getNHBFPerTF() * o2::constants::lhc::LHCMaxBunches + 2 * o2::tpc::constants::LHCBCPERTIMEBIN - 2) / o2::tpc::constants::LHCBCPERTIMEBIN;
    }

    const auto& clusters = getWorkflowTPCInput(pc);
    const o2::tpc::ClusterNativeAccess& clIndex = clusters->clusterIndex;
    const auto nCL = clIndex.nClustersTotal;
    LOGP(detail, "Processing TF {} with {} clusters", processing_helpers::getCurrentTF(pc), nCL);

    // init only once
    if (mInitICCBuffer) {
      const int slicesNew = static_cast<int>(mNSlicesTF * mHBScaling + 0.5);
      if (slicesNew != mNSlicesTF) {
        LOGP(info, "Adjusting number of slices to {}", slicesNew);
        mNSlicesTF = slicesNew;
      }
      mNTSPerSlice = mContinuousMaxTimeBin / mNSlicesTF;
      mInitICCBuffer = false;
      if (!mProcess3D) {
        mBufferCurrents.resize(mNSlicesTF);
      } else {
        mBufferCurrents.resize(mNSlicesTF * Mapper::getNumberOfPadsPerSide());
      }
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
            const float qMax = cl.getQmax();
            const float qTot = cl.getQtot();
            if (!mProcess3D) {
              if (isector < SECTORSPERSIDE) {
                mBufferCurrents.mIQMaxA[sliceInTF] += qMax;
                mBufferCurrents.mIQTotA[sliceInTF] += qTot;
                ++mBufferCurrents.mINClA[sliceInTF];
              } else {
                mBufferCurrents.mIQMaxC[sliceInTF] += qMax;
                mBufferCurrents.mIQTotC[sliceInTF] += qTot;
                ++mBufferCurrents.mINClC[sliceInTF];
              }
            } else {
              const int pad = static_cast<int>(cl.getPad() + 0.5f);
              const int region = Mapper::REGION[irow];
              const unsigned int index = sliceInTF * Mapper::getNumberOfPadsPerSide() + (isector % SECTORSPERSIDE) * Mapper::getPadsInSector() + Mapper::GLOBALPADOFFSET[region] + Mapper::OFFSETCRUGLOBAL[irow] + pad;

              if (index > mBufferCurrents.mIQMaxA.size()) {
                LOGP(warning, "Index {} is larger than max index {}", index, mBufferCurrents.mIQMaxA.size());
              }

              if (isector < SECTORSPERSIDE) {
                mBufferCurrents.mIQMaxA[index] += qMax;
                mBufferCurrents.mIQTotA[index] += qTot;
                ++mBufferCurrents.mINClA[index];
              } else {
                mBufferCurrents.mIQMaxC[index] += qMax;
                mBufferCurrents.mIQTotC[index] += qTot;
                ++mBufferCurrents.mINClC[index];
              }
            }
          } else {
            LOGP(debug, "slice in TF of ICC {} is larger than max slice {} with nTSPerSlice {}", sliceInTF, mNSlicesTF, mNTSPerSlice);
          }
        }
      }
    }

    const float nTSPerSliceInv = 1. / float(mNTSPerSlice);
    mBufferCurrents.normalize(nTSPerSliceInv);
    sendOutput(pc);
  }

  void endOfStream(EndOfStreamContext& eos) final { eos.services().get<ControlService>().readyToQuit(QuitRequest::Me); }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final { o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj); }

 private:
  int mNSlicesTF = 11;                                    ///< number of slices a TF is divided into
  float mHBScaling = 1;                                   ///< fraction of a TF filled with data (in case only fraction of TF stored)
  const bool mDisableWriter{false};                       ///< flag if no ROOT output will be written
  bool mProcess3D{false};                                 ///< flag if the 3D TPC currents are expected as input
  std::vector<int> mCounterNeighbours;                    ///< buffer for noise removal (contains number of neighbouring cluster for time +-mTimeCutNoisePS)
  ITPCC mBufferCurrents;                                  ///< buffer for integrate currents
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest; ///< info for CCDB request
  int mContinuousMaxTimeBin{-1};                          ///< max time bin of clusters
  bool mInitICCBuffer{true};                              ///< flag for initializing ICCs only once
  int mNTSPerSlice{1};                                    ///< number of time stamps per slice
  int mNBits = 32;                                        ///< number of bits used for rounding

  void sendOutput(ProcessingContext& pc)
  {
    if (mNBits < 32) {
      mBufferCurrents.compress(mNBits);
    }
    pc.outputs().snapshot(Output{header::gDataOriginTPC, getDataDescriptionTPCC()}, mBufferCurrents);
    // in case of ROOT output also store the TFinfo in the TTree
    if (!mDisableWriter) {
      o2::dataformats::TFIDInfo tfinfo;
      o2::base::TFIDInfoHelper::fillTFIDInfo(pc, tfinfo);
      pc.outputs().snapshot(Output{header::gDataOriginTPC, getDataDescriptionTPCTFId()}, tfinfo);
    }
  }
};

o2::framework::DataProcessorSpec getTPCIntegrateClusterSpec(const bool disableWriter)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("clusTPC", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, "CLUSTERNATIVE"}, Lifetime::Timeframe);

  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                                true,                           // GRPECS=true for nHBF per TF
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTPC, getDataDescriptionTPCC(), 0, Lifetime::Timeframe);
  if (!disableWriter) {
    outputs.emplace_back(o2::header::gDataOriginTPC, getDataDescriptionTPCTFId(), 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "TPCIntegrateClusters",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCIntegrateClusters>(ccdbRequest, disableWriter)},
    Options{{"nSlicesTF", VariantType::Int, 11, {"number of slices into which a TF is divided"}},
            {"heart-beat-scaling", VariantType::Float, 1.f, {"fraction of filled TFs (1 full TFs, 0.25 TFs filled only with 25%)"}},
            {"process-3D-currents", VariantType::Bool, false, {"Process full 3D currents instead of 1D integrated only currents"}},
            {"nBits", VariantType::Int, 32, {"Number of bits used for compression/rounding (values >= 32 results in no compression)"}}}};
}

} // namespace tpc
} // end namespace o2
