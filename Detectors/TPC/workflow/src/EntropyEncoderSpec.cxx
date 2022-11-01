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

/// @file   EntropyEncoderSpec.cxx
/// @author Michael Lettrich, Matthias Richter
/// @since  2020-01-16
/// @brief  ProcessorSpec for the TPC cluster entropy encoding

#include "TPCWorkflow/EntropyEncoderSpec.h"
#include "DataFormatsTPC/CompressedClusters.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "Headers/DataHeader.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "GPUO2InterfaceConfiguration.h"
#include "GPUParam.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "TPCClusterDecompressor.inc"

using namespace o2::framework;
using namespace o2::header;

namespace o2
{
namespace tpc
{

EntropyEncoderSpec::~EntropyEncoderSpec() = default;

EntropyEncoderSpec::EntropyEncoderSpec(bool fromFile, bool selIR) : mCTFCoder(o2::ctf::CTFCoderBase::OpType::Encoder), mFromFile(fromFile), mSelIR(selIR)
{
  mTimer.Stop();
  mTimer.Reset();
}

void EntropyEncoderSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (mCTFCoder.finaliseCCDB<CTF>(matcher, obj)) {
    return;
  }
}

void EntropyEncoderSpec::init(o2::framework::InitContext& ic)
{
  mCTFCoder.init<CTF>(ic);
  mCTFCoder.setCombineColumns(!ic.options().get<bool>("no-ctf-columns-combining"));

  mConfig.reset(new o2::gpu::GPUO2InterfaceConfiguration);
  mConfig->configGRP.solenoidBz = 0;
  mConfParam.reset(new o2::gpu::GPUSettingsO2(mConfig->ReadConfigurableParam()));

  mFastTransform = std::move(TPCFastTransformHelperO2::instance()->create(0));

  mParam.reset(new o2::gpu::GPUParam);
  mParam->SetDefaults(&mConfig->configGRP, &mConfig->configReconstruction, &mConfig->configProcessing, nullptr);
}

void EntropyEncoderSpec::run(ProcessingContext& pc)
{
  mCTFCoder.updateTimeDependentParams(pc);
  CompressedClusters clusters;

  if (mFromFile) {
    auto tmp = pc.inputs().get<CompressedClustersROOT*>("input");
    if (tmp == nullptr) {
      LOG(error) << "invalid input";
      return;
    }
    clusters = *tmp;
  } else {
    auto tmp = pc.inputs().get<CompressedClustersFlat*>("input");
    if (tmp == nullptr) {
      LOG(error) << "invalid input";
      return;
    }
    clusters = *tmp;
  }
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  auto& buffer = pc.outputs().make<std::vector<o2::ctf::BufferType>>(Output{"TPC", "CTFDATA", 0, Lifetime::Timeframe});
  std::vector<bool> rejectHits, rejectTracks, rejectTrackHits, rejectTrackHitsReduced;
  if (mSelIR) {
    CompressedClusters clustersFiltered = clusters;
    mCTFCoder.setSelectedIRFrames(pc.inputs().get<gsl::span<o2::dataformats::IRFrame>>("selIRFrames"));
    rejectHits.resize(clusters.nUnattachedClusters);
    rejectTracks.resize(clusters.nTracks);
    rejectTrackHits.resize(clusters.nAttachedClusters);
    rejectTrackHitsReduced.resize(clusters.nAttachedClustersReduced);

    unsigned int offset = 0, lasti = 0;
    const unsigned int maxTime = (mParam->par.continuousMaxTimeBin + 1) * o2::tpc::ClusterNative::scaleTimePacked - 1;
    for (unsigned int i = 0; i < clusters.nTracks; i++) {
      unsigned int tMin = maxTime, tMax = 0;
      auto checker = [&tMin, &tMax](const o2::tpc::ClusterNative& cl, unsigned int offset) {
        if (cl.getTimePacked() > tMax) {
          tMax = cl.getTimePacked();
        }
        if (cl.getTimePacked() < tMin) {
          tMin = cl.getTimePacked();
        }
      };
      o2::gpu::TPCClusterDecompressor::decompressTrack(&clusters, *mParam, maxTime, i, offset, checker);
      if (false) {
        for (unsigned int k = offset - clusters.nTrackClusters[i]; k < offset; k++) {
          rejectTrackHits[k] = true;
        }
        for (unsigned int k = offset - clusters.nTrackClusters[i] - i; k < offset - i - 1; k++) {
          rejectTrackHitsReduced[k] = true;
        }
        rejectTracks[i] = true;
        clustersFiltered.nTracks--;
      }
    }
    offset = 0;
    for (unsigned int i = 0; i < GPUCA_NSLICES; i++) {
      for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
        if (i * GPUCA_ROW_COUNT + j >= clusters.nSliceRows) {
          break;
        }
        offset += (i * GPUCA_ROW_COUNT + j >= clusters.nSliceRows) ? 0 : clusters.nSliceRowClusters[i * GPUCA_ROW_COUNT + j];
        auto checker = [i, j, &rejectHits, &clustersFiltered](const o2::tpc::ClusterNative& cl, unsigned int k) {
          if (false) {
            rejectHits[k] = true;
            clustersFiltered.nSliceRowClusters[i * GPUCA_ROW_COUNT + j]--;
          }
        };
        unsigned int end = offset + clusters.nSliceRowClusters[i * GPUCA_ROW_COUNT + j];
        o2::gpu::TPCClusterDecompressor::decompressHits(&clusters, offset, end, checker);
      }
    }
    clusters = clustersFiltered;
  }
  auto iosize = mCTFCoder.encode(buffer, clusters, mSelIR ? &rejectHits : nullptr, mSelIR ? &rejectTracks : nullptr, mSelIR ? &rejectTrackHits : nullptr, mSelIR ? &rejectTrackHitsReduced : nullptr);
  pc.outputs().snapshot({"ctfrep", 0}, iosize);
  mTimer.Stop();
  if (mSelIR) {
    mCTFCoder.getIRFramesSelector().clear();
  }
  LOG(info) << iosize.asString() << " in " << mTimer.CpuTime() - cput << " s";
}

void EntropyEncoderSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "TPC Entropy Encoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getEntropyEncoderSpec(bool inputFromFile, bool selIR)
{
  std::vector<InputSpec> inputs;
  header::DataDescription inputType = inputFromFile ? header::DataDescription("COMPCLUSTERS") : header::DataDescription("COMPCLUSTERSFLAT");
  inputs.emplace_back("input", "TPC", inputType, 0, Lifetime::Timeframe);
  inputs.emplace_back("ctfdict", "TPC", "CTFDICT", 0, Lifetime::Condition, ccdbParamSpec("TPC/Calib/CTFDictionary"));
  if (selIR) {
    inputs.emplace_back("selIRFrames", "CTF", "SELIRFRAMES", 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{
    "tpc-entropy-encoder", // process id
    inputs,
    Outputs{{"TPC", "CTFDATA", 0, Lifetime::Timeframe},
            {{"ctfrep"}, "TPC", "CTFENCREP", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<EntropyEncoderSpec>(inputFromFile)},
    Options{{"ctf-dict", VariantType::String, "ccdb", {"CTF dictionary: empty or ccdb=CCDB, none=no external dictionary otherwise: local filename"}},
            {"no-ctf-columns-combining", VariantType::Bool, false, {"Do not combine correlated columns in CTF"}},
            {"irframe-margin-bwd", VariantType::UInt32, 0u, {"margin in BC to add to the IRFrame lower boundary when selection is requested"}},
            {"irframe-margin-fwd", VariantType::UInt32, 0u, {"margin in BC to add to the IRFrame upper boundary when selection is requested"}},
            {"mem-factor", VariantType::Float, 1.f, {"Memory allocation margin factor"}}}};
}

} // namespace tpc
} // namespace o2
