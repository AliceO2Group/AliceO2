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
#include "GPUTPCCompressionKernels.inc"
#include "TPCCalibration/VDriftHelper.h"

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
  if (mTPCVDriftHelper->accountCCDBInputs(matcher, obj)) {
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

  mTPCVDriftHelper.reset(new VDriftHelper);

  mNThreads = ic.options().get<unsigned int>("nThreads");
  mMaxZ = ic.options().get<float>("irframe-clusters-maxz");
  mMaxEta = ic.options().get<float>("irframe-clusters-maxeta");
}

void EntropyEncoderSpec::run(ProcessingContext& pc)
{
  mCTFCoder.updateTimeDependentParams(pc);
  mTPCVDriftHelper->extractCCDBInputs(pc);
  if (mTPCVDriftHelper->isUpdated()) {
    TPCFastTransformHelperO2::instance()->updateCalibration(*mFastTransform, 0, mTPCVDriftHelper->getVDriftObject().corrFact, mTPCVDriftHelper->getVDriftObject().refVDrift);
  }

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
  CompressedClusters clustersFiltered = clusters;
  std::vector<std::pair<std::vector<unsigned int>, std::vector<unsigned short>>> tmpBuffer(std::max<int>(mNThreads, 1));
  if (mSelIR) {
    mCTFCoder.setSelectedIRFrames(pc.inputs().get<gsl::span<o2::dataformats::IRFrame>>("selIRFrames"));
    rejectHits.resize(clusters.nUnattachedClusters);
    rejectTracks.resize(clusters.nTracks);
    rejectTrackHits.resize(clusters.nAttachedClusters);
    rejectTrackHitsReduced.resize(clusters.nAttachedClustersReduced);

    const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
    const auto firstIR = o2::InteractionRecord(0, tinfo.firstTForbit);
    const float totalT = mFastTransform->convDeltaZtoDeltaTimeInTimeFrameAbs(250);

    unsigned int offset = 0, lasti = 0;
    const unsigned int maxTime = (mParam->par.continuousMaxTimeBin + 1) * o2::tpc::ClusterNative::scaleTimePacked - 1;
#ifdef WITH_OPENMP
#pragma omp parallel for firstprivate(offset, lasti) num_threads(mNThreads)
#endif
    for (unsigned int i = 0; i < clusters.nTracks; i++) {
      unsigned int tMinP = maxTime, tMaxP = 0;
      auto checker = [&tMinP, &tMaxP](const o2::tpc::ClusterNative& cl, unsigned int offset) {
        if (cl.getTimePacked() > tMaxP) {
          tMaxP = cl.getTimePacked();
        }
        if (cl.getTimePacked() < tMinP) {
          tMinP = cl.getTimePacked();
        }
      };
      if (i < lasti) {
        offset = lasti = 0; // dynamic OMP scheduling, need to reinitialize offset
      }
      while (lasti < i) {
        offset += clusters.nTrackClusters[lasti++];
      }
      lasti++;
      o2::gpu::TPCClusterDecompressor::decompressTrack(&clusters, *mParam, maxTime, i, offset, checker);
      const float tMin = o2::tpc::ClusterNative::unpackTime(tMinP), tMax = o2::tpc::ClusterNative::unpackTime(tMaxP);
      const auto chkVal = firstIR + (tMin * constants::LHCBCPERTIMEBIN);
      const auto chkExt = (totalT - (tMax - tMin)) * constants::LHCBCPERTIMEBIN + 1;
      const bool reject = mCTFCoder.getIRFramesSelector().check(o2::dataformats::IRFrame(chkVal, chkVal + 1), chkExt, 0) < 0;
      if (reject) {
        for (unsigned int k = offset - clusters.nTrackClusters[i]; k < offset; k++) {
          rejectTrackHits[k] = true;
        }
        for (unsigned int k = offset - clusters.nTrackClusters[i] - i; k < offset - i - 1; k++) {
          rejectTrackHitsReduced[k] = true;
        }
        rejectTracks[i] = true;
        static std::atomic_flag lock = ATOMIC_FLAG_INIT;
        while (lock.test_and_set(std::memory_order_acquire)) {
        }
        clustersFiltered.nTracks--;
        clustersFiltered.nAttachedClusters -= clusters.nTrackClusters[i];
        clustersFiltered.nAttachedClustersReduced -= clusters.nTrackClusters[i] - 1;
        lock.clear(std::memory_order_release);
      }
    }
    offset = 0;
    unsigned int offsets[GPUCA_NSLICES][GPUCA_ROW_COUNT];
    for (unsigned int i = 0; i < GPUCA_NSLICES; i++) {
      for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
        if (i * GPUCA_ROW_COUNT + j >= clusters.nSliceRows) {
          break;
        }
        offsets[i][j] = offset;
        offset += (i * GPUCA_ROW_COUNT + j >= clusters.nSliceRows) ? 0 : clusters.nSliceRowClusters[i * GPUCA_ROW_COUNT + j];
      }
    }

#ifdef WITH_OPENMP
#pragma omp parallel for num_threads(mNThreads) schedule(static, (GPUCA_NSLICES + mNThreads - 1) / mNThreads) // Static round-robin scheduling with one chunk per thread to ensure correct order of the final vector
#endif
    for (unsigned int ii = 0; ii < clusters.nSliceRows; ii++) {
      unsigned int i = ii / GPUCA_ROW_COUNT;
      unsigned int j = ii % GPUCA_ROW_COUNT;
      o2::tpc::ClusterNative preCl;
#ifdef WITH_OPENMP
      int myThread = omp_get_thread_num();
#else
      int myThread = 0;
#endif
      unsigned int count = 0;
      auto checker = [i, j, firstIR, totalT, this, &preCl, &count, &outBuffer = tmpBuffer[myThread], &rejectHits, &clustersFiltered](const o2::tpc::ClusterNative& cl, unsigned int k) {
        const auto chkVal = firstIR + (cl.getTime() * constants::LHCBCPERTIMEBIN);
        const auto chkExt = totalT * constants::LHCBCPERTIMEBIN;
        const bool reject = mCTFCoder.getIRFramesSelector().check(o2::dataformats::IRFrame(chkVal, chkVal + 1), chkExt, 0) < 0;
        if (reject) {
          rejectHits[k] = true;
          clustersFiltered.nSliceRowClusters[i * GPUCA_ROW_COUNT + j]--;
          static std::atomic_flag lock = ATOMIC_FLAG_INIT;
          while (lock.test_and_set(std::memory_order_acquire)) {
          }
          clustersFiltered.nUnattachedClusters--;
          lock.clear(std::memory_order_release);
        } else {
          outBuffer.first.emplace_back(0);
          outBuffer.second.emplace_back(0);
          GPUTPCCompression_EncodeUnattached(clustersFiltered.nComppressionModes, cl, outBuffer.first.back(), outBuffer.second.back(), count++ ? &preCl : nullptr);
          preCl = cl;
        }
      };
      unsigned int end = offsets[i][j] + clusters.nSliceRowClusters[i * GPUCA_ROW_COUNT + j];
      o2::gpu::TPCClusterDecompressor::decompressHits(&clusters, offsets[i][j], end, checker);
    }
    tmpBuffer[0].first.reserve(clustersFiltered.nUnattachedClusters);
    tmpBuffer[0].second.reserve(clustersFiltered.nUnattachedClusters);
    for (int i = 1; i < mNThreads; i++) {
      tmpBuffer[0].first.insert(tmpBuffer[0].first.end(), tmpBuffer[i].first.begin(), tmpBuffer[i].first.end());
      tmpBuffer[i].first.clear();
      tmpBuffer[0].second.insert(tmpBuffer[0].second.end(), tmpBuffer[i].second.begin(), tmpBuffer[i].second.end());
      tmpBuffer[i].second.clear();
    }
    clustersFiltered.timeDiffU = tmpBuffer[0].first.data();
    clustersFiltered.padDiffU = tmpBuffer[0].second.data();
  }
  auto iosize = mCTFCoder.encode(buffer, clusters, clustersFiltered, mSelIR ? &rejectHits : nullptr, mSelIR ? &rejectTracks : nullptr, mSelIR ? &rejectTrackHits : nullptr, mSelIR ? &rejectTrackHitsReduced : nullptr);
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
  o2::tpc::VDriftHelper::requestCCDBInputs(inputs);
  if (selIR) {
    inputs.emplace_back("selIRFrames", "CTF", "SELIRFRAMES", 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{
    "tpc-entropy-encoder", // process id
    inputs,
    Outputs{{"TPC", "CTFDATA", 0, Lifetime::Timeframe},
            {{"ctfrep"}, "TPC", "CTFENCREP", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<EntropyEncoderSpec>(inputFromFile, selIR)},
    Options{{"ctf-dict", VariantType::String, "ccdb", {"CTF dictionary: empty or ccdb=CCDB, none=no external dictionary otherwise: local filename"}},
            {"no-ctf-columns-combining", VariantType::Bool, false, {"Do not combine correlated columns in CTF"}},
            {"irframe-margin-bwd", VariantType::UInt32, 0u, {"margin in BC to add to the IRFrame lower boundary when selection is requested"}},
            {"irframe-margin-fwd", VariantType::UInt32, 0u, {"margin in BC to add to the IRFrame upper boundary when selection is requested"}},
            {"irframe-clusters-maxeta", VariantType::Float, 1.5f, {"Max eta for non-assigned clusters"}},
            {"irframe-clusters-maxz", VariantType::Float, 25.f, {"Max z for non assigned clusters (combined with maxeta)"}},
            {"mem-factor", VariantType::Float, 1.f, {"Memory allocation margin factor"}},
            {"nThreads", VariantType::UInt32, 1u, {"number of threads to use for decoding"}}}};
}

} // namespace tpc
} // namespace o2
