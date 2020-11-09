// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCATracking.cxx
/// \author David Rohr

#include "TPCReconstruction/GPUCATracking.h"

#include "FairLogger.h"
#include "ReconstructionDataFormats/Track.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "TChain.h"
#include "TClonesArray.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/PadRegionInfo.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/Digit.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "DetectorsRaw/HBFUtils.h"

#include "GPUO2Interface.h"
#include "GPUO2InterfaceConfiguration.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "GPUTPCGMMergerTypes.h"
#include "GPUHostDataTypes.h"
#include "TPCFastTransform.h"

#include <atomic>
#ifdef WITH_OPENMP
#include <omp.h>
#endif

using namespace o2::gpu;
using namespace o2::tpc;
using namespace o2;
using namespace o2::dataformats;

GPUCATracking::GPUCATracking() = default;
GPUCATracking::~GPUCATracking() { deinitialize(); }

int GPUCATracking::initialize(const GPUO2InterfaceConfiguration& config)
{
  mTrackingCAO2Interface.reset(new GPUTPCO2Interface);
  int retVal = mTrackingCAO2Interface->Initialize(config);
  if (retVal) {
    mTrackingCAO2Interface.reset();
  }
  return (retVal);
}

void GPUCATracking::deinitialize()
{
  mTrackingCAO2Interface.reset();
}

int GPUCATracking::runTracking(GPUO2InterfaceIOPtrs* data, GPUInterfaceOutputs* outputs)
{
  if ((int)(data->tpcZS != nullptr) + (int)(data->o2Digits != nullptr && (data->tpcZS == nullptr || data->o2DigitsMC == nullptr)) + (int)(data->clusters != nullptr) + (int)(data->compressedClusters != nullptr) != 1) {
    throw std::runtime_error("Invalid input for gpu tracking");
  }

  constexpr unsigned char flagsReject = GPUTPCGMMergedTrackHit::flagReject | GPUTPCGMMergedTrackHit::flagNotFit;
  const unsigned int flagsRequired = mTrackingCAO2Interface->getConfig().configInterface.dropSecondaryLegs ? gputpcgmmergertypes::attachGoodLeg : 0;

  std::vector<TrackTPC>* outputTracks = data->outputTracks;
  std::vector<uint32_t>* outClusRefs = data->outputClusRefs;
  std::vector<o2::MCCompLabel>* outputTracksMCTruth = data->outputTracksMCTruth;

  if (!outputTracks || !outClusRefs) {
    LOG(ERROR) << "Output tracks or clusRefs vectors are not initialized";
    return 0;
  }
  auto& detParam = ParameterDetector::Instance();
  auto& gasParam = ParameterGas::Instance();
  auto& elParam = ParameterElectronics::Instance();
  float vzbin = (elParam.ZbinWidth * gasParam.DriftV);
  Mapper& mapper = Mapper::instance();

  std::vector<o2::tpc::Digit> gpuDigits[Sector::MAXSECTOR];
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> gpuDigitsMC[Sector::MAXSECTOR];
  ClusterNativeAccess::ConstMCLabelContainerViewWithBuffer gpuDigitsMCConst[Sector::MAXSECTOR];

  GPUTrackingInOutDigits gpuDigitsMap;
  GPUTPCDigitsMCInput gpuDigitsMapMC;
  GPUTrackingInOutPointers ptrs;

  ptrs.tpcCompressedClusters = data->compressedClusters;
  ptrs.tpcZS = data->tpcZS;
  if (data->o2Digits) {
    const float zsThreshold = mTrackingCAO2Interface->getConfig().configReconstruction.tpcZSthreshold;
    const int maxContTimeBin = mTrackingCAO2Interface->getConfig().configEvent.continuousMaxTimeBin;
    for (int i = 0; i < Sector::MAXSECTOR; i++) {
      const auto& d = (*(data->o2Digits))[i];
      if (zsThreshold > 0 && data->tpcZS == nullptr) {
        gpuDigits[i].reserve(d.size());
      }
      for (int j = 0; j < d.size(); j++) {
        if (maxContTimeBin && d[j].getTimeStamp() >= maxContTimeBin) {
          throw std::runtime_error("Digit time bin exceeds time frame length");
        }
        if (zsThreshold > 0 && data->tpcZS == nullptr) {
          if (d[j].getChargeFloat() >= zsThreshold) {
            if (data->o2DigitsMC) {
              for (const auto& element : (*data->o2DigitsMC)[i]->getLabels(j)) {
                gpuDigitsMC[i].addElement(gpuDigits[i].size(), element);
              }
            }
            gpuDigits[i].emplace_back(d[j]);
          }
        }
      }
      if (zsThreshold > 0 && data->tpcZS == nullptr) {
        gpuDigitsMap.tpcDigits[i] = gpuDigits[i].data();
        gpuDigitsMap.nTPCDigits[i] = gpuDigits[i].size();
        if (data->o2DigitsMC) {
          gpuDigitsMC[i].flatten_to(gpuDigitsMCConst[i].first);
          gpuDigitsMCConst[i].second = gpuDigitsMCConst[i].first;
          gpuDigitsMapMC.v[i] = &gpuDigitsMCConst[i].second;
        }
      } else {
        gpuDigitsMap.tpcDigits[i] = (*(data->o2Digits))[i].data();
        gpuDigitsMap.nTPCDigits[i] = (*(data->o2Digits))[i].size();
        if (data->o2DigitsMC) {
          gpuDigitsMapMC.v[i] = (*data->o2DigitsMC)[i];
        }
      }
    }
    if (data->o2DigitsMC) {
      gpuDigitsMap.tpcDigitsMC = &gpuDigitsMapMC;
    }
    ptrs.tpcPackedDigits = &gpuDigitsMap;
  }
  ptrs.clustersNative = data->clusters;
  int retVal = mTrackingCAO2Interface->RunTracking(&ptrs, outputs);
  if (data->o2Digits || data->tpcZS || data->compressedClusters) {
    data->clusters = ptrs.clustersNative;
  }
  data->compressedClusters = ptrs.tpcCompressedClusters;
  const GPUTPCGMMergedTrack* tracks = ptrs.mergedTracks;
  int nTracks = ptrs.nMergedTracks;
  const GPUTPCGMMergedTrackHit* trackClusters = ptrs.mergedTrackHits;

  if (retVal) {
    return retVal;
  }

  std::vector<std::pair<int, float>> trackSort(nTracks);
  int tmp = 0, tmp2 = 0;
  uint32_t clBuff = 0;
  for (char cside = 0; cside < 2; cside++) {
    for (int i = 0; i < nTracks; i++) {
      if (tracks[i].OK() && tracks[i].CSide() == cside) {
        trackSort[tmp++] = {i, tracks[i].GetParam().GetTZOffset()};
        auto ncl = tracks[i].NClusters();
        clBuff += ncl + (ncl + 1) / 2; // actual N clusters to store will be less
      }
    }
    std::sort(trackSort.data() + tmp2, trackSort.data() + tmp,
              [](const auto& a, const auto& b) { return (a.second > b.second); });
    tmp2 = tmp;
  }
  nTracks = tmp;
  outputTracks->resize(nTracks);
  outClusRefs->resize(clBuff);

  std::atomic_int clusterOffsetCounter;
  clusterOffsetCounter.store(0);

  constexpr float MinDelta = 0.1;

#ifdef WITH_OPENMP
#pragma omp parallel for if(!outputTracksMCTruth) num_threads(4)
#endif
  for (int iTmp = 0; iTmp < nTracks; iTmp++) {
    auto& oTrack = (*outputTracks)[iTmp];
    const int i = trackSort[iTmp].first;

    oTrack =
      TrackTPC(tracks[i].GetParam().GetX(), tracks[i].GetAlpha(),
               {tracks[i].GetParam().GetY(), tracks[i].GetParam().GetZ(), tracks[i].GetParam().GetSinPhi(),
                tracks[i].GetParam().GetDzDs(), tracks[i].GetParam().GetQPt()},
               {tracks[i].GetParam().GetCov(0), tracks[i].GetParam().GetCov(1), tracks[i].GetParam().GetCov(2),
                tracks[i].GetParam().GetCov(3), tracks[i].GetParam().GetCov(4), tracks[i].GetParam().GetCov(5),
                tracks[i].GetParam().GetCov(6), tracks[i].GetParam().GetCov(7), tracks[i].GetParam().GetCov(8),
                tracks[i].GetParam().GetCov(9), tracks[i].GetParam().GetCov(10), tracks[i].GetParam().GetCov(11),
                tracks[i].GetParam().GetCov(12), tracks[i].GetParam().GetCov(13), tracks[i].GetParam().GetCov(14)});

    oTrack.setChi2(tracks[i].GetParam().GetChi2());
    auto& outerPar = tracks[i].OuterParam();
    oTrack.setdEdx(tracks[i].dEdxInfo());
    oTrack.setOuterParam(o2::track::TrackParCov(
      outerPar.X, outerPar.alpha,
      {outerPar.P[0], outerPar.P[1], outerPar.P[2], outerPar.P[3], outerPar.P[4]},
      {outerPar.C[0], outerPar.C[1], outerPar.C[2], outerPar.C[3], outerPar.C[4], outerPar.C[5],
       outerPar.C[6], outerPar.C[7], outerPar.C[8], outerPar.C[9], outerPar.C[10], outerPar.C[11],
       outerPar.C[12], outerPar.C[13], outerPar.C[14]}));
    int nOutCl = 0;
    for (int j = 0; j < tracks[i].NClusters(); j++) {
      if ((trackClusters[tracks[i].FirstClusterRef() + j].state & flagsReject) || (ptrs.mergedTrackHitAttachment[trackClusters[tracks[i].FirstClusterRef() + j].num] & flagsRequired) != flagsRequired) {
        continue;
      }
      nOutCl++;
    }
    clBuff = clusterOffsetCounter.fetch_add(nOutCl + (nOutCl + 1) / 2);
    oTrack.setClusterRef(clBuff, nOutCl);         // register the references
    uint32_t* clIndArr = &(*outClusRefs)[clBuff]; // cluster indices start here
    uint8_t* sectorIndexArr = reinterpret_cast<uint8_t*>(clIndArr + nOutCl);
    uint8_t* rowIndexArr = sectorIndexArr + nOutCl;

    std::vector<std::pair<MCCompLabel, unsigned int>> labels;
    int nOutCl2 = 0;
    float t1, t2;
    int sector1, sector2;
    for (int j = 0; j < tracks[i].NClusters(); j++) {
      if ((trackClusters[tracks[i].FirstClusterRef() + j].state & flagsReject) || (ptrs.mergedTrackHitAttachment[trackClusters[tracks[i].FirstClusterRef() + j].num] & flagsRequired) != flagsRequired) {
        continue;
      }
      int clusterIdGlobal = trackClusters[tracks[i].FirstClusterRef() + j].num;
      Sector sector = trackClusters[tracks[i].FirstClusterRef() + j].slice;
      int globalRow = trackClusters[tracks[i].FirstClusterRef() + j].row;
      int clusterIdInRow = clusterIdGlobal - data->clusters->clusterOffset[sector][globalRow];
      int regionNumber = 0;
      while (globalRow > mapper.getGlobalRowOffsetRegion(regionNumber) + mapper.getNumberOfRowsRegion(regionNumber)) {
        regionNumber++;
      }
      clIndArr[nOutCl2] = clusterIdInRow;
      sectorIndexArr[nOutCl2] = sector;
      rowIndexArr[nOutCl2] = globalRow;
      if (nOutCl2 == 0) {
        t1 = data->clusters->clustersLinear[clusterIdGlobal].getTime();
        sector1 = sector;
      }
      nOutCl2++;
      if (nOutCl2 == nOutCl) {
        t2 = data->clusters->clustersLinear[clusterIdGlobal].getTime();
        sector2 = sector;
      }
      if (outputTracksMCTruth && data->clusters->clustersMCTruth) {
        for (const auto& element : data->clusters->clustersMCTruth->getLabels(clusterIdGlobal)) {
          bool found = false;
          for (int l = 0; l < labels.size(); l++) {
            if (labels[l].first == element) {
              labels[l].second++;
              found = true;
              break;
            }
          }
          if (!found) {
            labels.emplace_back(element, 1);
          }
        }
      }
    }

    bool cce = tracks[i].CCE() && ((sector1 < Sector::MAXSECTOR / 2) ^ (sector2 < Sector::MAXSECTOR / 2));
    float time0 = 0.f, tFwd = 0.f, tBwd = 0.f;
    if (mTrackingCAO2Interface->GetParamContinuous()) {
      time0 = tracks[i].GetParam().GetTZOffset();

      if (cce) {
        bool lastSide = trackClusters[tracks[i].FirstClusterRef()].slice < Sector::MAXSECTOR / 2;
        float delta = 0.f;
        for (int iCl = 1; iCl < tracks[i].NClusters(); iCl++) {
          if (lastSide ^ (trackClusters[tracks[i].FirstClusterRef() + iCl].slice < Sector::MAXSECTOR / 2)) {
            auto& cacl1 = trackClusters[tracks[i].FirstClusterRef() + iCl];
            auto& cacl2 = trackClusters[tracks[i].FirstClusterRef() + iCl - 1];
            auto& cl1 = data->clusters->clustersLinear[cacl1.num];
            auto& cl2 = data->clusters->clustersLinear[cacl2.num];
            delta = fabs(cl1.getTime() - cl2.getTime()) * 0.5f;
            if (delta < MinDelta) {
              delta = MinDelta;
            }
            break;
          }
        }
        tFwd = tBwd = delta;
      } else {
        // estimate max/min time increments which still keep track in the physical limits of the TPC
        auto times = std::minmax(t1, t2);
        tFwd = times.first - time0;
        tBwd = time0 - times.second + mTrackingCAO2Interface->getConfig().configCalib.fastTransform->getMaxDriftTime(t1 > t2 ? sector1 : sector2);
      }
    }
    oTrack.setTime0(time0);
    oTrack.setDeltaTBwd(tBwd);
    oTrack.setDeltaTFwd(tFwd);
    if (cce) {
      oTrack.setHasCSideClusters();
      oTrack.setHasASideClusters();
    } else if (tracks[i].CSide()) {
      oTrack.setHasCSideClusters();
    } else {
      oTrack.setHasASideClusters();
    }

    if (outputTracksMCTruth) {
      if (labels.size() == 0) {
        outputTracksMCTruth->emplace_back(); //default constructor creates NotSet label
      } else {
        int bestLabelNum = 0, bestLabelCount = 0;
        for (int j = 0; j < labels.size(); j++) {
          if (labels[j].second > bestLabelCount) {
            bestLabelNum = j;
            bestLabelCount = labels[j].second;
          }
        }
        MCCompLabel& bestLabel = labels[bestLabelNum].first;
        if (bestLabelCount < (1.f - sTrackMCMaxFake) * nOutCl) {
          bestLabel.setFakeFlag();
        }
        outputTracksMCTruth->emplace_back(bestLabel);
      }
    }
  }
  outClusRefs->resize(clusterOffsetCounter.load()); // remove overhead

  mTrackingCAO2Interface->Clear(false);

  return (retVal);
}

float GPUCATracking::getPseudoVDrift()
{
  auto& gasParam = ParameterGas::Instance();
  auto& elParam = ParameterElectronics::Instance();
  return (elParam.ZbinWidth * gasParam.DriftV);
}

void GPUCATracking::GetClusterErrors2(int row, float z, float sinPhi, float DzDs, short clusterState, float& ErrY2, float& ErrZ2) const
{
  if (mTrackingCAO2Interface == nullptr) {
    return;
  }
  mTrackingCAO2Interface->GetClusterErrors2(row, z, sinPhi, DzDs, clusterState, ErrY2, ErrZ2);
}

int GPUCATracking::registerMemoryForGPU(const void* ptr, size_t size)
{
  return mTrackingCAO2Interface->registerMemoryForGPU(ptr, size);
}

int GPUCATracking::unregisterMemoryForGPU(const void* ptr)
{
  return mTrackingCAO2Interface->unregisterMemoryForGPU(ptr);
}
