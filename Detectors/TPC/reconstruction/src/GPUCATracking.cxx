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
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TChain.h"
#include "TClonesArray.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/PadRegionInfo.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/Digit.h"
#include "DetectorsRaw/HBFUtils.h"

#include "GPUO2Interface.h"
#include "GPUO2InterfaceConfiguration.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "GPUHostDataTypes.h"

using namespace o2::gpu;
using namespace o2::tpc;
using namespace o2;
using namespace o2::dataformats;

using MCLabelContainer = MCTruthContainer<MCCompLabel>;

GPUCATracking::GPUCATracking() : mTrackingCAO2Interface() {}
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
  if ((int)(data->tpcZS != nullptr) + (int)(data->o2Digits != nullptr) + (int)(data->clusters != nullptr) != 1) {
    return 0;
  }

  std::vector<TrackTPC>* outputTracks = data->outputTracks;
  std::vector<uint32_t>* outClusRefs = data->outputClusRefs;
  MCLabelContainer* outputTracksMCTruth = data->outputTracksMCTruth;

  if (!outputTracks || !outClusRefs) {
    LOG(ERROR) << "Output tracks or clusRefs vectors are not initialized";
    return 0;
  }
  auto& detParam = ParameterDetector::Instance();
  auto& gasParam = ParameterGas::Instance();
  auto& elParam = ParameterElectronics::Instance();
  float vzbin = (elParam.ZbinWidth * gasParam.DriftV);
  float vzbinInv = 1.f / vzbin;
  Mapper& mapper = Mapper::instance();

  const ClusterNativeAccess* clusters;
  std::vector<o2::tpc::Digit> gpuDigits[Sector::MAXSECTOR];
  GPUTrackingInOutDigits gpuDigitsMap;
  GPUTPCDigitsMCInput gpuDigitsMC;
  GPUTrackingInOutPointers ptrs;

  if (data->tpcZS) {
    ptrs.tpcZS = data->tpcZS;
  } else if (data->o2Digits) {
    ptrs.clustersNative = nullptr;
    const float zsThreshold = mTrackingCAO2Interface->getConfig().configReconstruction.tpcZSthreshold;
    const int maxContTimeBin = (o2::raw::HBFUtils::Instance().getNOrbitsPerTF() * o2::constants::lhc::LHCMaxBunches + Constants::LHCBCPERTIMEBIN - 1) / Constants::LHCBCPERTIMEBIN;
    for (int i = 0; i < Sector::MAXSECTOR; i++) {
      const auto& d = (*(data->o2Digits))[i];
      gpuDigits[i].reserve(d.size());
      gpuDigitsMap.tpcDigits[i] = gpuDigits[i].data();
      for (int j = 0; j < d.size(); j++) {
        if (d[j].getTimeStamp() >= maxContTimeBin) {
          throw std::runtime_error("Digit time bin exceeds time frame length");
        }
        if (d[j].getChargeFloat() >= zsThreshold) {
          gpuDigits[i].emplace_back(d[j]);
        }
      }
      gpuDigitsMap.nTPCDigits[i] = gpuDigits[i].size();
    }
    if (data->o2DigitsMC) {
      for (int i = 0; i < Sector::MAXSECTOR; i++) {
        gpuDigitsMC.v[i] = (*data->o2DigitsMC)[i].get();
      }
      gpuDigitsMap.tpcDigitsMC = &gpuDigitsMC;
    }
    ptrs.tpcPackedDigits = &gpuDigitsMap;
  } else {
    clusters = data->clusters;
    ptrs.clustersNative = clusters;
    ptrs.tpcPackedDigits = nullptr;
  }
  int retVal = mTrackingCAO2Interface->RunTracking(&ptrs, outputs);
  if (data->o2Digits || data->tpcZS) {
    clusters = ptrs.clustersNative;
  }
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
    if (cside == 0)
      mNTracksASide = tmp;
  }
  nTracks = tmp;

  outputTracks->resize(nTracks);
  outClusRefs->resize(clBuff);
  clBuff = 0;

  for (int iTmp = 0; iTmp < nTracks; iTmp++) {
    auto& oTrack = (*outputTracks)[iTmp];
    const int i = trackSort[iTmp].first;
    float time0 = 0.f, tFwd = 0.f, tBwd = 0.f;

    if (mTrackingCAO2Interface->GetParamContinuous()) {
      time0 = tracks[i].GetParam().GetTZOffset();

      if (tracks[i].CCE()) {
        bool lastSide = trackClusters[tracks[i].FirstClusterRef()].slice < Sector::MAXSECTOR / 2;
        float delta = 0.f;
        for (int iCl = 1; iCl < tracks[i].NClusters(); iCl++) {
          if (lastSide ^ (trackClusters[tracks[i].FirstClusterRef() + iCl].slice < Sector::MAXSECTOR / 2)) {
            auto& cacl1 = trackClusters[tracks[i].FirstClusterRef() + iCl];
            auto& cacl2 = trackClusters[tracks[i].FirstClusterRef() + iCl - 1];
            auto& cl1 = clusters->clustersLinear[cacl1.num];
            auto& cl2 = clusters->clustersLinear[cacl2.num];
            delta = fabs(cl1.getTime() - cl2.getTime()) * 0.5f;
            break;
          }
        }
        tFwd = tBwd = delta;
      } else {
        // estimate max/min time increments which still keep track in the physical limits of the TPC
        auto& c1 = trackClusters[tracks[i].FirstClusterRef()];
        auto& c2 = trackClusters[tracks[i].FirstClusterRef() + tracks[i].NClusters() - 1];
        float t1 = clusters->clustersLinear[c1.num].getTime();
        float t2 = clusters->clustersLinear[c2.num].getTime();
        auto times = std::minmax(t1, t2);
        tFwd = times.first - time0;
        tBwd = time0 - (times.second - detParam.TPClength * vzbinInv);
      }
    }

    oTrack =
      TrackTPC(tracks[i].GetParam().GetX(), tracks[i].GetAlpha(),
               {tracks[i].GetParam().GetY(), tracks[i].GetParam().GetZ(), tracks[i].GetParam().GetSinPhi(),
                tracks[i].GetParam().GetDzDs(), tracks[i].GetParam().GetQPt()},
               {tracks[i].GetParam().GetCov(0), tracks[i].GetParam().GetCov(1), tracks[i].GetParam().GetCov(2),
                tracks[i].GetParam().GetCov(3), tracks[i].GetParam().GetCov(4), tracks[i].GetParam().GetCov(5),
                tracks[i].GetParam().GetCov(6), tracks[i].GetParam().GetCov(7), tracks[i].GetParam().GetCov(8),
                tracks[i].GetParam().GetCov(9), tracks[i].GetParam().GetCov(10), tracks[i].GetParam().GetCov(11),
                tracks[i].GetParam().GetCov(12), tracks[i].GetParam().GetCov(13), tracks[i].GetParam().GetCov(14)});
    oTrack.setTime0(time0);
    oTrack.setDeltaTBwd(tBwd);
    oTrack.setDeltaTFwd(tFwd);
    if (tracks[i].CCE()) {
      oTrack.setHasCSideClusters();
      oTrack.setHasASideClusters();
    } else if (tracks[i].CSide()) {
      oTrack.setHasCSideClusters();
    } else {
      oTrack.setHasASideClusters();
    }

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
      if (!(trackClusters[tracks[i].FirstClusterRef() + j].state & GPUTPCGMMergedTrackHit::flagReject)) {
        nOutCl++;
      }
    }
    oTrack.setClusterRef(clBuff, nOutCl);         // register the references
    uint32_t* clIndArr = &(*outClusRefs)[clBuff]; // cluster indices start here
    uint8_t* sectorIndexArr = reinterpret_cast<uint8_t*>(clIndArr + nOutCl);
    uint8_t* rowIndexArr = sectorIndexArr + nOutCl;

    clBuff += nOutCl + (nOutCl + 1) / 2;
    std::vector<std::pair<MCCompLabel, unsigned int>> labels;
    nOutCl = 0;
    for (int j = 0; j < tracks[i].NClusters(); j++) {
      if (trackClusters[tracks[i].FirstClusterRef() + j].state & GPUTPCGMMergedTrackHit::flagReject) {
        continue;
      }
      int clusterIdGlobal = trackClusters[tracks[i].FirstClusterRef() + j].num;
      Sector sector = trackClusters[tracks[i].FirstClusterRef() + j].slice;
      int globalRow = trackClusters[tracks[i].FirstClusterRef() + j].row;
      int clusterIdInRow = clusterIdGlobal - clusters->clusterOffset[sector][globalRow];
      int regionNumber = 0;
      while (globalRow > mapper.getGlobalRowOffsetRegion(regionNumber) + mapper.getNumberOfRowsRegion(regionNumber)) {
        regionNumber++;
      }
      clIndArr[nOutCl] = clusterIdInRow;
      sectorIndexArr[nOutCl] = sector;
      rowIndexArr[nOutCl] = globalRow;
      nOutCl++;
      if (outputTracksMCTruth && clusters->clustersMCTruth) {
        for (const auto& element : clusters->clustersMCTruth->getLabels(clusterIdGlobal)) {
          bool found = false;
          for (int l = 0; l < labels.size(); l++) {
            if (labels[l].first == element) {
              labels[l].second++;
              found = true;
              break;
            }
          }
          if (!found)
            labels.emplace_back(element, 1);
        }
      }
    }
    if (outputTracksMCTruth) {
      if (labels.size() == 0) {
        outputTracksMCTruth->addElement(iTmp, MCCompLabel()); //default constructor creates NotSet label
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
        outputTracksMCTruth->addElement(iTmp, bestLabel);
      }
    }
    int lastSector = trackClusters[tracks[i].FirstClusterRef() + tracks[i].NClusters() - 1].slice;
  }
  outClusRefs->resize(clBuff); // remove overhead
  data->compressedClusters = ptrs.tpcCompressedClusters;
  if (data->o2Digits || data->tpcZS) {
    data->clusters = ptrs.clustersNative;
  }
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
