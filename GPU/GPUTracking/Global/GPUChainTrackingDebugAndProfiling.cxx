// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUChainTrackingDebugAndProfiling.cxx
/// \author David Rohr

#include "GPUChainTracking.h"
#include "GPUTrackingInputProvider.h"
#include <map>
#include <memory>
#include <string>
#ifdef GPUCA_TRACKLET_CONSTRUCTOR_DO_PROFILE
#include "bitmapfile.h"
#endif
#define PROFILE_MAX_SIZE (100 * 1024 * 1024)

using namespace GPUCA_NAMESPACE::gpu;

static inline unsigned int RGB(unsigned char r, unsigned char g, unsigned char b) { return (unsigned int)r | ((unsigned int)g << 8) | ((unsigned int)b << 16); }

int GPUChainTracking::PrepareProfile()
{
#ifdef GPUCA_TRACKLET_CONSTRUCTOR_DO_PROFILE
  char* tmpMem = (char*)mRec->AllocateUnmanagedMemory(PROFILE_MAX_SIZE, GPUMemoryResource::MEMORY_GPU);
  processorsShadow()->tpcTrackers[0].mStageAtSync = tmpMem;
  runKernel<GPUMemClean16>({BlockCount(), ThreadCount(), -1}, nullptr, krnlRunRangeNone, krnlEventNone, tmpMem, PROFILE_MAX_SIZE);
#endif
  return 0;
}

int GPUChainTracking::DoProfile()
{
#ifdef GPUCA_TRACKLET_CONSTRUCTOR_DO_PROFILE
  std::unique_ptr<char[]> stageAtSync{new char[PROFILE_MAX_SIZE]};
  mRec->GPUMemCpy(stageAtSync.get(), processorsShadow()->tpcTrackers[0].mStageAtSync, PROFILE_MAX_SIZE, -1, false);

  FILE* fp = fopen("profile.txt", "w+");
  FILE* fp2 = fopen("profile.bmp", "w+b");

  const int bmpheight = 8192;
  BITMAPFILEHEADER bmpFH;
  BITMAPINFOHEADER bmpIH;
  memset(&bmpFH, 0, sizeof(bmpFH));
  memset(&bmpIH, 0, sizeof(bmpIH));

  bmpFH.bfType = 19778; //"BM"
  bmpFH.bfSize = sizeof(bmpFH) + sizeof(bmpIH) + (ConstructorBlockCount() * ConstructorThreadCount() / 32 * 33 - 1) * bmpheight;
  bmpFH.bfOffBits = sizeof(bmpFH) + sizeof(bmpIH);

  bmpIH.biSize = sizeof(bmpIH);
  bmpIH.biWidth = ConstructorBlockCount() * ConstructorThreadCount() / 32 * 33 - 1;
  bmpIH.biHeight = bmpheight;
  bmpIH.biPlanes = 1;
  bmpIH.biBitCount = 32;

  fwrite(&bmpFH, 1, sizeof(bmpFH), fp2);
  fwrite(&bmpIH, 1, sizeof(bmpIH), fp2);

  int nEmptySync = 0;
  for (unsigned int i = 0; i < bmpheight * ConstructorBlockCount() * ConstructorThreadCount(); i += ConstructorBlockCount() * ConstructorThreadCount()) {
    int fEmpty = 1;
    for (unsigned int j = 0; j < ConstructorBlockCount() * ConstructorThreadCount(); j++) {
      fprintf(fp, "%d\t", stageAtSync[i + j]);
      int color = 0;
      if (stageAtSync[i + j] == 1) {
        color = RGB(255, 0, 0);
      }
      if (stageAtSync[i + j] == 2) {
        color = RGB(0, 255, 0);
      }
      if (stageAtSync[i + j] == 3) {
        color = RGB(0, 0, 255);
      }
      if (stageAtSync[i + j] == 4) {
        color = RGB(255, 255, 0);
      }
      fwrite(&color, 1, sizeof(int), fp2);
      if (j > 0 && j % 32 == 0) {
        color = RGB(255, 255, 255);
        fwrite(&color, 1, 4, fp2);
      }
      if (stageAtSync[i + j]) {
        fEmpty = 0;
      }
    }
    fprintf(fp, "\n");
    if (fEmpty) {
      nEmptySync++;
    } else {
      nEmptySync = 0;
    }
    (void)nEmptySync;
    // if (nEmptySync == GPUCA_SCHED_ROW_STEP + 2) break;
  }

  fclose(fp);
  fclose(fp2);
#endif
  return 0;
}

namespace
{
struct GPUChainTrackingMemUsage {
  void add(unsigned long n, unsigned long bound)
  {
    nMax = std::max(nMax, n);
    maxUse = std::max(n / std::max<double>(bound, 1.), maxUse);
    nSum += n;
    nBoundSum += bound;
    count++;
  }
  unsigned long nMax;
  unsigned long nSum = 0;
  unsigned long nBoundSum = 0;
  double maxUse = 0.;
  unsigned int count = 0;
};

void addToMap(std::string name, std::map<std::string, GPUChainTrackingMemUsage>& map, unsigned long n, unsigned long bound)
{
  GPUChainTrackingMemUsage& obj = map.insert({name, {}}).first->second;
  obj.add(n, bound);
}
} // namespace

void GPUChainTracking::PrintMemoryStatistics()
{
  std::map<std::string, GPUChainTrackingMemUsage> usageMap;
  for (int i = 0; i < NSLICES; i++) {
#ifdef GPUCA_TPC_GEOMETRY_O2
    addToMap("TPC Clusterer Sector Peaks", usageMap, processors()->tpcClusterer[i].mPmemory->counters.nPeaks, processors()->tpcClusterer[i].mNMaxPeaks);
    addToMap("TPC Clusterer Sector Clusters", usageMap, processors()->tpcClusterer[i].mPmemory->counters.nClusters, processors()->tpcClusterer[i].mNMaxClusters);
#endif
    addToMap("TPC Start Hits", usageMap, *processors()->tpcTrackers[i].NStartHits(), processors()->tpcTrackers[i].NMaxStartHits());
    addToMap("TPC Tracklets", usageMap, *processors()->tpcTrackers[i].NTracklets(), processors()->tpcTrackers[i].NMaxTracklets());
    addToMap("TPC TrackletHits", usageMap, *processors()->tpcTrackers[i].NRowHits(), processors()->tpcTrackers[i].NMaxRowHits());
    addToMap("TPC Sector Tracks", usageMap, *processors()->tpcTrackers[i].NTracks(), processors()->tpcTrackers[i].NMaxTracks());
    addToMap("TPC Sector TrackHits", usageMap, *processors()->tpcTrackers[i].NTrackHits(), processors()->tpcTrackers[i].NMaxTrackHits());
  }
  addToMap("TPC Clusters", usageMap, mIOPtrs.clustersNative->nClustersTotal, mInputsHost->mNClusterNative);
  addToMap("TPC Tracks", usageMap, processors()->tpcMerger.NOutputTracks(), processors()->tpcMerger.NMaxTracks());
  addToMap("TPC TrackHits", usageMap, processors()->tpcMerger.NOutputTrackClusters(), processors()->tpcMerger.NMaxOutputTrackClusters());

  for (auto& elem : usageMap) {
    GPUInfo("Mem Usage %-30s : %9lu / %9lu (%3.0f%% / %3.0f%% / count %3u / max %9lu)", elem.first.c_str(), elem.second.nSum, elem.second.nBoundSum, 100. * elem.second.nSum / std::max(1lu, elem.second.nBoundSum), 100. * elem.second.maxUse, elem.second.count, elem.second.nMax);
  }
}

void GPUChainTracking::PrintMemoryRelations()
{
  for (int i = 0; i < NSLICES; i++) {
    GPUInfo("MEMREL StartHits NCl %d NTrkl %d", processors()->tpcTrackers[i].NHitsTotal(), *processors()->tpcTrackers[i].NStartHits());
    GPUInfo("MEMREL Tracklets NCl %d NTrkl %d", processors()->tpcTrackers[i].NHitsTotal(), *processors()->tpcTrackers[i].NTracklets());
    GPUInfo("MEMREL Tracklets NCl %d NTrkl %d", processors()->tpcTrackers[i].NHitsTotal(), *processors()->tpcTrackers[i].NRowHits());
    GPUInfo("MEMREL SectorTracks NCl %d NTrk %d", processors()->tpcTrackers[i].NHitsTotal(), *processors()->tpcTrackers[i].NTracks());
    GPUInfo("MEMREL SectorTrackHits NCl %d NTrkH %d", processors()->tpcTrackers[i].NHitsTotal(), *processors()->tpcTrackers[i].NTrackHits());
  }
  GPUInfo("MEMREL Tracks NCl %d NTrk %d", processors()->tpcMerger.NMaxClusters(), processors()->tpcMerger.NOutputTracks());
  GPUInfo("MEMREL TrackHitss NCl %d NTrkH %d", processors()->tpcMerger.NMaxClusters(), processors()->tpcMerger.NOutputTrackClusters());
}

void GPUChainTracking::PrepareDebugOutput()
{
#ifdef GPUCA_KERNEL_DEBUGGER_OUTPUT
  const auto& threadContext = GetThreadContext();
  if (mRec->IsGPU()) {
    SetupGPUProcessor(&processors()->debugOutput, false);
    WriteToConstantMemory(RecoStep::NoRecoStep, (char*)&processors()->debugOutput - (char*)processors(), &processorsShadow()->debugOutput, sizeof(processors()->debugOutput), -1);
    memset(processors()->debugOutput.memory(), 0, processors()->debugOutput.memorySize() * sizeof(processors()->debugOutput.memory()[0]));
  }
  runKernel<GPUMemClean16>({BlockCount(), ThreadCount(), 0, RecoStep::TPCSliceTracking}, krnlRunRangeNone, {}, (mRec->IsGPU() ? processorsShadow() : processors())->debugOutput.memory(), processorsShadow()->debugOutput.memorySize() * sizeof(processors()->debugOutput.memory()[0]));
#endif
}

void GPUChainTracking::PrintDebugOutput()
{
#ifdef GPUCA_KERNEL_DEBUGGER_OUTPUT
  const auto& threadContext = GetThreadContext();
  TransferMemoryResourcesToHost(RecoStep::NoRecoStep, &processors()->debugOutput, -1);
  processors()->debugOutput.Print();
#endif
}

void GPUChainTracking::PrintOutputStat()
{
  int nTracks = 0, nAttachedClusters = 0, nAttachedClustersFitted = 0, nAdjacentClusters = 0;
  unsigned int nCls = GetProcessingSettings().doublePipeline ? mIOPtrs.clustersNative->nClustersTotal : GetTPCMerger().NMaxClusters();
  if (ProcessingSettings().createO2Output > 1) {
    nTracks = mIOPtrs.nOutputTracksTPCO2;
    nAttachedClusters = mIOPtrs.nMergedTrackHits;
  } else {
    for (unsigned int k = 0; k < mIOPtrs.nMergedTracks; k++) {
      if (mIOPtrs.mergedTracks[k].OK()) {
        nTracks++;
        nAttachedClusters += mIOPtrs.mergedTracks[k].NClusters();
        nAttachedClustersFitted += mIOPtrs.mergedTracks[k].NClustersFitted();
      }
    }
    for (unsigned int k = 0; k < nCls; k++) {
      int attach = mIOPtrs.mergedTrackHitAttachment[k];
      if (attach & gputpcgmmergertypes::attachFlagMask) {
        nAdjacentClusters++;
      }
    }
  }

  char trdText[1024] = "";
  if (GetRecoSteps() & GPUDataTypes::RecoStep::TRDTracking) {
    int nTRDTracks = 0;
    int nTRDTracklets = 0;
    for (unsigned int k = 0; k < mIOPtrs.nTRDTracks; k++) {
      auto& trk = mIOPtrs.trdTracks[k];
      nTRDTracklets += trk.GetNtracklets();
      nTRDTracks += trk.GetNtracklets() != 0;
    }
    snprintf(trdText, 1024, " - TRD Tracker reconstructed %d tracks (%d tracklets)", nTRDTracks, nTRDTracklets);
  }
  printf("Output Tracks: %d (%d / %d / %d / %d clusters (fitted / attached / adjacent / total))%s\n", nTracks, nAttachedClustersFitted, nAttachedClusters, nAdjacentClusters, nCls, trdText);
}
