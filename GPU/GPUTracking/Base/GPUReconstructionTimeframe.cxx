// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionTimeframe.cxx
/// \author David Rohr

#include "GPUReconstructionTimeframe.h"
#include "GPUReconstruction.h"
#include "GPUDisplay.h"
#include "GPUQA.h"
#include "AliHLTTPCClusterMCData.h"
#include "GPUTPCMCInfo.h"
#include "GPUTPCClusterData.h"
#include "AliHLTTPCRawCluster.h"
#include "TPCFastTransform.h"
#include "GPUO2DataTypes.h"

#include <cstdio>
#include <exception>
#include <memory>
#include <cstring>

#include "utils/qconfig.h"

using namespace GPUCA_NAMESPACE::gpu;

static auto& config = configStandalone.configTF;

GPUReconstructionTimeframe::GPUReconstructionTimeframe(GPUChainTracking* chain, int (*read)(int), int nEvents) : mChain(chain), mReadEvent(read), mNEventsInDirectory(nEvents), mDisUniReal(0., 1.), mRndGen1(configStandalone.seed), mRndGen2(mDisUniInt(mRndGen1))
{
  mMaxBunchesFull = TIME_ORBIT / config.bunchSpacing;
  mMaxBunches = (TIME_ORBIT - config.abortGapTime) / config.bunchSpacing;

  if (config.overlayRaw && chain->GetTPCTransform() == nullptr) {
    GPUInfo("Overlay Raw Events requires TPC Fast Transform");
    throw std::exception();
  }
  if (config.bunchSim) {
    if (config.bunchCount * config.bunchTrainCount > mMaxBunches) {
      GPUInfo("Invalid timeframe settings: too many colliding bunches requested!");
      throw std::exception();
    }
    mTrainDist = mMaxBunches / config.bunchTrainCount;
    mCollisionProbability = (float)config.interactionRate * (float)(mMaxBunchesFull * config.bunchSpacing / 1e9f) / (float)(config.bunchCount * config.bunchTrainCount);
    GPUInfo("Timeframe settings: %d trains of %d bunches, bunch spacing: %d, train spacing: %dx%d, filled bunches %d / %d (%d), collision probability %f, mixing %d events", config.bunchTrainCount, config.bunchCount, config.bunchSpacing, mTrainDist, config.bunchSpacing,
            config.bunchCount * config.bunchTrainCount, mMaxBunches, mMaxBunchesFull, mCollisionProbability, mNEventsInDirectory);
  }

  mEventStride = configStandalone.seed;
  mSimBunchNoRepeatEvent = configStandalone.StartEvent;
  mEventUsed.resize(mNEventsInDirectory);
  if (config.noEventRepeat == 2) {
    memset(mEventUsed.data(), 0, mNEventsInDirectory * sizeof(mEventUsed[0]));
  }
}

int GPUReconstructionTimeframe::ReadEventShifted(int iEvent, float shiftZ, float minZ, float maxZ, bool silent)
{
  mReadEvent(iEvent);
  if (config.overlayRaw) {
    float shiftTTotal = (((double)config.timeFrameLen - DRIFT_TIME) * ((double)TPCZ / (double)DRIFT_TIME) - shiftZ) / mChain->GetTPCTransform()->getVDrift();
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      for (unsigned int j = 0; j < mChain->mIOPtrs.nRawClusters[iSlice]; j++) {
        auto& tmp = mChain->mIOMem.rawClusters[iSlice][j];
        tmp.fTime += shiftTTotal;
      }
    }
  }
  if (shiftZ != 0.) {
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      for (unsigned int j = 0; j < mChain->mIOPtrs.nClusterData[iSlice]; j++) {
        auto& tmp = mChain->mIOMem.clusterData[iSlice][j];
        tmp.z += iSlice < NSLICES / 2 ? shiftZ : -shiftZ;
      }
    }
    for (unsigned int i = 0; i < mChain->mIOPtrs.nMCInfosTPC; i++) {
      auto& tmp = mChain->mIOMem.mcInfosTPC[i];
      tmp.z += i < NSLICES / 2 ? shiftZ : -shiftZ;
    }
  }

  // Remove clusters outside boundaries
  unsigned int nClusters = 0;
  unsigned int removed = 0;
  if (minZ > -1e6 || maxZ > -1e6) {
    unsigned int currentClusterTotal = 0;
    for (unsigned int iSlice = 0; iSlice < NSLICES; iSlice++) {
      unsigned int currentClusterSlice = 0;
      bool doRaw = config.overlayRaw && mChain->mIOPtrs.nClusterData[iSlice] == mChain->mIOPtrs.nRawClusters[iSlice];
      for (unsigned int i = 0; i < mChain->mIOPtrs.nClusterData[iSlice]; i++) {
        float sign = iSlice < NSLICES / 2 ? 1 : -1;
        if (sign * mChain->mIOMem.clusterData[iSlice][i].z >= minZ && sign * mChain->mIOMem.clusterData[iSlice][i].z <= maxZ) {
          if (currentClusterSlice != i) {
            mChain->mIOMem.clusterData[iSlice][currentClusterSlice] = mChain->mIOMem.clusterData[iSlice][i];
            if (doRaw) {
              mChain->mIOMem.rawClusters[iSlice][currentClusterSlice] = mChain->mIOMem.rawClusters[iSlice][i];
            }
          }
          if (mChain->mIOPtrs.nMCLabelsTPC > currentClusterTotal && nClusters != currentClusterTotal) {
            mChain->mIOMem.mcLabelsTPC[nClusters] = mChain->mIOMem.mcLabelsTPC[currentClusterTotal];
          }
          // GPUInfo("Keeping Cluster ID %d (ID in slice %d) Z=%f (sector %d) --> %d (slice %d)", currentClusterTotal, i, mChain->mIOMem.clusterData[iSlice][i].fZ, iSlice, nClusters, currentClusterSlice);
          currentClusterSlice++;
          nClusters++;
        } else {
          // GPUInfo("Removing Cluster ID %d (ID in slice %d) Z=%f (sector %d)", currentClusterTotal, i, mChain->mIOMem.clusterData[iSlice][i].fZ, iSlice);
          removed++;
        }
        currentClusterTotal++;
      }
      mChain->mIOPtrs.nClusterData[iSlice] = currentClusterSlice;
      if (doRaw) {
        mChain->mIOPtrs.nRawClusters[iSlice] = currentClusterSlice;
      }
    }
    if (mChain->mIOPtrs.nMCLabelsTPC) {
      mChain->mIOPtrs.nMCLabelsTPC = nClusters;
    }
  } else {
    for (unsigned int i = 0; i < NSLICES; i++) {
      nClusters += mChain->mIOPtrs.nClusterData[i];
    }
  }

  if (!silent) {
    GPUInfo("Read %u Clusters with %d MC labels and %d MC tracks", nClusters, (int)mChain->mIOPtrs.nMCLabelsTPC, (int)mChain->mIOPtrs.nMCInfosTPC);
    if (minZ > -1e6 || maxZ > 1e6) {
      GPUInfo("\tRemoved %u / %u clusters", removed, nClusters + removed);
    }
  }

  mShiftedEvents.emplace_back(mChain->mIOPtrs, std::move(mChain->mIOMem), mChain->mIOPtrs.clustersNative ? *mChain->mIOPtrs.clustersNative : o2::tpc::ClusterNativeAccess());
  return nClusters;
}

void GPUReconstructionTimeframe::MergeShiftedEvents()
{
  mChain->ClearIOPointers();
  for (unsigned int i = 0; i < mShiftedEvents.size(); i++) {
    auto& ptr = std::get<0>(mShiftedEvents[i]);
    for (unsigned int j = 0; j < NSLICES; j++) {
      mChain->mIOPtrs.nClusterData[j] += ptr.nClusterData[j];
      if (config.overlayRaw) {
        mChain->mIOPtrs.nRawClusters[j] += ptr.nRawClusters[j];
      }
    }
    mChain->mIOPtrs.nMCLabelsTPC += ptr.nMCLabelsTPC;
    mChain->mIOPtrs.nMCInfosTPC += ptr.nMCInfosTPC;
    SetDisplayInformation(i);
  }
  unsigned int nClustersTotal = 0;
  unsigned int nClustersTotalRaw = 0;
  unsigned int nClustersSliceOffset[NSLICES] = { 0 };
  for (unsigned int i = 0; i < NSLICES; i++) {
    nClustersSliceOffset[i] = nClustersTotal;
    nClustersTotal += mChain->mIOPtrs.nClusterData[i];
    nClustersTotalRaw += mChain->mIOPtrs.nRawClusters[i];
  }
  if (nClustersTotalRaw && nClustersTotalRaw != nClustersTotal) {
    GPUError("Inconsitency between raw clusters and cluster data");
    throw std::exception();
  }
  if (mChain->mIOPtrs.nMCLabelsTPC && nClustersTotal != mChain->mIOPtrs.nMCLabelsTPC) {
    GPUError("Inconsitency between TPC clusters and MC labels");
    throw std::exception();
  }
  mChain->AllocateIOMemory();
  mChain->mIOPtrs.clustersNative = nullptr;

  unsigned int nTrackOffset = 0;
  unsigned int nClustersEventOffset[NSLICES] = { 0 };
  for (unsigned int i = 0; i < mShiftedEvents.size(); i++) {
    auto& ptr = std::get<0>(mShiftedEvents[i]);
    unsigned int inEventOffset = 0;
    for (unsigned int j = 0; j < NSLICES; j++) {
      memcpy((void*)&mChain->mIOMem.clusterData[j][nClustersEventOffset[j]], (void*)ptr.clusterData[j], ptr.nClusterData[j] * sizeof(ptr.clusterData[j][0]));
      if (nClustersTotalRaw) {
        memcpy((void*)&mChain->mIOMem.rawClusters[j][nClustersEventOffset[j]], (void*)ptr.rawClusters[j], ptr.nRawClusters[j] * sizeof(ptr.rawClusters[j][0]));
      }
      if (mChain->mIOPtrs.nMCLabelsTPC) {
        memcpy((void*)&mChain->mIOMem.mcLabelsTPC[nClustersSliceOffset[j] + nClustersEventOffset[j]], (void*)&ptr.mcLabelsTPC[inEventOffset], ptr.nClusterData[j] * sizeof(ptr.mcLabelsTPC[0]));
      }
      for (unsigned int k = 0; k < ptr.nClusterData[j]; k++) {
        mChain->mIOMem.clusterData[j][nClustersEventOffset[j] + k].id = nClustersSliceOffset[j] + nClustersEventOffset[j] + k;
        if (mChain->mIOPtrs.nMCLabelsTPC) {
          for (int l = 0; l < 3; l++) {
            auto& label = mChain->mIOMem.mcLabelsTPC[nClustersSliceOffset[j] + nClustersEventOffset[j] + k].fClusterID[l];
            if (label.fMCID >= 0) {
              label.fMCID += nTrackOffset;
            }
          }
        }
      }

      nClustersEventOffset[j] += ptr.nClusterData[j];
      inEventOffset += ptr.nClusterData[j];
    }

    memcpy((void*)&mChain->mIOMem.mcInfosTPC[nTrackOffset], (void*)ptr.mcInfosTPC, ptr.nMCInfosTPC * sizeof(ptr.mcInfosTPC[0]));
    nTrackOffset += ptr.nMCInfosTPC;
  }

  GPUInfo("Merged %d events, %u clusters total", (int)mShiftedEvents.size(), nClustersTotal);

  mShiftedEvents.clear();
}

int GPUReconstructionTimeframe::LoadCreateTimeFrame(int iEvent)
{
  if (config.nTotalInTFEvents && mNTotalCollisions >= config.nTotalInTFEvents) {
    return (2);
  }

  long long int nBunch = -DRIFT_TIME / config.bunchSpacing;
  long long int lastBunch = config.timeFrameLen / config.bunchSpacing;
  long long int lastTFBunch = lastBunch - DRIFT_TIME / config.bunchSpacing;
  int nCollisions = 0, nBorderCollisions = 0, nTrainCollissions = 0, nMultipleCollisions = 0, nTrainMultipleCollisions = 0;
  int nTrain = 0;
  int mcMin = -1, mcMax = -1;
  unsigned int nTotalClusters = 0;
  while (nBunch < lastBunch) {
    for (int iTrain = 0; iTrain < config.bunchTrainCount && nBunch < lastBunch; iTrain++) {
      int nCollisionsInTrain = 0;
      for (int iBunch = 0; iBunch < config.bunchCount && nBunch < lastBunch; iBunch++) {
        const bool inTF = nBunch >= 0 && nBunch < lastTFBunch && (config.nTotalInTFEvents == 0 || nCollisions < mNTotalCollisions + config.nTotalInTFEvents);
        if (mcMin == -1 && inTF) {
          mcMin = mChain->mIOPtrs.nMCInfosTPC;
        }
        if (mcMax == -1 && nBunch >= 0 && !inTF) {
          mcMax = mChain->mIOPtrs.nMCInfosTPC;
        }
        int nInBunchPileUp = 0;
        double randVal = mDisUniReal(inTF ? mRndGen2 : mRndGen1);
        double p = exp(-mCollisionProbability);
        double p2 = p;
        while (randVal > p) {
          if (config.noBorder && (nBunch < 0 || nBunch >= lastTFBunch)) {
            break;
          }
          if (nCollisionsInTrain >= mNEventsInDirectory) {
            GPUError("Error: insuffient events for mixing!");
            return (1);
          }
          if (nCollisionsInTrain == 0 && config.noEventRepeat == 0) {
            memset(mEventUsed.data(), 0, mNEventsInDirectory * sizeof(mEventUsed[0]));
          }
          if (inTF) {
            nCollisions++;
          } else {
            nBorderCollisions++;
          }
          int useEvent;
          if (config.noEventRepeat == 1) {
            useEvent = mSimBunchNoRepeatEvent;
          } else {
            while (mEventUsed[useEvent = (inTF && config.eventStride ? (mEventStride += config.eventStride) : mDisUniInt(inTF ? mRndGen2 : mRndGen1)) % mNEventsInDirectory]) {
              ;
            }
          }
          if (config.noEventRepeat) {
            mSimBunchNoRepeatEvent++;
          }
          mEventUsed[useEvent] = 1;
          double shift = (double)nBunch * (double)config.bunchSpacing * (double)TPCZ / (double)DRIFT_TIME;
          int nClusters = ReadEventShifted(useEvent, shift, 0, (double)config.timeFrameLen * (double)TPCZ / (double)DRIFT_TIME, true);
          if (nClusters < 0) {
            GPUInfo("Unexpected error");
            return (1);
          }
          nTotalClusters += nClusters;
          GPUInfo("Placing event %4d+%d (ID %4d) at z %7.3f (time %'dns) %s(collisions %4d, bunch %6lld, train %3d) (%'10d clusters, %'10d MC labels, %'10d track MC info)", nCollisions, nBorderCollisions, useEvent, shift, (int)(nBunch * config.bunchSpacing), inTF ? " inside" : "outside",
                  nCollisions, nBunch, nTrain, nClusters, mChain->mIOPtrs.nMCLabelsTPC, mChain->mIOPtrs.nMCInfosTPC);
          nInBunchPileUp++;
          nCollisionsInTrain++;
          p2 *= mCollisionProbability / nInBunchPileUp;
          p += p2;
          if (config.noEventRepeat && mSimBunchNoRepeatEvent >= mNEventsInDirectory) {
            nBunch = lastBunch;
          }
        }
        if (nInBunchPileUp > 1) {
          nMultipleCollisions++;
        }
        nBunch++;
      }
      nBunch += mTrainDist - config.bunchCount;
      if (nCollisionsInTrain) {
        nTrainCollissions++;
      }
      if (nCollisionsInTrain > 1) {
        nTrainMultipleCollisions++;
      }
      nTrain++;
    }
    nBunch += mMaxBunchesFull - mTrainDist * config.bunchTrainCount;
  }
  mNTotalCollisions += nCollisions;
  GPUInfo("Timeframe statistics: collisions: %d+%d in %d trains (inside / outside), average rate %f (pile up: in bunch %d, in train %d)", nCollisions, nBorderCollisions, nTrainCollissions, (float)nCollisions / (float)(config.timeFrameLen - DRIFT_TIME) * 1e9, nMultipleCollisions,
          nTrainMultipleCollisions);
  MergeShiftedEvents();
  GPUInfo("\tTotal clusters: %u, MC Labels %d, MC Infos %d", nTotalClusters, (int)mChain->mIOPtrs.nMCLabelsTPC, (int)mChain->mIOPtrs.nMCInfosTPC);

  if (!config.noBorder && mChain->GetQA()) {
    mChain->GetQA()->SetMCTrackRange(mcMin, mcMax);
  }
  return (0);
}

int GPUReconstructionTimeframe::LoadMergedEvents(int iEvent)
{
  for (int iEventInTimeframe = 0; iEventInTimeframe < config.nMerge; iEventInTimeframe++) {
    float shift;
    if (config.shiftFirstEvent || iEventInTimeframe) {
      if (config.randomizeDistance) {
        shift = mDisUniReal(mRndGen2);
        if (config.shiftFirstEvent) {
          shift = (iEventInTimeframe + shift) * config.averageDistance;
        } else {
          if (iEventInTimeframe == 0) {
            shift = 0;
          } else {
            shift = (iEventInTimeframe - 0.5 + shift) * config.averageDistance;
          }
        }
      } else {
        if (config.shiftFirstEvent) {
          shift = config.averageDistance * (iEventInTimeframe + 0.5);
        } else {
          shift = config.averageDistance * (iEventInTimeframe);
        }
      }
    } else {
      shift = 0.;
    }

    if (ReadEventShifted(iEvent * config.nMerge + iEventInTimeframe, shift) < 0) {
      return (1);
    }
  }
  MergeShiftedEvents();
  return (0);
}

void GPUReconstructionTimeframe::SetDisplayInformation(int iCol)
{
  if (mChain->GetEventDisplay()) {
    for (unsigned int sl = 0; sl < NSLICES; sl++) {
      mChain->GetEventDisplay()->SetCollisionFirstCluster(iCol, sl, mChain->mIOPtrs.nClusterData[sl]);
    }
    mChain->GetEventDisplay()->SetCollisionFirstCluster(iCol, NSLICES, mChain->mIOPtrs.nMCInfosTPC);
  }
}
