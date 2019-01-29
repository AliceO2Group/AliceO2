#include "AliGPUReconstructionTimeframe.h"
#include "AliGPUReconstruction.h"
#include "AliGPUCADisplay.h"
#include "AliGPUCAQA.h"
#include "AliHLTTPCClusterMCData.h"
#include "AliGPUTPCMCInfo.h"
#include "AliGPUTPCClusterData.h"
#include "ClusterNativeAccessExt.h"

#include <cstdio>
#include <exception>
#include <memory>
#include <cstring>

#include "cmodules/qconfig.h"

static auto& config = configStandalone.configTF;

AliGPUReconstructionTimeframe::AliGPUReconstructionTimeframe(AliGPUReconstruction* rec, int (*read)(int), int nEvents) :
	mRec(rec), ReadEvent(read), nEventsInDirectory(nEvents), disUniReal(0., 1.), rndGen1(configStandalone.seed), rndGen2(disUniInt(rndGen1))

{
	maxBunchesFull = timeOrbit / configStandalone.configTF.bunchSpacing;
	maxBunches = (timeOrbit - configStandalone.configTF.abortGapTime) / configStandalone.configTF.bunchSpacing;

	if (configStandalone.configTF.bunchSim)
	{
		if (configStandalone.configTF.bunchCount * configStandalone.configTF.bunchTrainCount > maxBunches)
		{
			printf("Invalid timeframe settings: too many colliding bunches requested!\n");
			throw std::exception();
		}
		trainDist = maxBunches / configStandalone.configTF.bunchTrainCount;
		collisionProbability = (float) configStandalone.configTF.interactionRate * (float) (maxBunchesFull * configStandalone.configTF.bunchSpacing / 1e9f) / (float) (configStandalone.configTF.bunchCount * configStandalone.configTF.bunchTrainCount);
		printf("Timeframe settings: %d trains of %d bunches, bunch spacing: %d, train spacing: %dx%d, filled bunches %d / %d (%d), collision probability %f, mixing %d events\n",
			configStandalone.configTF.bunchTrainCount, configStandalone.configTF.bunchCount, configStandalone.configTF.bunchSpacing, trainDist, configStandalone.configTF.bunchSpacing,
			configStandalone.configTF.bunchCount * configStandalone.configTF.bunchTrainCount, maxBunches, maxBunchesFull, collisionProbability, nEventsInDirectory);
	}

	eventStride = configStandalone.seed;
	simBunchNoRepeatEvent = configStandalone.StartEvent;
	eventUsed.resize(nEventsInDirectory);
	if (config.noEventRepeat == 2) memset(eventUsed.data(), 0, nEventsInDirectory * sizeof(eventUsed[0]));
}

int AliGPUReconstructionTimeframe::ReadEventShifted(int iEvent, float shift, float minZ, float maxZ, bool silent)
{
	ReadEvent(iEvent);
	
	if (shift != 0.)
	{
		for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
		{
			for (unsigned int j = 0;j < mRec->mIOPtrs.nClusterData[iSlice];j++)
			{
				auto& tmp = mRec->mIOMem.clusterData[iSlice][j];
				tmp.fZ += iSlice < NSLICES / 2 ? shift : -shift;
			}
		}
		for (unsigned int i = 0;i < mRec->mIOPtrs.nMCInfosTPC;i++)
		{
			auto& tmp = mRec->mIOMem.mcInfosTPC[i];;
			tmp.fZ += i < NSLICES / 2 ? shift : -shift;
		}
	}

	//Remove clusters outside boundaries
	unsigned int nClusters = 0;
	unsigned int removed = 0;
	if (minZ > -1e6 || maxZ > -1e6)
	{
		unsigned int currentClusterTotal = 0;
		for (unsigned int iSlice = 0;iSlice < NSLICES;iSlice++)
		{
			unsigned int currentClusterSlice = 0;
			for (unsigned int i = 0;i < mRec->mIOPtrs.nClusterData[iSlice];i++)
			{
				float sign = iSlice < NSLICES / 2 ? 1 : -1;
				if (sign * mRec->mIOMem.clusterData[iSlice][i].fZ >= minZ && sign * mRec->mIOMem.clusterData[iSlice][i].fZ <= maxZ)
				{
					if (currentClusterSlice != i) mRec->mIOMem.clusterData[iSlice][currentClusterSlice] = mRec->mIOMem.clusterData[iSlice][i];
					if (mRec->mIOPtrs.nMCLabelsTPC > currentClusterTotal && nClusters != currentClusterTotal) mRec->mIOMem.mcLabelsTPC[nClusters] = mRec->mIOMem.mcLabelsTPC[currentClusterTotal];
					//printf("Keeping Cluster ID %d (ID in slice %d) Z=%f (sector %d) --> %d (slice %d)\n", currentClusterTotal, i, mRec->mIOMem.clusterData[iSlice][i].fZ, iSlice, nClusters, currentClusterSlice);
					currentClusterSlice++;
					nClusters++;
				}
				else
				{
					//printf("Removing Cluster ID %d (ID in slice %d) Z=%f (sector %d)\n", currentClusterTotal, i, mRec->mIOMem.clusterData[iSlice][i].fZ, iSlice);
					removed++;
				}
				currentClusterTotal++;
			}
			mRec->mIOPtrs.nClusterData[iSlice] = currentClusterSlice;
		}
		mRec->mIOPtrs.nMCLabelsTPC = nClusters;
	}
	else
	{
		for (unsigned int i = 0;i < NSLICES;i++) nClusters += mRec->mIOPtrs.nClusterData[i];
	}

	if (!silent)
	{
		printf("Read %d Clusters with %d MC labels and %d MC tracks\n", nClusters, (int) mRec->mIOPtrs.nMCLabelsTPC, (int) mRec->mIOPtrs.nMCInfosTPC);
		if (minZ > -1e6 || maxZ > 1e6) printf("\tRemoved %d / %d clusters\n", removed, nClusters + removed);
	}

	shiftedEvents.emplace_back(mRec->mIOPtrs, std::move(mRec->mIOMem), mRec->mIOPtrs.clustersNative ? *mRec->mIOPtrs.clustersNative : o2::TPC::ClusterNativeAccessFullTPC());
	return nClusters;
}

void AliGPUReconstructionTimeframe::MergeShiftedEvents()
{
	mRec->ClearIOPointers();
	for (unsigned int i = 0;i < shiftedEvents.size();i++)
	{
		auto& ptr = std::get<0>(shiftedEvents[i]);
		for (unsigned int j = 0;j < NSLICES;j++)
		{
			mRec->mIOPtrs.nClusterData[j] += ptr.nClusterData[j];
		}
		mRec->mIOPtrs.nMCLabelsTPC += ptr.nMCLabelsTPC;
		mRec->mIOPtrs.nMCInfosTPC += ptr.nMCInfosTPC;
		SetDisplayInformation(i);
	}
	unsigned int nClustersTotal = 0;
	unsigned int nClustersSliceOffset[NSLICES] = {0};
	for (unsigned int i = 0;i < NSLICES;i++)
	{
		nClustersSliceOffset[i] = nClustersTotal;
		nClustersTotal += mRec->mIOPtrs.nClusterData[i];
	}
	const bool doLabels = nClustersTotal == mRec->mIOPtrs.nMCLabelsTPC;
	mRec->AllocateIOMemory();
	
	unsigned int nTrackOffset = 0;
	unsigned int nClustersEventOffset[NSLICES] = {0};
	for (unsigned int i = 0;i < shiftedEvents.size();i++)
	{
		auto& ptr = std::get<0>(shiftedEvents[i]);
		unsigned int inEventOffset = 0;
		for (unsigned int j = 0;j < NSLICES;j++)
		{
			memcpy((void*) &mRec->mIOMem.clusterData[j][nClustersEventOffset[j]], (void*) ptr.clusterData[j], ptr.nClusterData[j] * sizeof(ptr.clusterData[j][0]));
			if (doLabels)
			{
				memcpy((void*) &mRec->mIOMem.mcLabelsTPC[nClustersSliceOffset[j] + nClustersEventOffset[j]], (void*) &ptr.mcLabelsTPC[inEventOffset], ptr.nClusterData[j] * sizeof(ptr.mcLabelsTPC[0]));
			}
			for (unsigned int k = 0;k < ptr.nClusterData[j];k++)
			{
				mRec->mIOMem.clusterData[j][nClustersEventOffset[j] + k].fId = nClustersSliceOffset[j] + nClustersEventOffset[j] + k;
				if (doLabels)
				{
					for (int l = 0;l < 3;l++)
					{
						auto& label = mRec->mIOMem.mcLabelsTPC[nClustersSliceOffset[j] + nClustersEventOffset[j] + k].fClusterID[l];
						if (label.fMCID >= 0) label.fMCID += nTrackOffset;
					}
				}
			}
			
			nClustersEventOffset[j] += ptr.nClusterData[j];
			inEventOffset += ptr.nClusterData[j];
		}
		
		memcpy((void*) &mRec->mIOMem.mcInfosTPC[nTrackOffset], (void*) ptr.mcInfosTPC, ptr.nMCInfosTPC * sizeof(ptr.mcInfosTPC[0]));
		nTrackOffset += ptr.nMCInfosTPC;
	}
	
	shiftedEvents.clear();
}

int AliGPUReconstructionTimeframe::LoadCreateTimeFrame(int iEvent)
{
	if (configStandalone.configTF.nTotalInTFEvents && nTotalCollisions >= configStandalone.configTF.nTotalInTFEvents) return(2);

	long long int nBunch = -driftTime / config.bunchSpacing;
	long long int lastBunch = config.timeFrameLen / config.bunchSpacing;
	long long int lastTFBunch = lastBunch - driftTime / config.bunchSpacing;
	int nCollisions = 0, nBorderCollisions = 0, nTrainCollissions = 0, nMultipleCollisions = 0, nTrainMultipleCollisions = 0;
	int nTrain = 0;
	int mcMin = -1, mcMax = -1;
	unsigned int nTotalClusters = 0;
	while (nBunch < lastBunch)
	{
		for (int iTrain = 0;iTrain < config.bunchTrainCount && nBunch < lastBunch;iTrain++)
		{
			int nCollisionsInTrain = 0;
			for (int iBunch = 0;iBunch < config.bunchCount && nBunch < lastBunch;iBunch++)
			{
				const bool inTF = nBunch >= 0 && nBunch < lastTFBunch && (config.nTotalInTFEvents == 0 || nCollisions < nTotalCollisions + config.nTotalInTFEvents);
				if (mcMin == -1 && inTF) mcMin = mRec->mIOPtrs.nMCInfosTPC;
				if (mcMax == -1 && nBunch >= 0 && !inTF) mcMax = mRec->mIOPtrs.nMCInfosTPC;
				int nInBunchPileUp = 0;
				double randVal = disUniReal(inTF ? rndGen2 : rndGen1);
				double p = exp(-collisionProbability);
				double p2 = p;
				while (randVal > p)
				{
					if (config.noBorder && (nBunch < 0 || nBunch >= lastTFBunch)) break;
					if (nCollisionsInTrain >= nEventsInDirectory)
					{
						printf("Error: insuffient events for mixing!\n");
						return(1);
					}
					if (nCollisionsInTrain == 0 && config.noEventRepeat == 0) memset(eventUsed.data(), 0, nEventsInDirectory * sizeof(eventUsed[0]));
					if (inTF) nCollisions++;
					else nBorderCollisions++;
					int useEvent;
					if (config.noEventRepeat == 1) useEvent = simBunchNoRepeatEvent;
					else while (eventUsed[useEvent = (inTF && config.eventStride ? (eventStride += config.eventStride) : disUniInt(inTF ? rndGen2 : rndGen1)) % nEventsInDirectory]);
					if (config.noEventRepeat) simBunchNoRepeatEvent++;
					eventUsed[useEvent] = 1;
					double shift = (double) nBunch * (double) config.bunchSpacing * (double) TPCZ / (double) driftTime;
					int nClusters = ReadEventShifted(useEvent, shift, 0, (double) config.timeFrameLen * TPCZ / driftTime, true);
					if (nClusters < 0)
					{
						printf("Unexpected error\n");
						return(1);
					}
					nTotalClusters += nClusters;
					printf("Placing event %4d+%d (ID %4d) at z %7.3f (time %'dns) %s(collisions %4d, bunch %6lld, train %3d) (%'10d clusters, %'10d MC labels, %'10d track MC info)\n",
						nCollisions, nBorderCollisions, useEvent, shift, (int) (nBunch * config.bunchSpacing), inTF ? " inside" : "outside", nCollisions, nBunch, nTrain, nClusters, mRec->mIOPtrs.nMCLabelsTPC, mRec->mIOPtrs.nMCInfosTPC);
					nInBunchPileUp++;
					nCollisionsInTrain++;
					p2 *= collisionProbability / nInBunchPileUp;
					p += p2;
					if (config.noEventRepeat && simBunchNoRepeatEvent >= nEventsInDirectory) nBunch = lastBunch;
				}
				if (nInBunchPileUp > 1) nMultipleCollisions++;
				nBunch++;
			}
			nBunch += trainDist - config.bunchCount;
			if (nCollisionsInTrain) nTrainCollissions++;
			if (nCollisionsInTrain > 1) nTrainMultipleCollisions++;
			nTrain++;
		}
		nBunch += maxBunchesFull - trainDist * config.bunchTrainCount;
	}
	nTotalCollisions += nCollisions;
	printf("Timeframe statistics: collisions: %d+%d in %d trains (inside / outside), average rate %f (pile up: in bunch %d, in train %d)\n",
		nCollisions, nBorderCollisions, nTrainCollissions, (float) nCollisions / (float) (config.timeFrameLen - driftTime) * 1e9, nMultipleCollisions, nTrainMultipleCollisions);
	MergeShiftedEvents();
	printf("\tTotal clusters: %d, MC Labels %d, MC Infos %d\n", nTotalClusters, mRec->mIOPtrs.nMCLabelsTPC, mRec->mIOPtrs.nMCInfosTPC);

	if (!config.noBorder) mRec->GetQA()->SetMCTrackRange(mcMin, mcMax);
	return(0);
}

int AliGPUReconstructionTimeframe::LoadMergedEvents(int iEvent)
{
	for (int iEventInTimeframe = 0;iEventInTimeframe < config.nMerge;iEventInTimeframe++)
	{
		float shift;
		if (config.shiftFirstEvent || iEventInTimeframe)
		{
			if (config.randomizeDistance)
			{
				shift = disUniReal(rndGen2);
				if (config.shiftFirstEvent)
				{
					if (iEventInTimeframe == 0) shift = shift * config.averageDistance;
					else shift = (iEventInTimeframe + shift) * config.averageDistance;
				}
				else
				{
					if (iEventInTimeframe == 0) shift = 0;
					else shift = (iEventInTimeframe - 0.5 + shift) * config.averageDistance;
				}
			}
			else
			{
				if (config.shiftFirstEvent)
				{
					shift = config.averageDistance * (iEventInTimeframe + 0.5);
				}
				else
				{
					shift = config.averageDistance * (iEventInTimeframe);
				}
			}
		}
		else
		{
			shift = 0.;
		}

		if (ReadEventShifted(iEvent * config.nMerge + iEventInTimeframe, shift) < 0) return(1);
	}
	MergeShiftedEvents();
	return(0);
}

void AliGPUReconstructionTimeframe::SetDisplayInformation(int iCol)
{
	if (mRec->GetEventDisplay())
	{
		for (unsigned int sl = 0;sl < NSLICES;sl++) mRec->GetEventDisplay()->SetCollisionFirstCluster(iCol, sl, mRec->mIOPtrs.nClusterData[sl]);
		mRec->GetEventDisplay()->SetCollisionFirstCluster(iCol, NSLICES, mRec->mIOPtrs.nMCInfosTPC);
	}
}
