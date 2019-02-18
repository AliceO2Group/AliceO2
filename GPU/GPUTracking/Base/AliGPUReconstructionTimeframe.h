#ifndef ALIGPURECONSTRUCTIONTIMEFRAME_H
#define ALIGPURECONSTRUCTIONTIMEFRAME_H

#include "AliGPUChainTracking.h"
#include <vector>
#include <random>
#include <tuple>

namespace o2 { namespace TPC { struct ClusterNativeAccessFullTPC; struct ClusterNative; }}

class AliGPUReconstructionTimeframe
{
public:
	AliGPUReconstructionTimeframe(AliGPUChainTracking* rec, int (*read)(int), int nEvents);
	int LoadCreateTimeFrame(int iEvent);
	int LoadMergedEvents(int iEvent);
	int ReadEventShifted(int i, float shift, float minZ = -1e6, float maxZ = -1e6, bool silent = false);
	void MergeShiftedEvents();
	
private:
	constexpr static unsigned int NSLICES = AliGPUReconstruction::NSLICES;
	
	void SetDisplayInformation(int iCol);

	AliGPUChainTracking* mChain;
	int (*ReadEvent)(int);
	int nEventsInDirectory;
	
	std::uniform_real_distribution<double> disUniReal;
	std::uniform_int_distribution<unsigned long long int> disUniInt;
	std::mt19937_64 rndGen1;
	std::mt19937_64 rndGen2;
	
	int trainDist = 0;
	float collisionProbability = 0.;
	const int orbitRate = 11245;
	const int driftTime = 93000;
	const int TPCZ = 250;
	const int timeOrbit = 1000000000 / orbitRate;
	int maxBunchesFull;
	int maxBunches;

	int nTotalCollisions = 0;

	long long int eventStride;
	int simBunchNoRepeatEvent;
	std::vector<char> eventUsed;
	std::vector<std::tuple<AliGPUChainTracking::InOutPointers, AliGPUChainTracking::InOutMemory, o2::TPC::ClusterNativeAccessFullTPC>> shiftedEvents;
};

#endif
