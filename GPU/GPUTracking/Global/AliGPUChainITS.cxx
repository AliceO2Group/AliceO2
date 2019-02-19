#include "AliGPUChainITS.h"
#include "AliGPUReconstructionIncludesITS.h"
#include "DataFormatsITS/TrackITS.h"
#include <algorithm>

using namespace o2::ITS;

AliGPUChainITS::~AliGPUChainITS()
{
	mITSTrackerTraits.reset();
	mITSVertexerTraits.reset();
}

AliGPUChainITS::AliGPUChainITS(AliGPUReconstruction* rec) : AliGPUChain(rec)
{
}

void AliGPUChainITS::RegisterPermanentMemoryAndProcessors()
{
	mRec->RegisterGPUProcessor(&workers()->itsFitter, GetRecoStepsGPU() & RecoStep::ITSTracking);
}

void AliGPUChainITS::RegisterGPUProcessors()
{
	if (GetRecoStepsGPU() & RecoStep::ITSTracking) mRec->RegisterGPUDeviceProcessor(&workersShadow()->itsFitter, &workers()->itsFitter);
}

int AliGPUChainITS::Init()
{
	mRec->GetITSTraits(mITSTrackerTraits, mITSVertexerTraits);
	mITSTrackerTraits->SetRecoChain(this, &AliGPUChainITS::RunITSTrackFit);
	return 0;
}

int AliGPUChainITS::Finalize()
{
	return 0;
}

int AliGPUChainITS::RunStandalone()
{
	return 0;
}

int AliGPUChainITS::RunITSTrackFit(std::vector<Road>& roads, std::array<const Cluster*, 7> clusters, std::array<const Cell*, 5> cells, const std::array<std::vector<TrackingFrameInfo>, 7> &tf, std::vector<TrackITS>& tracks)
{
	mRec->PrepareEvent();
	ActivateThreadContext();
	mRec->SetThreadCounts(RecoStep::ITSTracking);
	bool doGPU = GetRecoStepsGPU() & RecoStep::ITSTracking;
	GPUITSFitter& Fitter = workers()->itsFitter;
	GPUITSFitter& FitterShadow = doGPU ? workersShadow()->itsFitter : Fitter;
	
	Fitter.clearMemory();
	Fitter.SetNumberOfRoads(roads.size());
	for (int i = 0;i < 7;i++) Fitter.SetNumberTF(i, tf[i].size());
	Fitter.SetMaxData();
	std::copy(clusters.begin(), clusters.end(), Fitter.clusters());
	std::copy(cells.begin(), cells.end(), Fitter.cells());
	SetupGPUProcessor(&Fitter, true);
	std::copy(roads.begin(), roads.end(), Fitter.roads());
	for (int i = 0;i < 7;i++) std::copy(tf[i].begin(), tf[i].end(), Fitter.trackingFrame()[i]);
	
	WriteToConstantMemory((char*) &workers()->itsFitter - (char*) workers(), &FitterShadow, sizeof(FitterShadow), 0);
	TransferMemoryResourcesToGPU(&Fitter, 0);
	runKernel<GPUITSFitterKernel>({BlockCount(), ThreadCount(), 0}, nullptr, krnlRunRangeNone, krnlEventNone);
	TransferMemoryResourcesToHost(&Fitter, 0);
	
	SynchronizeGPU();
	
	for (int i = 0;i < Fitter.NumberOfTracks();i++)
	{
		auto& trkin = Fitter.tracks()[i];
		
		tracks.emplace_back(
			TrackITS{{trkin.X(), trkin.mAlpha, {trkin.Par()[0], trkin.Par()[1], trkin.Par()[2], trkin.Par()[3], trkin.Par()[4]},
			{trkin.Cov()[0], trkin.Cov()[1], trkin.Cov()[2], trkin.Cov()[3], trkin.Cov()[4], trkin.Cov()[5], trkin.Cov()[6], trkin.Cov()[7], trkin.Cov()[8],
				trkin.Cov()[9], trkin.Cov()[10], trkin.Cov()[11], trkin.Cov()[12], trkin.Cov()[13], trkin.Cov()[14]}},
			(short int) ((trkin.NDF() + 5) / 2), 0.139f, trkin.Chi2(), 0,
			{trkin.mOuterParam.fX, trkin.mOuterParam.fAlpha, {trkin.mOuterParam.fP[0], trkin.mOuterParam.fP[1], trkin.mOuterParam.fP[2], trkin.mOuterParam.fP[3], trkin.mOuterParam.fP[4]},
				{trkin.mOuterParam.fC[0], trkin.mOuterParam.fC[1], trkin.mOuterParam.fC[2], trkin.mOuterParam.fC[3], trkin.mOuterParam.fC[4], trkin.mOuterParam.fC[5], trkin.mOuterParam.fC[6], trkin.mOuterParam.fC[7], trkin.mOuterParam.fC[8],
					trkin.mOuterParam.fC[9], trkin.mOuterParam.fC[10], trkin.mOuterParam.fC[11], trkin.mOuterParam.fC[12], trkin.mOuterParam.fC[13], trkin.mOuterParam.fC[14]}},
			{{trkin.mClusters[0], trkin.mClusters[1], trkin.mClusters[2], trkin.mClusters[3], trkin.mClusters[4], trkin.mClusters[5], trkin.mClusters[6]}}}
			);
	}
	
	ReleaseThreadContext();
	return 0;
}
