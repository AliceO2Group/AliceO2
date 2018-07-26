#include "AliHLTTPCGMTracksToTPCSeeds.h"
#include "AliHLTTPCCAGlobalMergerComponent.h"
#include "AliHLTTPCGMMerger.h"
#include "AliTPCtracker.h"
#include "AliTPCtrack.h"
#include "AliTPCseed.h"
#include "AliTPCtrackerSector.h"
#include "TObjArray.h"
#include "AliTPCclusterMI.h"

void AliHLTTPCGMTracksToTPCSeeds::CreateSeedsFromHLTTracks(TObjArray* seeds, AliTPCtracker* tpctracker)
{
	const AliHLTTPCGMMerger* merger = AliHLTTPCCAGlobalMergerComponent::GetCurrentMerger();
	if (merger == NULL) return;
	seeds->Clear();
	int index = 0;
	for (int i = 0;i < merger->NOutputTracks();i++)
	{
		const AliHLTTPCGMMergedTrack &track = merger->OutputTracks()[i];
		if (!track.OK()) continue;
		
	
		AliTPCtrack tr;
		tr.Set(track.GetParam().GetX(), track.GetAlpha(), track.GetParam().GetPar(), track.GetParam().GetCov());
		AliTPCseed* seed = new(tpctracker->NextFreeSeed()) AliTPCseed(tr);
		for (int j = 0;j < HLTCA_ROW_COUNT;j++)
		{
			seed->SetClusterPointer(j, NULL);
			seed->SetClusterIndex(j, -1);
		}
		int ncls = 0;
		int lastrow = -1;
		int lastleg = -1;
		for (int j = track.NClusters() - 1;j >= 0;j--)
		{
			AliHLTTPCGMMergedTrackHit& cls = merger->Clusters()[track.FirstClusterRef() + j];
			if (cls.fState & AliHLTTPCGMMergedTrackHit::flagReject) continue;
			if (lastrow != -1 && (cls.fRow < lastrow || cls.fLeg != lastleg)) break;
			if (cls.fRow == lastrow) continue;
			
			AliTPCtrackerRow& row = tpctracker->GetRow(cls.fSlice % 18, cls.fRow);
			unsigned int clIndexOffline = 0;
			AliTPCclusterMI* clOffline = row.FindNearest2(cls.fY, cls.fZ, 0.01f, 0.01f, clIndexOffline); 
			if (!clOffline) continue;
			clIndexOffline = row.GetIndex(clIndexOffline);
			
			clOffline->Use(10);
			seed->SetClusterPointer(cls.fRow, clOffline);
			seed->SetClusterIndex2(cls.fRow, clIndexOffline);
			
			lastrow = cls.fRow;
			lastleg = cls.fLeg;
			ncls++;
		}
		
		seed->SetRelativeSector(track.GetAlpha() / (M_PI / 9.f));
		seed->SetNumberOfClusters(ncls);
		seed->SetNFoundable(ncls);
		seed->SetChi2(track.GetParam().GetChi2());

		float alpha = seed->GetAlpha();
		if (alpha >= 2 * M_PI) alpha -= 2. * M_PI;
		if (alpha < 0) alpha += 2. * M_PI;
		seed->SetRelativeSector(track.GetAlpha() / (M_PI / 9.f));

		seed->SetPoolID(tpctracker->GetLastSeedId());
		seed->SetIsSeeding(kTRUE);
		seed->SetSeed1(HLTCA_ROW_COUNT - 1);
		seed->SetSeed2(HLTCA_ROW_COUNT - 2);
		seed->SetSeedType(0);
		seed->SetFirstPoint(-1);
		seed->SetLastPoint(-1);
		seeds->AddLast(seed); // note, track is seed, don't free the seed
		index++;
	}
}

void AliHLTTPCGMTracksToTPCSeeds::UpdateParamsOuter(TObjArray* seeds)
{
	const AliHLTTPCGMMerger* merger = AliHLTTPCCAGlobalMergerComponent::GetCurrentMerger();
	if (merger == NULL) return;
	int index = 0;
	for (int i = 0;i < merger->NOutputTracks();i++)
	{
		const AliHLTTPCGMMergedTrack &track = merger->OutputTracks()[i];
		if (!track.OK()) continue;
		if (index > seeds->GetEntriesFast())
		{
			printf("Invalid number of offline seeds\n");
			return;
		}
		AliTPCseed* seed = (AliTPCseed*) seeds->UncheckedAt(index++);
		const AliHLTTPCGMTrackParam::AliHLTTPCGMTrackParam::AliHLTTPCCAOuterParam& param = track.OuterParam();
		seed->Set(param.fX, param.fAlpha, param.fP, param.fC);
	}
}

void AliHLTTPCGMTracksToTPCSeeds::UpdateParamsInner(TObjArray* seeds)
{
	const AliHLTTPCGMMerger* merger = AliHLTTPCCAGlobalMergerComponent::GetCurrentMerger();
	if (merger == NULL) return;
	int index = 0;
	for (int i = 0;i < merger->NOutputTracks();i++)
	{
		const AliHLTTPCGMMergedTrack &track = merger->OutputTracks()[i];
		if (!track.OK()) continue;
		if (index > seeds->GetEntriesFast())
		{
			printf("Invalid number of offline seeds\n");
			return;
		}
		AliTPCseed* seed = (AliTPCseed*) seeds->UncheckedAt(index++);
		seed->Set(track.GetParam().GetX(), track.GetAlpha(), track.GetParam().GetPar(), track.GetParam().GetCov());
	}
}
