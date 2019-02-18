#include "AliGPUTPCTracker.h"
#include "AliGPUReconstruction.h"
#include "AliGPUTPCHitId.h"
#include "AliGPUTPCTrack.h"

#include <iostream>
#include <string.h>
#include <iomanip>

void AliGPUTPCTracker::DumpOutput(FILE* out)
{
	fprintf(out, "Slice %d\n", fISlice);
	const AliGPUTPCSliceOutTrack* track = (*(Output()))->GetFirstTrack();
	for (unsigned int j = 0;j < (*(Output()))->NTracks();j++)
	{
		fprintf(out, "Track %d (%d): ", j, track->NClusters());
		for (int k = 0;k < track->NClusters();k++)
		{
			fprintf(out, "(%2.3f,%2.3f,%2.4f) ", track->Cluster(k).GetX(), track->Cluster(k).GetY(), track->Cluster(k).GetZ());
		}
		fprintf(out, " - (%8.5f %8.5f %8.5f %8.5f %8.5f)", track->Param().Y(), track->Param().Z(), track->Param().SinPhi(), track->Param().DzDs(), track->Param().QPt());
		fprintf(out, "\n");
		track = track->GetNextTrack();
	}
}

void AliGPUTPCTracker::DumpSliceData(std::ostream &out)
{
	//Dump Slice Input Data to File
	out << "Slice Data (Slice" << fISlice << "):" << std::endl;
	for (int i = 0;i < GPUCA_ROW_COUNT;i++)
	{
		if (Row(i).NHits() == 0) continue;
		out << "Row: " << i << std::endl;
		for (int j = 0;j < Row(i).NHits();j++)
		{
			if (j && j % 16 == 0) out << std::endl;
			out << j << '-' << Data().HitDataY(Row(i), j) << '-' << Data().HitDataZ(Row(i), j) << ", ";
		}
		out << std::endl;
	}
}

void AliGPUTPCTracker::DumpLinks(std::ostream &out)
{
	//Dump Links (after Neighbours Finder / Cleaner) to file
	out << "Hit Links(Slice" << fISlice << "):" << std::endl;
	for (int i = 0;i < GPUCA_ROW_COUNT;i++)
	{
		if (Row(i).NHits() == 0) continue;
		out << "Row: " << i << std::endl;
		for (int j = 0;j < Row(i).NHits();j++)
		{
			if (j && j % 32 == 0) out << std::endl;
			out << HitLinkUpData(Row(i), j) << "/" << HitLinkDownData(Row(i), j) << ", ";
		}
		out << std::endl;
	}
}

void AliGPUTPCTracker::DumpHitWeights(std::ostream &out)
{
	//dump hit weights to file
	out << "Hit Weights(Slice" << fISlice << "):" << std::endl;
	for (int i = 0;i < GPUCA_ROW_COUNT;i++)
	{
		if (Row(i).NHits() == 0) continue;
		out << "Row: " << i << ":" << std::endl;
		for (int j = 0;j < Row(i).NHits();j++)
		{
			if (j && j % 32 == 0) out << std::endl;
			out << HitWeight(Row(i), j) << ", ";
		}
		out << std::endl;
	}
}

int AliGPUTPCTracker::StarthitSortComparison(const void*a, const void* b)
{
	//qsort helper function to sort start hits
	AliGPUTPCHitId* aa = (AliGPUTPCHitId*) a;
	AliGPUTPCHitId* bb = (AliGPUTPCHitId*) b;

	if (aa->RowIndex() != bb->RowIndex()) return(aa->RowIndex() - bb->RowIndex());
	return(aa->HitIndex() - bb->HitIndex());
}

void AliGPUTPCTracker::DumpStartHits(std::ostream &out)
{
	//sort start hits and dump to file
	out << "Start Hits: (Slice" << fISlice << ") (" << *NTracklets() << ")" << std::endl;
	if (mRec->GetDeviceProcessingSettings().comparableDebutOutput) qsort(TrackletStartHits(), *NTracklets(), sizeof(AliGPUTPCHitId), StarthitSortComparison);
	for (int i = 0;i < *NTracklets();i++)
	{
		out << TrackletStartHit(i).RowIndex() << "-" << TrackletStartHit(i).HitIndex() << std::endl;
	}
	out << std::endl;
}

void AliGPUTPCTracker::DumpTrackHits(std::ostream &out)
{
	//dump tracks to file
	out << "Tracks: (Slice" << fISlice << ") (" << *NTracks() << ")" << std::endl;
	for (int k = 0;k < GPUCA_ROW_COUNT;k++)
	{
		for (int l = 0;l < Row(k).NHits();l++)
		{
			for (int j = 0;j < *NTracks();j++)
			{
				if (Tracks()[j].NHits() == 0 || !Tracks()[j].Alive()) continue;
				if (TrackHits()[Tracks()[j].FirstHitID()].RowIndex() == k && TrackHits()[Tracks()[j].FirstHitID()].HitIndex() == l)
				{
					for (int i = 0;i < Tracks()[j].NHits();i++)
					{
						out << TrackHits()[Tracks()[j].FirstHitID() + i].RowIndex() << "-" << TrackHits()[Tracks()[j].FirstHitID() + i].HitIndex() << ", ";
					}
					if (!mRec->GetDeviceProcessingSettings().comparableDebutOutput) out << "(Track: " << j << ")";
					out << std::endl;
				}
			}
		}
	}
}

void AliGPUTPCTracker::DumpTrackletHits(std::ostream &out)
{
	//dump tracklets to file
	int nTracklets = *NTracklets();
	if( nTracklets<0 ) nTracklets = 0;
	if( nTracklets>GPUCA_MAX_TRACKLETS ) nTracklets = GPUCA_MAX_TRACKLETS;
	out << "Tracklets: (Slice" << fISlice << ") (" << nTracklets << ")" << std::endl;
	if (mRec->GetDeviceProcessingSettings().comparableDebutOutput)
	{
		AliGPUTPCHitId* tmpIds = new AliGPUTPCHitId[nTracklets];
		AliGPUTPCTracklet* tmpTracklets = new AliGPUTPCTracklet[nTracklets];
		memcpy(tmpIds, TrackletStartHits(), nTracklets * sizeof(AliGPUTPCHitId));
		memcpy(tmpTracklets, Tracklets(), nTracklets * sizeof(AliGPUTPCTracklet));
#ifdef EXTERN_ROW_HITS
		calink* tmpHits = new calink[nTracklets * GPUCA_ROW_COUNT];
		memcpy(tmpHits, TrackletRowHits(), nTracklets * GPUCA_ROW_COUNT * sizeof(calink));
#endif
		qsort(TrackletStartHits(), nTracklets, sizeof(AliGPUTPCHitId), StarthitSortComparison);
		for (int i = 0;i < nTracklets; i++ ){
			for (int j = 0;j < nTracklets; j++ ){
				if (tmpIds[i].RowIndex() == TrackletStartHit(j).RowIndex() && tmpIds[i].HitIndex() == TrackletStartHit(j).HitIndex() ){
					memcpy(&Tracklets()[j], &tmpTracklets[i], sizeof(AliGPUTPCTracklet));
#ifdef EXTERN_ROW_HITS
					if (tmpTracklets[i].NHits() ){
						for (int k = tmpTracklets[i].FirstRow();k <= tmpTracklets[i].LastRow();k++){
							const int pos = k * nTracklets + j;
							if (pos < 0 || pos >= GPUCA_MAX_TRACKLETS * GPUCA_ROW_COUNT){
								printf("internal error: invalid tracklet position k=%d j=%d pos=%d\n", k, j, pos);
							} else {
								fTrackletRowHits[pos] = tmpHits[k * nTracklets + i];
							}
						}
					}
#endif
					break;
				}
			}
		}
		delete[] tmpIds;
		delete[] tmpTracklets;
#ifdef EXTERN_ROW_HITS
		delete[] tmpHits;
#endif
	}
	for (int j = 0;j < nTracklets; j++ )
	{
		out << "Tracklet " << std::setw(4) << j << " (Hits: " << std::setw(3) << Tracklets()[j].NHits() << ", Start: " << std::setw(3) << TrackletStartHit(j).RowIndex() << "-" << std::setw(3) << TrackletStartHit(j).HitIndex() << ", Rows: " << (Tracklets()[j].NHits() ? Tracklets()[j].FirstRow() : -1) << " - " << (Tracklets()[j].NHits() ? Tracklets()[j].LastRow() : -1) << ") ";
		if (Tracklets()[j].NHits() == 0);
		else if (Tracklets()[j].LastRow() > Tracklets()[j].FirstRow() && (Tracklets()[j].FirstRow() >= GPUCA_ROW_COUNT || Tracklets()[j].LastRow() >= GPUCA_ROW_COUNT))
		{
			printf("\nError: Tracklet %d First %d Last %d Hits %d", j, Tracklets()[j].FirstRow(), Tracklets()[j].LastRow(), Tracklets()[j].NHits());
			out << " (Error: Tracklet " << j << " First " << Tracklets()[j].FirstRow() << " Last " << Tracklets()[j].LastRow() << " Hits " << Tracklets()[j].NHits() << ") ";
			for (int i = 0;i < GPUCA_ROW_COUNT;i++)
			{
				//if (Tracklets()[j].RowHit(i) != CALINK_INVAL)
#ifdef EXTERN_ROW_HITS
				out << i << "-" << fTrackletRowHits[i * fCommonMem->fNTracklets + j] << ", ";
#else
				out << i << "-" << Tracklets()[j].RowHit(i) << ", ";
#endif
			}
		}
		else if (Tracklets()[j].NHits() && Tracklets()[j].LastRow() >= Tracklets()[j].FirstRow())
		{
			int nHits = 0;;
			for (int i = Tracklets()[j].FirstRow();i <= Tracklets()[j].LastRow();i++)
			{
#ifdef EXTERN_ROW_HITS
				calink ih = fTrackletRowHits[i * fCommonMem->fNTracklets + j];
#else
				calink ih = Tracklets()[j].RowHit(i);
#endif
				if (ih != CALINK_INVAL)
				{
					nHits++;
				}
#ifdef EXTERN_ROW_HITS
				out << i << "-" << fTrackletRowHits[i * fCommonMem->fNTracklets + j] << ", ";
#else
				out << i << "-" << Tracklets()[j].RowHit(i) << ", ";
#endif
			}
			if (nHits != Tracklets()[j].NHits())
			{
				std::cout << std::endl << "Wrong NHits!: Expected " << Tracklets()[j].NHits() << ", found " << nHits;
				out << std::endl << "Wrong NHits!: Expected " << Tracklets()[j].NHits() << ", found " << nHits;
			}
		}
		out << std::endl;
	}
}
