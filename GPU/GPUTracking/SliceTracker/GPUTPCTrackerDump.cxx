// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCTrackerDump.cxx
/// \author David Rohr

#include "GPUTPCTracker.h"
#include "GPUTPCSliceOutput.h"
#include "GPUReconstruction.h"
#include "GPUTPCHitId.h"
#include "GPUTPCTrack.h"

#include <iostream>
#include <cstring>
#include <iomanip>

using namespace GPUCA_NAMESPACE::gpu;

void GPUTPCTracker::DumpOutput(std::ostream& out)
{
#ifndef LATE_TPC_TRANSFORM
  out << "Slice " << mISlice << "\n";
  const GPUTPCSliceOutTrack* track = (Output())->GetFirstTrack();
  for (unsigned int j = 0; j < (Output())->NTracks(); j++) {
    out << "Track " << j << " (" << track->NClusters() << "): ";
    for (int k = 0; k < track->NClusters(); k++) {
      out << "(" << track->Cluster(k).GetX() << "," << track->Cluster(k).GetY() << "," << track->Cluster(k).GetZ() << ") ";
    }
    out << " - (" << track->Param().Y() << " " << track->Param().Z() << " " << track->Param().SinPhi() << " " << track->Param().DzDs() << " " << track->Param().QPt() << "\n";
    track = track->GetNextTrack();
  }
#endif
}

void GPUTPCTracker::DumpSliceData(std::ostream& out)
{
  // Dump Slice Input Data to File
  out << "Slice Data (Slice" << mISlice << "):" << std::endl;
  for (int i = 0; i < GPUCA_ROW_COUNT; i++) {
    if (Row(i).NHits() == 0) {
      continue;
    }
    out << "Row: " << i << std::endl;
    for (int j = 0; j < Row(i).NHits(); j++) {
      if (j && j % 16 == 0) {
        out << std::endl;
      }
      out << j << '-' << Data().HitDataY(Row(i), j) << '-' << Data().HitDataZ(Row(i), j) << ", ";
    }
    out << std::endl;
  }
}

void GPUTPCTracker::DumpLinks(std::ostream& out)
{
  // Dump Links (after Neighbours Finder / Cleaner) to file
  out << "Hit Links(Slice" << mISlice << "):" << std::endl;
  for (int i = 0; i < GPUCA_ROW_COUNT; i++) {
    if (Row(i).NHits() == 0) {
      continue;
    }
    out << "Row: " << i << std::endl;
    for (int j = 0; j < Row(i).NHits(); j++) {
      if (j && j % 32 == 0) {
        out << std::endl;
      }
      out << HitLinkUpData(Row(i), j) << "/" << HitLinkDownData(Row(i), j) << ", ";
    }
    out << std::endl;
  }
}

void GPUTPCTracker::DumpHitWeights(std::ostream& out)
{
  // dump hit weights to file
  out << "Hit Weights(Slice" << mISlice << "):" << std::endl;
  for (int i = 0; i < GPUCA_ROW_COUNT; i++) {
    if (Row(i).NHits() == 0) {
      continue;
    }
    out << "Row: " << i << ":" << std::endl;
    for (int j = 0; j < Row(i).NHits(); j++) {
      if (j && j % 32 == 0) {
        out << std::endl;
      }
      out << HitWeight(Row(i), j) << ", ";
    }
    out << std::endl;
  }
}

int GPUTPCTracker::StarthitSortComparison(const void* a, const void* b)
{
  // qsort helper function to sort start hits
  const GPUTPCHitId* aa = reinterpret_cast<const GPUTPCHitId*>(a);
  const GPUTPCHitId* bb = reinterpret_cast<const GPUTPCHitId*>(b);

  if (aa->RowIndex() != bb->RowIndex()) {
    return (aa->RowIndex() - bb->RowIndex());
  }
  return (aa->HitIndex() - bb->HitIndex());
}

void GPUTPCTracker::DumpStartHits(std::ostream& out)
{
  // sort start hits and dump to file
  out << "Start Hits: (Slice" << mISlice << ") (" << *NTracklets() << ")" << std::endl;
  if (mRec->GetDeviceProcessingSettings().comparableDebutOutput) {
    qsort(TrackletStartHits(), *NTracklets(), sizeof(GPUTPCHitId), StarthitSortComparison);
  }
  for (unsigned int i = 0; i < *NTracklets(); i++) {
    out << TrackletStartHit(i).RowIndex() << "-" << TrackletStartHit(i).HitIndex() << std::endl;
  }
  out << std::endl;
}

void GPUTPCTracker::DumpTrackHits(std::ostream& out)
{
  // dump tracks to file
  out << "Tracks: (Slice" << mISlice << ") (" << *NTracks() << ")" << std::endl;
  for (int k = 0; k < GPUCA_ROW_COUNT; k++) {
    for (int l = 0; l < Row(k).NHits(); l++) {
      for (unsigned int j = 0; j < *NTracks(); j++) {
        if (Tracks()[j].NHits() == 0 || !Tracks()[j].Alive()) {
          continue;
        }
        if (TrackHits()[Tracks()[j].FirstHitID()].RowIndex() == k && TrackHits()[Tracks()[j].FirstHitID()].HitIndex() == l) {
          for (int i = 0; i < Tracks()[j].NHits(); i++) {
            out << TrackHits()[Tracks()[j].FirstHitID() + i].RowIndex() << "-" << TrackHits()[Tracks()[j].FirstHitID() + i].HitIndex() << ", ";
          }
          if (!mRec->GetDeviceProcessingSettings().comparableDebutOutput) {
            out << "(Track: " << j << ")";
          }
          out << std::endl;
        }
      }
    }
  }
}

void GPUTPCTracker::DumpTrackletHits(std::ostream& out)
{
  // dump tracklets to file
  int nTracklets = *NTracklets();
  if (nTracklets < 0) {
    nTracklets = 0;
  }
  out << "Tracklets: (Slice" << mISlice << ") (" << nTracklets << ")" << std::endl;
  if (mRec->GetDeviceProcessingSettings().comparableDebutOutput) {
    GPUTPCHitId* tmpIds = new GPUTPCHitId[nTracklets];
    GPUTPCTracklet* tmpTracklets = new GPUTPCTracklet[nTracklets];
    memcpy(tmpIds, TrackletStartHits(), nTracklets * sizeof(GPUTPCHitId));
    memcpy(tmpTracklets, Tracklets(), nTracklets * sizeof(GPUTPCTracklet));
#ifdef GPUCA_EXTERN_ROW_HITS
    calink* tmpHits = new calink[nTracklets * GPUCA_ROW_COUNT];
    memcpy(tmpHits, TrackletRowHits(), nTracklets * GPUCA_ROW_COUNT * sizeof(calink));
#endif
    qsort(TrackletStartHits(), nTracklets, sizeof(GPUTPCHitId), StarthitSortComparison);
    for (int i = 0; i < nTracklets; i++) {
      for (int j = 0; j < nTracklets; j++) {
        if (tmpIds[i].RowIndex() == TrackletStartHit(j).RowIndex() && tmpIds[i].HitIndex() == TrackletStartHit(j).HitIndex()) {
          memcpy(&Tracklets()[j], &tmpTracklets[i], sizeof(GPUTPCTracklet));
#ifdef GPUCA_EXTERN_ROW_HITS
          if (tmpTracklets[i].NHits()) {
            for (int k = tmpTracklets[i].FirstRow(); k <= tmpTracklets[i].LastRow(); k++) {
              const int pos = k * nTracklets + j;
              if (pos < 0 || pos >= (int)mNMaxTracklets * GPUCA_ROW_COUNT) {
                GPUError("internal error: invalid tracklet position k=%d j=%d pos=%d", k, j, pos);
              } else {
                mTrackletRowHits[pos] = tmpHits[k * nTracklets + i];
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
#ifdef GPUCA_EXTERN_ROW_HITS
    delete[] tmpHits;
#endif
  }
  for (int j = 0; j < nTracklets; j++) {
    out << "Tracklet " << std::setw(4) << j << " (Hits: " << std::setw(3) << Tracklets()[j].NHits() << ", Start: " << std::setw(3) << TrackletStartHit(j).RowIndex() << "-" << std::setw(3) << TrackletStartHit(j).HitIndex() << ", Rows: " << (Tracklets()[j].NHits() ? Tracklets()[j].FirstRow() : -1)
        << " - " << (Tracklets()[j].NHits() ? Tracklets()[j].LastRow() : -1) << ") ";
    if (Tracklets()[j].NHits() == 0) {
      ;
    } else if (Tracklets()[j].LastRow() > Tracklets()[j].FirstRow() && (Tracklets()[j].FirstRow() >= GPUCA_ROW_COUNT || Tracklets()[j].LastRow() >= GPUCA_ROW_COUNT)) {
      GPUError("Error: Tracklet %d First %d Last %d Hits %d", j, Tracklets()[j].FirstRow(), Tracklets()[j].LastRow(), Tracklets()[j].NHits());
      out << " (Error: Tracklet " << j << " First " << Tracklets()[j].FirstRow() << " Last " << Tracklets()[j].LastRow() << " Hits " << Tracklets()[j].NHits() << ") ";
      for (int i = 0; i < GPUCA_ROW_COUNT; i++) {
// if (Tracklets()[j].RowHit(i) != CALINK_INVAL)
#ifdef GPUCA_EXTERN_ROW_HITS
        out << i << "-" << mTrackletRowHits[i * mCommonMem->nTracklets + j] << ", ";
#else
        out << i << "-" << Tracklets()[j].RowHit(i) << ", ";
#endif
      }
    } else if (Tracklets()[j].NHits() && Tracklets()[j].LastRow() >= Tracklets()[j].FirstRow()) {
      int nHits = 0;
      for (int i = Tracklets()[j].FirstRow(); i <= Tracklets()[j].LastRow(); i++) {
#ifdef GPUCA_EXTERN_ROW_HITS
        calink ih = mTrackletRowHits[i * mCommonMem->nTracklets + j];
#else
        calink ih = Tracklets()[j].RowHit(i);
#endif
        if (ih != CALINK_INVAL) {
          nHits++;
        }

#ifdef GPUCA_EXTERN_ROW_HITS
        out << i << "-" << mTrackletRowHits[i * mCommonMem->nTracklets + j] << ", ";
#else
        out << i << "-" << Tracklets()[j].RowHit(i) << ", ";
#endif
      }
      if (nHits != Tracklets()[j].NHits()) {
        std::cout << std::endl
                  << "Wrong NHits!: Expected " << Tracklets()[j].NHits() << ", found " << nHits;
        out << std::endl
            << "Wrong NHits!: Expected " << Tracklets()[j].NHits() << ", found " << nHits;
      }
    }
    out << std::endl;
  }
}
