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
#include <algorithm>
#include <numeric>

using namespace GPUCA_NAMESPACE::gpu;

void GPUTPCTracker::DumpOutput(std::ostream& out)
{
  if (Param().par.earlyTpcTransform) {
    out << "Slice " << mISlice << "\n";
    const GPUTPCTrack* track = (Output())->GetFirstTrack();
    for (unsigned int j = 0; j < (Output())->NTracks(); j++) {
      out << "Track " << j << " (" << track->NHits() << "): ";
      for (int k = 0; k < track->NHits(); k++) {
        out << "(" << track->OutTrackCluster(k).GetX() << "," << track->OutTrackCluster(k).GetY() << "," << track->OutTrackCluster(k).GetZ() << ") ";
      }
      out << " - (" << track->Param().Y() << " " << track->Param().Z() << " " << track->Param().SinPhi() << " " << track->Param().DzDs() << " " << track->Param().QPt() << "\n";
      track = track->GetNextTrack();
    }
  }
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
  out << "Start Hits: (Slice" << mISlice << ") (" << *NStartHits() << ")" << std::endl;
  if (mRec->GetProcessingSettings().comparableDebutOutput) {
    qsort(TrackletStartHits(), *NStartHits(), sizeof(GPUTPCHitId), StarthitSortComparison);
  }
  for (unsigned int i = 0; i < *NStartHits(); i++) {
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
        if (Tracks()[j].NHits() == 0) {
          continue;
        }
        if (TrackHits()[Tracks()[j].FirstHitID()].RowIndex() == k && TrackHits()[Tracks()[j].FirstHitID()].HitIndex() == l) {
          for (int i = 0; i < Tracks()[j].NHits(); i++) {
            out << TrackHits()[Tracks()[j].FirstHitID() + i].RowIndex() << "-" << TrackHits()[Tracks()[j].FirstHitID() + i].HitIndex() << ", ";
          }
          if (!mRec->GetProcessingSettings().comparableDebutOutput) {
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
  std::vector<int> Ids(nTracklets);
  std::iota(Ids.begin(), Ids.end(), 0);
  if (mRec->GetProcessingSettings().comparableDebutOutput) {
    std::sort(Ids.begin(), Ids.end(), [this](const int& a, const int& b) {
      if (this->Tracklets()[a].FirstRow() == this->Tracklets()[b].FirstRow()) {
        return this->Tracklets()[a].Param().Y() > this->Tracklets()[b].Param().Y();
      }
      return this->Tracklets()[a].FirstRow() > this->Tracklets()[b].FirstRow();
    });
  }
  for (int jj = 0; jj < nTracklets; jj++) {
    const int j = Ids[jj];
    const auto& tracklet = Tracklets()[j];
    out << "Tracklet " << std::setw(4) << jj << " (Hits: " << std::setw(3) << Tracklets()[j].NHits() << ", Rows: " << (Tracklets()[j].NHits() ? Tracklets()[j].FirstRow() : -1)
        << " - " << (tracklet.NHits() ? tracklet.LastRow() : -1) << ") ";
    if (tracklet.NHits() == 0) {
      ;
    } else if (tracklet.LastRow() > tracklet.FirstRow() && (tracklet.FirstRow() >= GPUCA_ROW_COUNT || tracklet.LastRow() >= GPUCA_ROW_COUNT)) {
      GPUError("Error: Tracklet %d First %d Last %d Hits %d", j, tracklet.FirstRow(), tracklet.LastRow(), tracklet.NHits());
      out << " (Error: Tracklet " << j << " First " << tracklet.FirstRow() << " Last " << tracklet.LastRow() << " Hits " << tracklet.NHits() << ") ";
      for (int i = 0; i < GPUCA_ROW_COUNT; i++) {
        // if (tracklet.RowHit(i) != CALINK_INVAL)
        out << i << "-" << mTrackletRowHits[tracklet.FirstHit() + (i - tracklet.FirstRow())] << ", ";
      }
    } else if (tracklet.NHits() && tracklet.LastRow() >= tracklet.FirstRow()) {
      int nHits = 0;
      for (int i = tracklet.FirstRow(); i <= tracklet.LastRow(); i++) {
        calink ih = mTrackletRowHits[tracklet.FirstHit() + (i - tracklet.FirstRow())];
        if (ih != CALINK_INVAL) {
          nHits++;
        }
        out << i << "-" << mTrackletRowHits[tracklet.FirstHit() + (i - tracklet.FirstRow())] << ", ";
      }
      if (nHits != tracklet.NHits()) {
        std::cout << std::endl
                  << "Wrong NHits!: Expected " << tracklet.NHits() << ", found " << nHits;
        out << std::endl
            << "Wrong NHits!: Expected " << tracklet.NHits() << ", found " << nHits;
      }
    }
    out << std::endl;
  }
}
