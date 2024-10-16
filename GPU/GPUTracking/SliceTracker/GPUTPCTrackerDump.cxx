// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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
    out << "\nSlice " << mISlice << "\n";
    const GPUTPCTrack* track = (Output())->GetFirstTrack();
    for (uint32_t j = 0; j < (Output())->NTracks(); j++) {
      out << "Track " << j << " (" << track->NHits() << "): ";
      for (int32_t k = 0; k < track->NHits(); k++) {
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
  out << "\nSlice Data (Slice" << mISlice << "):" << std::endl;
  for (int32_t i = 0; i < GPUCA_ROW_COUNT; i++) {
    if (Row(i).NHits() == 0) {
      continue;
    }
    out << "Row: " << i << std::endl;
    for (int32_t j = 0; j < Row(i).NHits(); j++) {
      if (j && j % 16 == 0) {
        out << std::endl;
      }
      out << j << '-' << Data().HitDataY(Row(i), j) << '-' << Data().HitDataZ(Row(i), j) << ", ";
    }
    out << std::endl;
  }
}

void GPUTPCTracker::DumpLinks(std::ostream& out, int32_t phase)
{
  // Dump Links (after Neighbours Finder / Cleaner) to file
  out << "\nHit Links (Phase " << phase << ", Slice" << mISlice << "):" << std::endl;
  for (int32_t i = 0; i < GPUCA_ROW_COUNT; i++) {
    if (Row(i).NHits() == 0) {
      continue;
    }
    out << "Row: " << i << std::endl;
    for (int32_t j = 0; j < Row(i).NHits(); j++) {
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
  out << "\nHit Weights(Slice" << mISlice << "):" << std::endl;
  for (int32_t i = 0; i < GPUCA_ROW_COUNT; i++) {
    if (Row(i).NHits() == 0) {
      continue;
    }
    out << "Row: " << i << ":" << std::endl;
    for (int32_t j = 0; j < Row(i).NHits(); j++) {
      if (j && j % 32 == 0) {
        out << std::endl;
      }
      out << HitWeight(Row(i), j) << ", ";
    }
    out << std::endl;
  }
}

void GPUTPCTracker::DumpStartHits(std::ostream& out)
{
  // dump start hits to file
  out << "\nStart Hits: (Slice" << mISlice << ") (" << *NStartHits() << ")" << std::endl;
  for (uint32_t i = 0; i < *NStartHits(); i++) {
    out << TrackletStartHit(i).RowIndex() << "-" << TrackletStartHit(i).HitIndex() << std::endl;
  }
  out << std::endl;
}

void GPUTPCTracker::DumpTrackHits(std::ostream& out)
{
  // dump tracks to file
  out << "\nTracks: (Slice" << mISlice << ") (" << *NTracks() << ")" << std::endl;
  for (uint32_t j = 0; j < *NTracks(); j++) {
    if (Tracks()[j].NHits() == 0) {
      continue;
    }
    const GPUTPCBaseTrackParam& p = Tracks()[j].Param();
    out << "  " << j << " x " << p.GetX() << " offset " << p.GetZOffset() << " y " << p.GetY() << " z " << p.GetZ() << " snp " << p.GetSinPhi() << " tgl " << p.GetDzDs() << " qpt " << p.GetQPt() << " - ";
    for (int32_t k = 0; k < 15; k++) {
      out << p.GetCov(k) << " ";
    }
    out << "- ";
    for (int32_t i = 0; i < Tracks()[j].NHits(); i++) {
      out << TrackHits()[Tracks()[j].FirstHitID() + i].RowIndex() << "-" << TrackHits()[Tracks()[j].FirstHitID() + i].HitIndex() << ", ";
    }
    if (!mRec->GetProcessingSettings().deterministicGPUReconstruction) {
      out << "(Track: " << j << ")";
    }
    out << std::endl;
  }
}

void GPUTPCTracker::DumpTrackletHits(std::ostream& out)
{
  // dump tracklets to file
  int32_t nTracklets = *NTracklets();
  if (nTracklets < 0) {
    nTracklets = 0;
  }
  out << "\nTracklets: (Slice" << mISlice << ") (" << nTracklets << ")" << std::endl;
  std::vector<int32_t> Ids(nTracklets);
  std::iota(Ids.begin(), Ids.end(), 0);
  if (mRec->GetProcessingSettings().deterministicGPUReconstruction) {
    std::sort(Ids.begin(), Ids.end(), [this](const int32_t& a, const int32_t& b) {
      if (this->Tracklets()[a].FirstRow() != this->Tracklets()[b].FirstRow()) {
        return this->Tracklets()[a].FirstRow() > this->Tracklets()[b].FirstRow();
      }
      if (this->Tracklets()[a].LastRow() != this->Tracklets()[b].LastRow()) {
        return this->Tracklets()[a].LastRow() > this->Tracklets()[b].LastRow();
      }
      if (this->Tracklets()[a].Param().Y() != this->Tracklets()[b].Param().Y()) {
        return this->Tracklets()[a].Param().Y() > this->Tracklets()[b].Param().Y();
      }
      return this->Tracklets()[a].Param().Z() > this->Tracklets()[b].Param().Z();
    });
  }
  for (int32_t jj = 0; jj < nTracklets; jj++) {
    const int32_t j = Ids[jj];
    const auto& tracklet = Tracklets()[j];
    out << "Tracklet " << std::setw(4) << jj << " (Rows: " << Tracklets()[j].FirstRow() << " - " << tracklet.LastRow() << ", Weight " << Tracklets()[j].HitWeight() << ") ";
    if (tracklet.LastRow() > tracklet.FirstRow() && (tracklet.FirstRow() >= GPUCA_ROW_COUNT || tracklet.LastRow() >= GPUCA_ROW_COUNT)) {
      GPUError("Error: Tracklet %d First %d Last %d", j, tracklet.FirstRow(), tracklet.LastRow());
      out << " (Error: Tracklet " << j << " First " << tracklet.FirstRow() << " Last " << tracklet.LastRow() << ") ";
      for (int32_t i = 0; i < GPUCA_ROW_COUNT; i++) {
        // if (tracklet.RowHit(i) != CALINK_INVAL)
        out << i << "-" << mTrackletRowHits[tracklet.FirstHit() + (i - tracklet.FirstRow())] << ", ";
      }
    } else if (tracklet.LastRow() >= tracklet.FirstRow()) {
      for (int32_t i = tracklet.FirstRow(); i <= tracklet.LastRow(); i++) {
        out << i << "-" << mTrackletRowHits[tracklet.FirstHit() + (i - tracklet.FirstRow())] << ", ";
      }
    }
    out << std::endl;
  }
}
