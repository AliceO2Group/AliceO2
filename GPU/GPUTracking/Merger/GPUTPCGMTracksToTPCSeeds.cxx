// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGMTracksToTPCSeeds.cxx
/// \author David Rohr

#include "GPUTPCGMTracksToTPCSeeds.h"
#include "GPUTPCGlobalMergerComponent.h"
#include "GPUTPCGMMerger.h"
#include "GPULogging.h"
#include "AliTPCtracker.h"
#include "AliTPCtrack.h"
#include "AliTPCseed.h"
#include "AliTPCtrackerSector.h"
#include "TObjArray.h"
#include "AliTPCclusterMI.h"

using namespace GPUCA_NAMESPACE::gpu;

void GPUTPCGMTracksToTPCSeeds::CreateSeedsFromHLTTracks(TObjArray* seeds, AliTPCtracker* tpctracker)
{
  const GPUTPCGMMerger* merger = GPUTPCGlobalMergerComponent::GetCurrentMerger();
  if (merger == nullptr) {
    return;
  }
  seeds->Clear();
  int index = 0;
  for (int i = 0; i < merger->NOutputTracks(); i++) {
    const GPUTPCGMMergedTrack& track = merger->OutputTracks()[i];
    if (!track.OK()) {
      continue;
    }

    AliTPCtrack tr;
    tr.Set(track.GetParam().GetX(), track.GetAlpha(), track.GetParam().GetPar(), track.GetParam().GetCov());
    AliTPCseed* seed = new (tpctracker->NextFreeSeed()) AliTPCseed(tr);
    for (int j = 0; j < GPUCA_ROW_COUNT; j++) {
      seed->SetClusterPointer(j, nullptr);
      seed->SetClusterIndex(j, -1);
    }
    int ncls = 0;
    int lastrow = -1;
    int lastleg = -1;
    for (int j = track.NClusters() - 1; j >= 0; j--) {
      const GPUTPCGMMergedTrackHit& cls = merger->Clusters()[track.FirstClusterRef() + j];
      if (cls.state & GPUTPCGMMergedTrackHit::flagReject) {
        continue;
      }
      if (lastrow != -1 && (cls.row < lastrow || cls.leg != lastleg)) {
        break;
      }
      if (cls.row == lastrow) {
        continue;
      }

      AliTPCtrackerRow& row = tpctracker->GetRow(cls.slice % 18, cls.row);
      unsigned int clIndexOffline = 0;
      AliTPCclusterMI* clOffline = row.FindNearest2(cls.y, cls.z, 0.01f, 0.01f, clIndexOffline);
      if (!clOffline) {
        continue;
      }
      clIndexOffline = row.GetIndex(clIndexOffline);

      clOffline->Use(10);
      seed->SetClusterPointer(cls.row, clOffline);
      seed->SetClusterIndex2(cls.row, clIndexOffline);

      lastrow = cls.row;
      lastleg = cls.leg;
      ncls++;
    }

    seed->SetRelativeSector(track.GetAlpha() / (M_PI / 9.f));
    seed->SetNumberOfClusters(ncls);
    seed->SetNFoundable(ncls);
    seed->SetChi2(track.GetParam().GetChi2());

    float alpha = seed->GetAlpha();
    if (alpha >= 2 * M_PI) {
      alpha -= 2. * M_PI;
    }
    if (alpha < 0) {
      alpha += 2. * M_PI;
    }
    seed->SetRelativeSector(track.GetAlpha() / (M_PI / 9.f));

    seed->SetPoolID(tpctracker->GetLastSeedId());
    seed->SetIsSeeding(kTRUE);
    seed->SetSeed1(GPUCA_ROW_COUNT - 1);
    seed->SetSeed2(GPUCA_ROW_COUNT - 2);
    seed->SetSeedType(0);
    seed->SetFirstPoint(-1);
    seed->SetLastPoint(-1);
    seeds->AddLast(seed); // note, track is seed, don't free the seed
    index++;
  }
}

void GPUTPCGMTracksToTPCSeeds::UpdateParamsOuter(TObjArray* seeds)
{
  const GPUTPCGMMerger* merger = GPUTPCGlobalMergerComponent::GetCurrentMerger();
  if (merger == nullptr) {
    return;
  }
  int index = 0;
  for (int i = 0; i < merger->NOutputTracks(); i++) {
    const GPUTPCGMMergedTrack& track = merger->OutputTracks()[i];
    if (!track.OK()) {
      continue;
    }
    if (index > seeds->GetEntriesFast()) {
      GPUError("Invalid number of offline seeds");
      return;
    }
    AliTPCseed* seed = (AliTPCseed*)seeds->UncheckedAt(index++);
    const GPUTPCGMTrackParam::GPUTPCGMTrackParam::GPUTPCOuterParam& param = track.OuterParam();
    seed->Set(param.X, param.alpha, param.P, param.C);
  }
}

void GPUTPCGMTracksToTPCSeeds::UpdateParamsInner(TObjArray* seeds)
{
  const GPUTPCGMMerger* merger = GPUTPCGlobalMergerComponent::GetCurrentMerger();
  if (merger == nullptr) {
    return;
  }
  int index = 0;
  for (int i = 0; i < merger->NOutputTracks(); i++) {
    const GPUTPCGMMergedTrack& track = merger->OutputTracks()[i];
    if (!track.OK()) {
      continue;
    }
    if (index > seeds->GetEntriesFast()) {
      GPUError("Invalid number of offline seeds");
      return;
    }
    AliTPCseed* seed = (AliTPCseed*)seeds->UncheckedAt(index++);
    seed->Set(track.GetParam().GetX(), track.GetAlpha(), track.GetParam().GetPar(), track.GetParam().GetCov());
  }
}
