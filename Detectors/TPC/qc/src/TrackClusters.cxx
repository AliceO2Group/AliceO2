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

#define _USE_MATH_DEFINES

#include <cmath>
#include <memory>

// root includes
#include "TFile.h"
#include "TObjArray.h"

// o2 includes
#include "DataFormatsTPC/TrackTPC.h"
#include "TPCQC/TrackClusters.h"
#include "TPCQC/Tracks.h"
#include "TPCQC/Helpers.h"
#include "GPUO2InterfaceRefit.h"
#include "GlobalTracking/TrackMethods.h"

ClassImp(o2::tpc::qc::TrackClusters);
using namespace o2::tpc::qc;

struct binning {
  int bins;
  double min;
  double max;
};

const binning binsSharedClusters{160, 0., 160.};
const binning binsFoundClusters{160, 0., 160.};
const binning binsCrossedRows{160, 0., 160.};

//______________________________________________________________________________
void TrackClusters::initializeHistograms()
{
  TH1::AddDirectory(false);
  mMapHist["sharedClusters"].emplace_back(std::make_unique<TH1F>("sharedClusters", "sharedClusters;NSharedClusters;Entries", binsSharedClusters.bins, binsSharedClusters.min, binsSharedClusters.max));
  mMapHist["foundClusters"].emplace_back(std::make_unique<TH1F>("foundClusters", "foundClusters;foundClusters;Entries", binsFoundClusters.bins, binsFoundClusters.min, binsFoundClusters.max));
  mMapHist["crossedRows"].emplace_back(std::make_unique<TH1F>("crossedRows", "crossedRows;crossedRows;Entries", binsCrossedRows.bins, binsCrossedRows.min, binsCrossedRows.max));
}

//______________________________________________________________________________
void TrackClusters::resetHistograms()
{
  for (const auto& pair : mMapHist) {
    for (auto& hist : pair.second) {
      hist->Reset();
    }
  }
}

//______________________________________________________________________________
bool TrackClusters::processTrackAndClusters(const std::vector<o2::tpc::TrackTPC>* tracks, const o2::tpc::ClusterNativeAccess* clusterIndex, std::vector<o2::tpc::TPCClRefElem>* clusRefs)
{

  std::vector<unsigned char> mBufVec;
  mBufVec.resize(clusterIndex->nClustersTotal);

  o2::gpu::GPUO2InterfaceRefit::fillSharedClustersMap(clusterIndex, *tracks, clusRefs->data(), mBufVec.data());

  for (auto const& track : (*tracks)) {
    const auto dEdxTot = track.getdEdx().dEdxTotTPC;
    const auto nCls = uint8_t(track.getNClusters());
    const auto eta = track.getEta();

    if (nCls < mCutMinNCls || dEdxTot < mCutMindEdxTot || abs(eta) > mCutAbsEta) {
      continue;
    }

    uint8_t shared = 200, found = 0, crossed = 0;

    o2::TrackMethods::countTPCClusters(track, *clusRefs, mBufVec, *clusterIndex, shared, found, crossed);

    mMapHist["sharedClusters"][0]->Fill(shared);
    mMapHist["foundClusters"][0]->Fill(found);
    mMapHist["crossedRows"][0]->Fill(crossed);
  }

  return true;
}

//______________________________________________________________________________
void TrackClusters::dumpToFile(const std::string filename)
{
  auto f = std::unique_ptr<TFile>(TFile::Open(filename.c_str(), "recreate"));
  for (const auto& [name, histos] : mMapHist) {
    TObjArray arr;
    arr.SetName(name.data());
    for (auto& hist : histos) {
      arr.Add(hist.get());
    }
    arr.Write(arr.GetName(), TObject::kSingleKey);
  }
  f->Close();
}
