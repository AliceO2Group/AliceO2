// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataInterpreterITS.cxx
/// \brief converting ITS data to Event Visualisation primitives
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch

#include "EventVisualisationBase/ConfigurationManager.h"
#include "EventVisualisationDetectors/DataInterpreterITS.h"
#include "EventVisualisationDataConverter/VisualisationEvent.h"

#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSBase/GeometryTGeo.h"

#include <TEveManager.h>
#include <TEveTrackPropagator.h>
#include <TEveTrack.h>
#include <TGListTree.h>
#include <TFile.h>
#include <TTree.h>
#include <TVector2.h>

#include <gsl/span>

using namespace std;

namespace o2
{
namespace event_visualisation
{

DataInterpreterITS::DataInterpreterITS()
{
  //Prepare coordinate translator
  base::GeometryManager::loadGeometry("O2geometry.root", "FAIRGeom");
  its::GeometryTGeo* gman = its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2GRot));
}

std::unique_ptr<VisualisationEvent> DataInterpreterITS::interpretDataForType(TObject* data, EVisualisationDataType type)
{
  TList* list = (TList*)data;
  Int_t event = ((TVector2*)list->At(2))->X();

  auto ret_event = std::make_unique<VisualisationEvent>(0, 0, 0, 0, "", 0);

  if (type == Clusters) {
    its::GeometryTGeo* gman = its::GeometryTGeo::Instance();

    TFile* clustFile = (TFile*)list->At(1);
    TTree* clusters = (TTree*)clustFile->Get("o2sim");

    //Read all clusters to a buffer
    std::vector<itsmft::Cluster>* clusArr = nullptr;
    clusters->SetBranchAddress("ITSCluster", &clusArr);

    std::vector<itsmft::ROFRecord>* clusterROFrames = nullptr;
    clusters->SetBranchAddress("ITSClustersROF", &clusterROFrames);
    clusters->GetEntry(0);

    auto currentClusterROF = clusterROFrames->at(event);

    int first, last;
    first = currentClusterROF.getFirstEntry();
    last = first + currentClusterROF.getNEntries();

    gsl::span<itsmft::Cluster> mClusters = gsl::make_span(&(*clusArr)[first], last - first);

    for (const auto& c : mClusters) {
      const auto& gloC = c.getXYZGloRot(*gman);
      double xyz[3] = {gloC.X(), gloC.Y(), gloC.Z()};
      VisualisationCluster cluster(xyz);

      ret_event->addCluster(cluster);
    }
  } else if (type == ESD) {
    TFile* trackFile = (TFile*)list->At(0);
    TFile* clustFile = (TFile*)list->At(1);

    TTree* tracks = (TTree*)trackFile->Get("o2sim");

    TTree* clusters = (TTree*)clustFile->Get("o2sim");

    //Read all tracks to a buffer
    std::vector<its::TrackITS>* trkArr = nullptr;
    tracks->SetBranchAddress("ITSTrack", &trkArr);

    //Read all track RO frames to a buffer
    std::vector<itsmft::ROFRecord>* trackROFrames = nullptr;
    tracks->SetBranchAddress("ITSTracksROF", &trackROFrames);

    tracks->GetEntry(0);

    //Read all clusters to a buffer
    std::vector<itsmft::Cluster>* clusArr = nullptr;
    clusters->SetBranchAddress("ITSCluster", &clusArr);

    std::vector<itsmft::ROFRecord>* clusterROFrames = nullptr;
    clusters->SetBranchAddress("ITSClustersROF", &clusterROFrames);
    clusters->GetEntry(0);

    TEveTrackList* trackList = new TEveTrackList("tracks");
    trackList->IncDenyDestroy();
    auto prop = trackList->GetPropagator();
    prop->SetMagField(0.5);
    prop->SetMaxR(50.);

    auto currentTrackROF = trackROFrames->at(event);

    int first, last;
    first = currentTrackROF.getFirstEntry();
    last = first + currentTrackROF.getNEntries();

    gsl::span<its::TrackITS> mTracks = gsl::make_span(&(*trkArr)[first], last - first);

    for (const auto& rec : mTracks) {
      std::array<float, 3> p;
      rec.getPxPyPzGlo(p);
      TEveRecTrackD t;
      t.fP = {p[0], p[1], p[2]};
      t.fSign = (rec.getSign() < 0) ? -1 : 1;
      TEveTrack* eve_track = new TEveTrack(&t, prop);
      eve_track->MakeTrack();

      auto start = eve_track->GetLineStart();
      auto end = eve_track->GetLineEnd();
      double track_start[3] = {start.fX, start.fY, start.fZ};
      double track_end[3] = {end.fX, end.fY, end.fZ};
      double track_p[3] = {p[0], p[1], p[2]};

      VisualisationTrack track(rec.getSign(), 0.0, 0, 0, 0.0, 0.0, track_start, track_end, track_p, 0, 0.0, 0.0, 0.0, 0);

      for (Int_t i = 0; i < eve_track->GetN(); ++i) {
        Float_t x, y, z;
        eve_track->GetPoint(i, x, y, z);
        track.addPolyPoint(x, y, z);
      }
      delete eve_track;

      //                    TEvePointSet* tpoints = new TEvePointSet("tclusters");
      //                    int nc = rec.getNumberOfClusters();
      //                    while (nc--) {
      //                        Int_t idx = rec.getClusterEntry(nc);
      //                        itsmft::Cluster& c = (*clusArr)[idx];
      //                        const auto& gloC = c.getXYZGloRot(*gman);
      //                        tpoints->SetNextPoint(gloC.X(), gloC.Y(), gloC.Z());
      //                    }

      ret_event->addTrack(track);
    }
    delete trackList;
  }
  return ret_event;
}

} // namespace event_visualisation
} // namespace o2
