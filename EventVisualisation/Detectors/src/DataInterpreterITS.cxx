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
#include <gsl/span_ext>

namespace o2
{
namespace event_visualisation
{

DataInterpreterITS::DataInterpreterITS()
{
  //Prepare coordinate translator
  base::GeometryManager::loadGeometry();
  its::GeometryTGeo* gman = its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2GRot));
}

VisualisationEvent DataInterpreterITS::interpretDataForType(TObject* data, EVisualisationDataType type)
{
  TList* list = (TList*)data;
  Int_t event = ((TVector2*)list->At(2))->X();

  VisualisationEvent ret_event({.eventNumber = 0,
                                .runNumber = 0,
                                .energy = 0,
                                .multiplicity = 0,
                                .collidingSystem = "",
                                .timeStamp = 0});

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
      ret_event.addCluster(xyz);
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
      VisualisationTrack* track = ret_event.addTrack({.charge = rec.getSign(),
                                                      .energy = 0.0,
                                                      .ID = 0,
                                                      .PID = 0,
                                                      .mass = 0.0,
                                                      .signedPT = 0.0,
                                                      .startXYZ = {start.fX, start.fY, start.fZ},
                                                      .endXYZ = {end.fX, end.fY, end.fZ},
                                                      .pxpypz = {p[0], p[1], p[2]},
                                                      .parentID = 0,
                                                      .phi = 0.0,
                                                      .theta = 0.0,
                                                      .helixCurvature = 0.0,
                                                      .type = 0,
                                                      .source = ITSSource});

      for (Int_t i = 0; i < eve_track->GetN(); ++i) {
        Float_t x, y, z;
        eve_track->GetPoint(i, x, y, z);
        track->addPolyPoint(x, y, z);
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
    }
    delete trackList;
  }
  return ret_event;
}

} // namespace event_visualisation
} // namespace o2
