// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataInterpreterTPC.cxx
/// \brief converting TPC data to Event Visualisation primitives
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch

#include "EventVisualisationDataConverter/VisualisationCluster.h"
#include "EventVisualisationDetectors/DataInterpreterTPC.h"
#include "EventVisualisationBase/ConfigurationManager.h"
#include "EventVisualisationDataConverter/VisualisationEvent.h"

#include <TEveManager.h>
#include <TEveTrackPropagator.h>
#include <TEveTrack.h>
#include <TGListTree.h>
#include <TFile.h>
#include <TTree.h>
#include <TVector2.h>

#include "TPCBase/Mapper.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"

#include <iostream>
#include <gsl/span>

using namespace std;

namespace o2
{
namespace event_visualisation
{

DataInterpreterTPC::~DataInterpreterTPC() = default;

std::unique_ptr<VisualisationEvent> DataInterpreterTPC::interpretDataForType(TObject* data, EVisualisationDataType type)
{
  TList* list = (TList*)data;

  // Int_t event = ((TVector2*)list->At(2))->X();
  auto ret_event = std::make_unique<VisualisationEvent>(0, 0, 0, 0, "", 0);

  if (type == Clusters) {
    TFile* clustFile = (TFile*)list->At(1);

    //Why cannot TPC clusters be read like other clusters?
    auto reader = new tpc::ClusterNativeHelper::Reader();
    reader->init(clustFile->GetName());
    reader->read(0);

    auto access = std::make_unique<o2::tpc::ClusterNativeAccess>();
    std::unique_ptr<tpc::ClusterNative[]> clusterBuffer;
    tpc::MCLabelContainer clusterMCBuffer;

    reader->fillIndex(*access, clusterBuffer, clusterMCBuffer);

    const auto& mapper = tpc::Mapper::instance();
    const auto& clusterRefs = access->clusters;

    for (int sector = 0; sector < o2::tpc::Constants::MAXSECTOR; sector++) {
      for (int row = 0; row < o2::tpc::Constants::MAXGLOBALPADROW; row++) {
        const auto& c = clusterRefs[sector][row];

        const auto pad = mapper.globalPadNumber(tpc::PadPos(row, c->getPad()));
        const tpc::LocalPosition3D localXYZ(mapper.padCentre(pad).X(), mapper.padCentre(pad).Y(), c->getTime());
        const auto globalXYZ = mapper.LocalToGlobal(localXYZ, sector);
        double xyz[3] = {globalXYZ.X(), globalXYZ.Y(), globalXYZ.Z()};

        VisualisationCluster cluster(xyz);
        ret_event->addCluster(cluster);
      }
    }
  } else if (type == ESD) {
    TFile* trackFile = (TFile*)list->At(0);

    TTree* tracks = (TTree*)trackFile->Get("tpcrec");

    //Read all tracks to a buffer
    std::vector<tpc::TrackTPC>* trkArr = nullptr;
    tracks->SetBranchAddress("TPCTracks", &trkArr);
    tracks->GetEntry(0);

    TEveTrackList* trackList = new TEveTrackList("tracks");
    trackList->IncDenyDestroy();
    auto prop = trackList->GetPropagator();
    prop->SetMagField(0.5);

    int first, last;
    first = 0;
    last = trkArr->size();

    gsl::span<tpc::TrackTPC> mTracks = gsl::make_span(&(*trkArr)[first], last - first);

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

      ret_event->addTrack(track);
    }
    delete trackList;
  }
  return ret_event;
}

} // namespace event_visualisation
} // namespace o2
