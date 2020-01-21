// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "DetectorsBase/DCAFitter.h"
#include "ReconstructionDataFormats/Track.h"
//#include <ROOT/RDataframe.hxx>
//#include <ROOT/RArrowDS.hxx>

#include <TFile.h>
#include <TH1F.h>
#include <cmath>
#include <array>
namespace o2::aod
{
namespace etaphi
{
DECLARE_SOA_COLUMN(Eta, etas, float, "fEta");
DECLARE_SOA_COLUMN(Phi, phis, float, "fPhi");
} // namespace etaphi
namespace secvtx
{
DECLARE_SOA_COLUMN(Posx, posx, float, "fPosx");
DECLARE_SOA_COLUMN(Posy, posy, float, "fPosy");
DECLARE_SOA_COLUMN(Label0, label0, float, "fLabel0");
DECLARE_SOA_COLUMN(Label1, label1, float, "fLabel1");
DECLARE_SOA_COLUMN(Label2, label2, float, "fLabel2");
} // namespace secvtx

DECLARE_SOA_TABLE(EtaPhi, "RN2", "ETAPHI",
                  etaphi::Eta, etaphi::Phi);
DECLARE_SOA_TABLE(SecVtx, "AOD", "SECVTX",
                  secvtx::Posx, secvtx::Posy, secvtx::Label0, secvtx::Label1, secvtx::Label2);
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

// This is a very simple example showing how to iterate over tracks
// and create a new collection for them.
// FIXME: this should really inherit from AnalysisTask but
//        we need GCC 7.4+ for that
struct ATask {
  Produces<aod::EtaPhi> etaphi;

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      auto phi = asin(track.snp()) + track.alpha() + M_PI;
      auto eta = log(tan(0.25 * M_PI - 0.5 * atan(track.tgl())));

      etaphi(phi, eta);
    }
  }
};
struct VertexerHFTask {
  OutputObj<TH1F> hvtx_x_out{TH1F("hvtx_x", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hvtx_y_out{TH1F("hvtx_y", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hvtx_z_out{TH1F("hvtx_z", "2-track vtx", 100, -0.1, 0.1)};
  Produces<aod::SecVtx> secvtx;

  //void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksCov> const& tracks)
  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksCov> const& tracks)
  {
    LOGF(info, "Tracks for collision: %d", tracks.size());
    o2::base::DCAFitter df(5.0, 10.);
    auto hvtx_x = new TH1F("hvtx_x", "2-track vtx", 100, -10., 10.);

    for (auto it1 = tracks.begin(); it1 != tracks.end(); ++it1) {
      auto& track1 = *it1;
      float x1_ = track1.x();
      float alpha1_ = track1.alpha();
      std::array<float, 5> arraypar1 = {track1.y(), track1.z(), track1.snp(), track1.tgl(), track1.signed1Pt()};
      std::array<float, 15> covpar1 = {track1.cYY(), track1.cZY(), track1.cZZ(), track1.cSnpY(), track1.cSnpZ(), \
	                               track1.cSnpSnp(), track1.cTglY(), track1.cTglZ(), track1.cTglSnp(), track1.cTglTgl(), \
	                               track1.c1PtY(), track1.c1PtZ(), track1.c1PtSnp(), track1.c1PtTgl(), track1.c1Pt21Pt2()};
      o2::track::TrackParCov trackparvar1(x1_, alpha1_, arraypar1, covpar1);

      for (auto it2 = it1 + 1; it2 != tracks.end(); ++it2) {
        auto& track2 = *it2;
        float x2_ = track2.x();
        float alpha2_ = track2.alpha();
        std::array<float, 5> arraypar2 = {track2.y(), track2.z(), track2.snp(), track2.tgl(), track2.signed1Pt()};
        std::array<float, 15> covpar2 = {track2.cYY(), track2.cZY(), track2.cZZ(), track2.cSnpY(), track2.cSnpZ(), \
	                                 track2.cSnpSnp(), track2.cTglY(), track2.cTglZ(), track2.cTglSnp(), track2.cTglTgl(), \
	                                 track2.c1PtY(), track2.c1PtZ(), track2.c1PtSnp(), track2.c1PtTgl(), track2.c1Pt21Pt2()};
        o2::track::TrackParCov trackparvar2(x2_, alpha2_, arraypar2, covpar2);
        //LOGF(info, "track2 %f", trackparvar2.getSigmaY2());

        df.setUseAbsDCA(true);
        int nCand = df.process(trackparvar1, trackparvar2);
        printf("\n\nTesting with abs DCA minimization: %d candidates found\n", nCand);
        // we can have up to 2 candidates
        for (int ic = 0; ic < nCand; ic++) {
          const o2::base::DCAFitter::Triplet& vtx = df.getPCACandidate(ic);
          LOGF(info, "print %f", vtx.x);
          //LOGF(info, "LABELELELELELLE %d", track2.index());
          hvtx_x_out->Fill(vtx.x);
          hvtx_y_out->Fill(vtx.y);
          hvtx_z_out->Fill(vtx.z);
          secvtx(vtx.x, vtx.y, -1, -1, -1.);
          //secvtx(vtx.x, vtx.y, track1.index, track2.index, -1.);
          //LOGF(info, "DONE");
        }
      }
    }
  }
};

struct SkimVtxTable {
  void process(aod::SecVtx const& secVtxs)
  {
    //auto source = std::make_unique<ROOT::RDF::RArrowDS>(tracks.asArrowTable(), std::vector<std::string>{});
    //ROOT::RDataFrame rdf(std::move(source));
    //rdf.Snapshot("outputTree", "outputFile.root", {"fPosx", "fPosy"});
    for (auto& secVtx : secVtxs) {
      LOGF(INFO, "Consume the table (%f, %f)", secVtx.posx(), secVtx.posy());
      LOGF(INFO, "Labels tracks (%f, %f, %f)", secVtx.label0(), secVtx.label1(), secVtx.label2());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("produce-etaphi"),
    adaptAnalysisTask<VertexerHFTask>("vertexerhf-task"),
    adaptAnalysisTask<SkimVtxTable>("skimvtxtable-task")};
}
