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

#include <TFile.h>
#include <TH1F.h>
#include <cmath>
#include <array>
namespace o2::aod
{
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

struct ValidationTask {
  OutputObj<TH1F> hpt_nocuts{TH1F("hpt_nocuts", "pt tracks (#GeV)", 100, 0., 10.)};
  OutputObj<TH1F> hrun_number{TH1F("hrun_number", "run number", 1000, 0., 1000000.)};
  OutputObj<TH1F> hfCYY{TH1F("hfCYY", "cYY", 1000, 0., 150.)};
  OutputObj<TH1F> hfCZY{TH1F("hfCZY", "cZY", 1000, -40., 10.)};
  OutputObj<TH1F> hfCZZ{TH1F("hfCZZ", "cZZ", 1000, 0., 150.)};
  OutputObj<TH1F> hfCSnpY{TH1F("hfCSnpY", "cSnpY", 1000, -2.5, 1.)};
  OutputObj<TH1F> hfCSnpZ{TH1F("hfCSnpZ", "cSnpZ", 1000, -2.5, 1.)};
  OutputObj<TH1F> hfCSnpSnp{TH1F("hfCSnpSnp", "cSnpSnp", 1000, 0., 0.1)};
  OutputObj<TH1F> hfCTglY{TH1F("hfCTglY", "cTglY", 1000, -0.1, 0.3)};
  OutputObj<TH1F> hfCTglZ{TH1F("hfCTglZ", "cTglZ", 1000, -3., 3.)};
  OutputObj<TH1F> hfCTglSnp{TH1F("hfCTglSnp", "cTglSnp", 1000, -0.01, 0.01)};
  OutputObj<TH1F> hfCTglTgl{TH1F("hfCTglTgl", "cTglTgl", 1000, 0., 0.2)};
  OutputObj<TH1F> hfC1PtY{TH1F("hfC1PtY", "c1PtY", 1000, 0., 30.)};
  OutputObj<TH1F> hfC1PtZ{TH1F("hfC1PtZ", "c1PtZ", 1000, -9., 3.)};
  OutputObj<TH1F> hfC1PtSnp{TH1F("hfC1PtSnp", "c1PtSnp", 1000, -0.8, 1.)};
  OutputObj<TH1F> hfC1PtTgl{TH1F("hfC1PtTgl", "c1PtTgl", 1000, -0.3, 0.1)};
  OutputObj<TH1F> hfC1Pt21Pt2{TH1F("hfC1Pt21Pt2", "c1Pt21Pt2", 1000, -0.3, 0.1)};

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksCov> const& tracks)
  {
    hrun_number->Fill(collision.runNumber());
    LOGF(info, "Tracks for collision: %d", tracks.size());
    for (auto& track : tracks) {
      hpt_nocuts->Fill(track.pt());
      hfCYY->Fill(track.cYY());
      hfCZY->Fill(track.cZY());
      hfCZZ->Fill(track.cZZ());
      hfCSnpY->Fill(track.cSnpY());
      hfCSnpZ->Fill(track.cSnpZ());
      hfCSnpSnp->Fill(track.cSnpSnp());
      hfCTglY->Fill(track.cTglY());
      hfCTglZ->Fill(track.cTglZ());
      hfCTglSnp->Fill(track.cTglSnp());
      hfCTglTgl->Fill(track.cTglTgl());
      hfC1PtY->Fill(track.c1PtY());
      hfC1PtZ->Fill(track.c1PtZ());
      hfC1PtSnp->Fill(track.c1PtSnp());
      hfC1PtTgl->Fill(track.c1PtTgl());
      hfC1Pt21Pt2->Fill(track.c1Pt21Pt2());
      LOGF(info, "track tgl  ss   s%f", track.tgl());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ValidationTask>("validation-qa")};
}
