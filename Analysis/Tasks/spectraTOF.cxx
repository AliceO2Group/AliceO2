// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// O2 includes
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "ReconstructionDataFormats/Track.h"
#include "PID/PIDResponse.h"
#include "Framework/ASoAHelpers.h"

// ROOT includes
#include <TH1F.h>

#define DOTH1F(OBJ, ...) \
  OutputObj<TH1F> OBJ{TH1F(#OBJ, __VA_ARGS__)};
#define DOTH2F(OBJ, ...) \
  OutputObj<TH2F> OBJ{TH2F(#OBJ, __VA_ARGS__)};

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct TOFPIDQATask {
  DOTH1F(hp_NoCut, ";#it{p} (GeV/#it{c});Tracks", 100, 0, 20);
  DOTH1F(hp_TrkCut, ";#it{p} (GeV/#it{c});Tracks", 100, 0, 20);
  DOTH1F(hp_TOFCut, ";#it{p} (GeV/#it{c});Tracks", 100, 0, 20);
  //
  DOTH1F(hlength_NoCut, ";Track Length (cm);Tracks", 100, 0, 1000);
  DOTH1F(htime_NoCut, ";TOF Time (ns);Tracks", 1000, 0, 600);
  DOTH1F(hevtime_NoCut, ";Event time (ns);Tracks", 100, -2, 2);
  //
  DOTH2F(hnsigmaPi_NoCut, ";#it{p} (GeV/#it{c});Tracks", 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaKa_NoCut, ";#it{p} (GeV/#it{c});Tracks", 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaPr_NoCut, ";#it{p} (GeV/#it{c});Tracks", 100, 0, 5, 100, -10, 10);
  //
  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTOF> const& tracks)
  {
    for (auto i : tracks) {
      float Mom = p(i.eta(), i.signed1Pt());
      hp_NoCut->Fill(Mom);
      // Track selection
      UChar_t clustermap = i.itsClusterMap();
      bool issel = (i.tpcNClsFindable() > 70) && (i.flags() & 0x4) && (TESTBIT(clustermap, 0) || TESTBIT(clustermap, 1));
      if (issel)
        hp_TrkCut->Fill(Mom);
      issel = issel && (i.flags() & 0x2000);     //kTOFout
      issel = issel && (i.flags() & 0x80000000); //kTIME
      if (issel)
        hp_TOFCut->Fill(Mom);
      //
      hlength_NoCut->Fill(i.length());
      htime_NoCut->Fill(i.tofSignal() / 1000);
      //
      hevtime_NoCut->Fill(collision.collisionTime() / 1000);
      // hevtime_NoCut->Fill(collision.collisionTime0() / 1000);
      //
      hnsigmaPi_NoCut->Fill(i.p(), i.nsigmaPi());
      hnsigmaKa_NoCut->Fill(i.p(), i.nsigmaKa());
      hnsigmaPr_NoCut->Fill(i.p(), i.nsigmaPr());
    }
  }
};

struct ElectronSpectraTask {
  //
  OutputObj<TH1F> hp_El{TH1F("hp_El", ";#it{p} (GeV/#it{c});Tracks", 100, 0, 20)};
  OutputObj<TH1F> hpt_El{TH1F("hpt_El", ";#it{p}_{T} (GeV/#it{c});Tracks", 100, 0, 20)};
  //
  OutputObj<TH1F> hlength_El{TH1F("hlength_El", ";Track Length (cm);Tracks", 100, 0, 1000)};
  OutputObj<TH1F> htime_El{TH1F("htime_El", ";TOF Time (ns);Tracks", 1000, 0, 600)};
  // OutputObj<TH1F> hevtime_El{TH1F("hevtime_El", ";Event time (ns);Tracks", 100, -2, 2)};
  //
  OutputObj<TH2F> hp_beta{TH2F("hp_beta", ";#it{p} (GeV/#it{c});TOF #beta;Tracks", 100, 0, 20, 100, 0, 2)};
  OutputObj<TH2F> hp_beta_El{TH2F("hp_beta_El", ";#it{p} (GeV/#it{c});#beta - #beta_{el};Tracks", 100, 0, 20, 100, -0.01, 0.01)};
  OutputObj<TH2F> hp_betasigma_El{TH2F("hp_betasigma_El", ";#it{p} (GeV/#it{c});(#beta - #beta_{el})/#sigma;Tracks", 100, 0, 20, 100, -5, 5)};

  // Filter trk_filter = (aod::track::tpcNClsFindable > 70);

  // void process(soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTOF>> const& tracks)
  void process(soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTOF> const& tracks)
  {
    for (auto i : tracks) {
      UChar_t clustermap = i.itsClusterMap();
      bool issel = (i.tpcNClsFindable() > 70) && (i.flags() & 0x4) && (TESTBIT(clustermap, 0) || TESTBIT(clustermap, 1));
      issel = issel && (i.flags() & 0x2000);     //kTOFout
      issel = issel && (i.flags() & 0x80000000); //kTIME
      if (!issel)
        continue;
      hp_El->Fill(i.p());
      hpt_El->Fill(i.pt());
      //
      hlength_El->Fill(i.length());
      htime_El->Fill(i.tofSignal() / 1000);
      // hevtime_El->Fill(collision.eventTime() / 1000);
      //
      hp_beta->Fill(i.p(), i.beta());
      hp_beta_El->Fill(i.p(), i.diffbetael());
      hp_betasigma_El->Fill(i.p(), i.separationbetael());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<pidTOFTask>("pidTOF-task"),
    adaptAnalysisTask<TOFPIDQATask>("tofpidqa-task"),
    adaptAnalysisTask<ElectronSpectraTask>("filterEl-task")};
}
