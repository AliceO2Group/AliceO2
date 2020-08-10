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

struct TPCPIDQAExpSignalTask {
#define BIN_AXIS 100, 0, 5, 1000, 0, 1000

  DOTH2F(htpcsignal, ";#it{p} (GeV/#it{c});TPC Signal;Tracks", BIN_AXIS);
  DOTH2F(hexpEl, ";#it{p} (GeV/#it{c});TPC expected signal e;Tracks", BIN_AXIS);
  DOTH2F(hexpMu, ";#it{p} (GeV/#it{c});TPC expected signal #mu;Tracks", BIN_AXIS);
  DOTH2F(hexpPi, ";#it{p} (GeV/#it{c});TPC expected signal #pi;Tracks", BIN_AXIS);
  DOTH2F(hexpKa, ";#it{p} (GeV/#it{c});TPC expected signal K;Tracks", BIN_AXIS);
  DOTH2F(hexpPr, ";#it{p} (GeV/#it{c});TPC expected signal p;Tracks", BIN_AXIS);
  DOTH2F(hexpDe, ";#it{p} (GeV/#it{c});TPC expected signal d;Tracks", BIN_AXIS);
  DOTH2F(hexpTr, ";#it{p} (GeV/#it{c});TPC expected signal t;Tracks", BIN_AXIS);
  DOTH2F(hexpHe, ";#it{p} (GeV/#it{c});TPC expected signal ^{3}He;Tracks", BIN_AXIS);
  DOTH2F(hexpAl, ";#it{p} (GeV/#it{c});TPC expected signal #alpha;Tracks", BIN_AXIS);

#undef BIN_AXIS

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTPC> const& tracks)
  {
    for (auto const& i : tracks) {
      // Track selection
      const UChar_t clustermap = i.itsClusterMap();
      bool issel = (i.tpcNClsFindable() > 70);
      issel = issel && (i.flags() & 0x4);
      issel = issel && (TESTBIT(clustermap, 0) || TESTBIT(clustermap, 1));
      if (!issel)
        continue;
      //
      htpcsignal->Fill(i.p(), i.tpcSignal());
      hexpEl->Fill(i.p(), i.tpcExpSignalEl());
      hexpMu->Fill(i.p(), i.tpcExpSignalMu());
      hexpPi->Fill(i.p(), i.tpcExpSignalPi());
      hexpKa->Fill(i.p(), i.tpcExpSignalKa());
      hexpPr->Fill(i.p(), i.tpcExpSignalPr());
      hexpDe->Fill(i.p(), i.tpcExpSignalDe());
      hexpTr->Fill(i.p(), i.tpcExpSignalTr());
      hexpHe->Fill(i.p(), i.tpcExpSignalHe());
      hexpAl->Fill(i.p(), i.tpcExpSignalAl());
    }
  }
};

struct TPCPIDQANSigmaTask {
  // TPC NSigma
  DOTH2F(hnsigmaEl, ";#it{p} (GeV/#it{c});TPC N_{sigma e};Tracks", 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaMu, ";#it{p} (GeV/#it{c});TPC N_{sigma #mu};Tracks", 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaPi, ";#it{p} (GeV/#it{c});TPC N_{sigma #pi};Tracks", 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaKa, ";#it{p} (GeV/#it{c});TPC N_{sigma K};Tracks", 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaPr, ";#it{p} (GeV/#it{c});TPC N_{sigma p};Tracks", 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaDe, ";#it{p} (GeV/#it{c});TPC N_{sigma d};Tracks", 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaTr, ";#it{p} (GeV/#it{c});TPC N_{sigma t};Tracks", 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaHe, ";#it{p} (GeV/#it{c});TPC N_{sigma ^{3}He};Tracks", 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaAl, ";#it{p} (GeV/#it{c});TPC N_{sigma #alpha};Tracks", 100, 0, 5, 100, -10, 10);

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTPC> const& tracks)
  {
    for (auto const& i : tracks) {
      // Track selection
      const UChar_t clustermap = i.itsClusterMap();
      bool issel = (i.tpcNClsFindable() > 70);
      issel = issel && (i.flags() & 0x4);
      issel = issel && (TESTBIT(clustermap, 0) || TESTBIT(clustermap, 1));
      if (!issel)
        continue;
      //
      hnsigmaEl->Fill(i.p(), i.tpcNSigmaEl());
      hnsigmaMu->Fill(i.p(), i.tpcNSigmaMu());
      hnsigmaPi->Fill(i.p(), i.tpcNSigmaPi());
      hnsigmaKa->Fill(i.p(), i.tpcNSigmaKa());
      hnsigmaPr->Fill(i.p(), i.tpcNSigmaPr());
      hnsigmaDe->Fill(i.p(), i.tpcNSigmaDe());
      hnsigmaTr->Fill(i.p(), i.tpcNSigmaTr());
      hnsigmaHe->Fill(i.p(), i.tpcNSigmaHe());
      hnsigmaAl->Fill(i.p(), i.tpcNSigmaAl());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TPCPIDQAExpSignalTask>("TPCpidqa-expsignal-task"),
    adaptAnalysisTask<TPCPIDQANSigmaTask>("TPCpidqa-nsigma-task")};
}
