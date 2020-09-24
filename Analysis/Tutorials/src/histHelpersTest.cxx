// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/AnalysisTask.h"
#include "Analysis/HistHelpers.h"

#include "Framework/AnalysisDataModel.h"
#include "Framework/runDataProcessing.h"
#include <cmath>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::experimental::histhelpers;

struct HistHelpersTest {

  // some unique names for the histograms
  enum HistNames : uint8_t {
    Hist_pt = 0,
    Hist_eta,
    Hist_phi,
    Hist_nClsTPC,
    Hist_nCrossedRowsTPC,
    Hist_Chi2PerClTPC,
    Hist_nClsITS,
    Hist_Chi2PerClITS,

    // for testing
    Hist_test_1d_TH1D,
    Hist_test_2d_TH2F,
    Hist_test_3d_TH3I,
    Hist_test_1d_TH1D_Builder,
    Hist_test_2d_TH2F_Builder,

    Hist_test_3d_THnI,
    Hist_test_5d_THnSparseI,

    Hist_test_1d_TProfile,
    Hist_test_2d_TProfile2D,
    Hist_test_3d_TProfile3D,

    Hist_test_7d_THnF_first,
    Hist_test_7d_THnF_second,
    Hist_test_8d_THnC_third,
    LAST,
  };

  OutputObj<HistArray> test{HistArray("Test"), OutputObjHandlingPolicy::QAObject};
  OutputObj<HistArray> kine{HistArray("Kine"), OutputObjHandlingPolicy::QAObject};
  OutputObj<HistFolder> tpc{HistFolder("TPC"), OutputObjHandlingPolicy::QAObject};
  OutputObj<HistList> its{HistList("ITS"), OutputObjHandlingPolicy::QAObject};

  OutputObj<THnF> standaloneHist{"standaloneHist", OutputObjHandlingPolicy::QAObject};

  void init(o2::framework::InitContext&)
  {
    // add some plain and simple histograms
    test->Add(Hist_test_1d_TH1D, TH1D("testHist_TH1", ";x", 100, 0., 50.));
    test->Add(Hist_test_2d_TH2F, new TH2F("testHist_TH2", ";x;y", 100, -0.5, 0.5, 100, -0.5, 0.5));
    test->Add(Hist_test_3d_TH3I, TH3I("testHist_TH3", ";x;y;z", 100, 0., 20., 100, 0., 20., 100, 0., 20.));

    // alternatively use Hist to generate the histogram and add it to container afterwards
    Hist<TH1D> sameAsBefore;
    sameAsBefore.AddAxis("x", "x", 100, 0., 50.);
    // via Hist::Create() we can generate the actual root histogram with the requested axes
    // Parameters: the name and optionally the decision wether to call SumW2 in case we want to fill this histogram with weights
    test->Add(Hist_test_1d_TH1D_Builder, sameAsBefore.Create("testHist_TH1_Builder", true));

    // this helper enables us to have combinations of flexible + fixed binning in 2d or 3d histograms
    // (which are not available via default root constructors)
    Hist<TH2F> sameButDifferent;
    sameButDifferent.AddAxis("x", "x", 100, -0.5, 0.5);
    sameButDifferent.AddAxis("y", "y", {-0.5, -0.48, -0.3, 0.4, 0.5}); // use variable binning for y axsis this time
    test->Add(Hist_test_2d_TH2F_Builder, sameButDifferent.Create("testHist_TH2_Builder"));

    // also for n dimensional histograms things become much simpler:
    std::vector<double> ptBins = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                  1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 5.0, 10.0, 20.0, 50.0};
    std::vector<double> centBins = {0., 30., 60., 90.};

    // varaiable binning
    Axis ptAxis = {"pt", "#it{p}_{T} (GeV/c)", ptBins};
    Axis centAxis = {"cent", "centrality", centBins};
    // equidistant binning
    Axis etaAxis = {"eta", "#eta", {-0.8, 0.8}, 5};
    Axis phiAxis = {"phi", "#phi", {0., 2. * M_PI}, 4}; // 36 to see tpc sectors
    const int nCuts = 5;
    Axis cutAxis = {"cut", "cut setting", {-0.5, nCuts - 0.5}, nCuts};

    Hist<THnI> myHistogram;
    myHistogram.AddAxis(ptAxis);
    myHistogram.AddAxis(etaAxis);
    myHistogram.AddAxis("signed1Pt", "q/p_{T}", 200, -8, 8);
    test->Add(Hist_test_3d_THnI, myHistogram.Create("testHist_THn"));

    Hist<THnSparseC> testSparseHist;
    testSparseHist.AddAxis(ptAxis);
    testSparseHist.AddAxis(etaAxis);
    testSparseHist.AddAxis(phiAxis);
    testSparseHist.AddAxis(centAxis);
    testSparseHist.AddAxis(cutAxis);
    test->Add(Hist_test_5d_THnSparseI, testSparseHist.Create("testHist_THnSparse"));

    Hist<TProfile> testProfile;
    testProfile.AddAxis(ptAxis);
    test->Add(Hist_test_1d_TProfile, testProfile.Create("testProfile_TProfile"));
    test->Get<TProfile>(Hist_test_1d_TProfile)->GetYaxis()->SetTitle("eta profile");

    Hist<TProfile2D> testProfile2d;
    testProfile2d.AddAxis(ptAxis);
    testProfile2d.AddAxis(etaAxis);
    test->Add(Hist_test_2d_TProfile2D, testProfile2d.Create("testProfile_TProfile2D"));

    Hist<TProfile3D> testProfile3d;
    testProfile3d.AddAxis(ptAxis);
    testProfile3d.AddAxis(etaAxis);
    testProfile3d.AddAxis(phiAxis);
    test->Add(Hist_test_3d_TProfile3D, testProfile3d.Create("testProfile_TProfile3D"));

    // now add some more useful histograms
    kine->Add(Hist_pt, TH1F("track-pt", "p_{T};p_{T} [GeV/c]", 100, 0., 50.));
    kine->Add(Hist_eta, TH1F("track-eta", "#eta;#eta", 101, -1.0, 1.0));
    kine->Add(Hist_phi, TH1F("track-phi", "#phi;#phi [rad]", 100, 0., 2 * M_PI));

    tpc->Add(Hist_nCrossedRowsTPC, TH1F("tpc-crossedRows", "number of crossed TPC rows;# crossed rows TPC", 165, -0.5, 164.5));
    tpc->Add(Hist_nClsTPC, TH1F("tpc-foundClusters", "number of found TPC clusters;# clusters TPC", 165, -0.5, 164.5));
    tpc->Add(Hist_Chi2PerClTPC, TH1F("tpc-chi2PerCluster", "chi2 per cluster in TPC;chi2 / cluster TPC", 100, 0, 10));

    its->Add(Hist_Chi2PerClITS, TH1F("its-chi2PerCluster", "chi2 per ITS cluster;chi2 / cluster ITS", 100, 0, 40));

    // we can also re-use axis definitions in case they are similar in many histograms:
    Hist<THnF> baseDimensions;
    baseDimensions.AddAxis(ptAxis);
    baseDimensions.AddAxis(etaAxis);
    baseDimensions.AddAxis(phiAxis);
    baseDimensions.AddAxis(centAxis);
    baseDimensions.AddAxis(cutAxis);
    baseDimensions.AddAxis(centAxis);

    Hist<THnF> firstHist = baseDimensions;
    firstHist.AddAxis("something", "v (m/s)", 10, -1, 1);
    its->Add(Hist_test_7d_THnF_first, firstHist.Create("firstHist"));

    Hist<THnF> secondHist = baseDimensions;
    secondHist.AddAxis("somethingElse", "a (m/(s*s))", 10, -1, 1);
    its->Add(Hist_test_7d_THnF_second, secondHist.Create("secondHist"));

    // or if we want to have the baseDimensions somewhere in between:
    Hist<THnC> thirdHist;
    thirdHist.AddAxis("myFirstDimension", "a (m/(s*s))", 10, -1, 1);
    thirdHist.AddAxes(baseDimensions);
    thirdHist.AddAxis("myLastDimension", "a (m/(s*s))", 10, -1, 1);
    its->Add(Hist_test_8d_THnC_third, thirdHist.Create("thirdHist"));

    // we can also use the Hist heleper tool independent of the HistCollections:
    Hist<THnF> myHist;
    myHist.AddAxis(ptAxis);
    myHist.AddAxis(etaAxis);
    myHist.AddAxis(phiAxis);
    standaloneHist.setObject(myHist.Create("standaloneHist"));
  }

  void process(soa::Join<aod::Tracks, aod::TracksExtra>::iterator const& track)
  {
    test->Fill(Hist_test_1d_TH1D, 1.);
    test->Fill(Hist_test_2d_TH2F, 0.1, 0.3);
    test->Fill(Hist_test_3d_TH3I, 10, 10, 15);

    // this time fill the 1d histogram with weight of 10:
    test->FillWeight(Hist_test_1d_TH1D_Builder, 1., 10.);
    test->Fill(Hist_test_2d_TH2F_Builder, 0.1, 0.3);

    test->Fill(Hist_test_3d_THnI, 1., 0., 1.5);
    test->Fill(Hist_test_5d_THnSparseI, 1., 0., 1.5, 30, 1);

    // we can also directly access to the underlying histogram
    // for this the correct type has to be known (TH1, TH2, TH3, THn, THnSparse, TProfile, TProfile2D or TProfile3D)
    test->Get<TH2>(Hist_test_2d_TH2F)->Fill(0.2, 0.2);

    test->Fill(Hist_test_1d_TProfile, track.pt(), track.eta());
    test->Fill(Hist_test_2d_TProfile2D, track.pt(), track.eta(), track.phi());
    test->Fill(Hist_test_3d_TProfile3D, track.pt(), track.eta(), track.phi(), track.tpcNClsFound());

    kine->Fill(Hist_pt, track.pt());
    kine->Fill(Hist_eta, track.eta());
    kine->Fill(Hist_phi, track.phi());

    tpc->Fill(Hist_nClsTPC, track.tpcNClsFound());
    tpc->Fill(Hist_nCrossedRowsTPC, track.tpcNClsCrossedRows());
    tpc->Fill(Hist_Chi2PerClTPC, track.itsChi2NCl());

    its->Fill(Hist_Chi2PerClITS, track.tpcChi2NCl());

    double dummyArray[] = {track.pt(), track.eta(), track.phi()};
    standaloneHist->Fill(dummyArray);
  }
};

//****************************************************************************************
/**
 * Workflow definition.
 */
//****************************************************************************************
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<HistHelpersTest>("hist-helpers-test")};
}
