// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \brief Demonstrates various ways to create, manage, and fill histograms.
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "AnalysisCore/HistHelpers.h"

#include <cmath>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::experimental::histhelpers;

struct HistHelpersTest {

  // some unique names for the histograms
  enum HistNamesTest : uint8_t {
    test_1d_TH1D,
    test_2d_TH2F,
    test_3d_TH3I,
    test_1d_TH1D_Weight,
    test_2d_TH2F_VarBinningY,

    test_3d_THnI,
    test_5d_THnSparseI,

    test_1d_TProfile,
    test_1d_TProfile_Weight,
    test_2d_TProfile2D,
    test_3d_TProfile3D,

    test_7d_THnF_first,
    test_7d_THnF_second,
    test_8d_THnC_third,
  };
  enum HistNamesKine : uint8_t {
    pt,
    eta,
    phi,
  };
  enum HistNamesTPC : uint8_t {
    tpcNClsFound,
    tpcNClsCrossedRows,
    tpcChi2NCl,
  };
  enum HistNamesITS : uint8_t {
    itsChi2NCl,
  };

  OutputObj<HistArray> test{HistArray("Test"), OutputObjHandlingPolicy::QAObject};
  OutputObj<HistArray> kine{HistArray("Kine"), OutputObjHandlingPolicy::QAObject};
  OutputObj<HistFolder> tpc{HistFolder("TPC"), OutputObjHandlingPolicy::QAObject};
  OutputObj<HistList> its{HistList("ITS"), OutputObjHandlingPolicy::QAObject};

  OutputObj<THnF> standaloneHist{"standaloneHist", OutputObjHandlingPolicy::QAObject};

  void init(o2::framework::InitContext&)
  {
    // add some plain and simple histograms
    test->Add<test_1d_TH1D>(new TH1D("test_1d_TH1D", ";x", 100, 0., 50.));
    test->Add<test_2d_TH2F>(new TH2F("test_2d_TH2F", ";x;y", 100, -0.5, 0.5, 100, -0.5, 0.5));
    test->Add<test_3d_TH3I>(new TH3I("test_3d_TH3I", ";x;y;z", 100, 0., 20., 100, 0., 20., 100, 0., 20.));

    // alternatively use Hist to generate the histogram and add it to container afterwards
    Hist sameAsBefore;
    sameAsBefore.AddAxis("x", "x", 100, 0., 50.);
    // via Hist::Create() we can generate the actual root histogram with the requested axes
    // Parameters: the name and optionally the decision wether to call SumW2 in case we want to fill this histogram with weights
    test->Add<test_1d_TH1D_Weight>(sameAsBefore.Create<TH1D>("test_1d_TH1D_Weight", true));

    // this helper enables us to have combinations of flexible + fixed binning in 2d or 3d histograms
    // (which are not available via default root constructors)
    Hist sameButDifferent;
    sameButDifferent.AddAxis("x", "x", 100, -0.5, 0.5);
    sameButDifferent.AddAxis("y", "y", {-0.5, -0.48, -0.3, 0.4, 0.5}); // use variable binning for y axsis this time
    test->Add<test_2d_TH2F_VarBinningY>(sameButDifferent.Create<TH2F>("test_2d_TH2F_VarBinningY"));

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

    Hist myHistogram({ptAxis, etaAxis, {"signed1Pt", "q/p_{T}", {-8, 8}, 200}});
    test->Add<test_3d_THnI>(myHistogram.Create<THnI>("test_3d_THnI"));

    Hist testSparseHist({ptAxis, etaAxis, phiAxis, centAxis, cutAxis});
    test->Add<test_5d_THnSparseI>(testSparseHist.Create<THnSparseI>("test_5d_THnSparseI"));

    Hist testProfile({ptAxis});
    test->Add<test_1d_TProfile>(testProfile.Create<TProfile>("test_1d_TProfile"));
    test->Get<TProfile>(test_1d_TProfile)->GetYaxis()->SetTitle("eta profile");

    // now add same histogram but intended for weighted filling
    test->Add<test_1d_TProfile_Weight>(testProfile.Create<TProfile>("test_1d_TProfile_Weight", true));

    Hist testProfile2d;
    testProfile2d.AddAxis(ptAxis);
    testProfile2d.AddAxis(etaAxis);
    test->Add<test_2d_TProfile2D>(testProfile2d.Create<TProfile2D>("test_2d_TProfile2D"));

    Hist testProfile3d;
    testProfile3d.AddAxis(ptAxis);
    testProfile3d.AddAxis(etaAxis);
    testProfile3d.AddAxis(phiAxis);
    test->Add<test_3d_TProfile3D>(testProfile3d.Create<TProfile3D>("test_3d_TProfile3D"));

    // we can also re-use axis definitions in case they are similar in many histograms:
    Hist baseDimensions({ptAxis, etaAxis, phiAxis, centAxis, cutAxis, centAxis});

    Hist firstHist{baseDimensions};
    firstHist.AddAxis("something", "v (m/s)", 10, -1, 1);
    test->Add<test_7d_THnF_first>(firstHist.Create<THnF>("test_7d_THnF_first"));

    Hist secondHist{baseDimensions};
    secondHist.AddAxis("somethingElse", "a (m/(s*s))", 10, -1, 1);
    test->Add<test_7d_THnF_second>(secondHist.Create<THnF>("test_7d_THnF_second"));

    // or if we want to have the baseDimensions somewhere in between:
    Hist thirdHist;
    thirdHist.AddAxis("myFirstDimension", "a (m/(s*s))", 10, -1, 1);
    thirdHist.AddAxes(baseDimensions);
    thirdHist.AddAxis("myLastDimension", "a (m/(s*s))", 10, -1, 1);
    test->Add<test_8d_THnC_third>(thirdHist.Create<THnC>("test_8d_THnC_third"));

    // we can also use the Hist helper tool independent of the HistCollections:
    Hist myHist;
    myHist.AddAxis(ptAxis);
    myHist.AddAxis(etaAxis);
    myHist.AddAxis(phiAxis);
    standaloneHist.setObject(myHist.Create<THnF>("standaloneHist"));

    // now add some more useful histograms
    kine->Add<pt>(new TH1F("pt", "p_{T};p_{T} [GeV/c]", 100, 0., 5.));
    kine->Add<eta>(new TH1F("eta", "#eta;#eta", 101, -1.0, 1.0));
    kine->Add<phi>(new TH1F("phi", "#phi;#phi [rad]", 100, 0., 2 * M_PI));

    tpc->Add<tpcNClsFound>(new TH1F("tpcNClsFound", "number of found TPC clusters;# clusters TPC", 165, -0.5, 164.5));
    tpc->Add<tpcNClsCrossedRows>(new TH1F("tpcNClsCrossedRows", "number of crossed TPC rows;# crossed rows TPC", 165, -0.5, 164.5));
    tpc->Add<tpcChi2NCl>(new TH1F("tpcChi2NCl", "chi2 per cluster in TPC;chi2 / cluster TPC", 100, 0, 10));

    its->Add<itsChi2NCl>(new TH1F("itsChi2NCl", "chi2 per ITS cluster;chi2 / cluster ITS", 100, 0, 40));
  }

  void process(soa::Join<aod::Tracks, aod::TracksExtra>::iterator const& track)
  {
    test->Fill<test_1d_TH1D>(20.);
    test->Fill<test_2d_TH2F>(0.1, 0.3);
    test->Fill<test_3d_TH3I>(10, 10, 15.5);

    // this time fill the 1d histogram with weight of 10:
    test->FillWeight<test_1d_TH1D_Weight>(20., 10.);

    test->Fill<test_2d_TH2F_VarBinningY>(0.1, 0.3);

    test->Fill<test_3d_THnI>(1., 0., 1.5);
    test->Fill<test_5d_THnSparseI>(1., 0., 1.5, 30, 1);

    // we can also directly access to the underlying histogram
    // for this the correct type has to be known (TH1, TH2, TH3, THn, THnSparse, TProfile, TProfile2D or TProfile3D)
    test->Get<TH2>(test_2d_TH2F)->Fill(0.2, 0.2);

    test->Fill<test_1d_TProfile>(track.pt(), track.eta());
    // now fill same histogram, but with random weight
    test->FillWeight<test_1d_TProfile_Weight>(track.pt(), track.eta(), std::rand());
    test->Fill<test_2d_TProfile2D>(track.pt(), track.eta(), track.phi());
    test->Fill<test_3d_TProfile3D>(track.pt(), track.eta(), track.phi(), track.tpcNClsFound());

    kine->Fill<pt>(track.pt());
    kine->Fill<eta>(track.eta());
    kine->Fill<phi>(track.phi());

    tpc->Fill<tpcNClsFound>(track.tpcNClsFound());
    tpc->Fill<tpcNClsCrossedRows>(track.tpcNClsCrossedRows());
    tpc->Fill<tpcChi2NCl>(track.tpcChi2NCl());

    its->Fill<itsChi2NCl>(track.itsChi2NCl());

    double dummyArray[] = {track.pt(), track.eta(), track.phi()};
    standaloneHist->Fill(dummyArray);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<HistHelpersTest>(cfgc),
  };
}
