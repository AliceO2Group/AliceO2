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

#include "CompareTask.h"
#include "CompareTracks.h"
#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InitContext.h"
#include "Framework/InputRecord.h"
#include "Framework/ProcessingContext.h"
#include "Histos.h"
#include "MCHEvaluation/Draw.h"
#include "MCHEvaluation/ExtendedTrack.h"
#include "MCHTracking/TrackExtrap.h"
#include <TCanvas.h>
#include <TFile.h>
#include <TH1.h>
#include <TString.h>

using namespace o2::framework;

namespace o2::mch::eval
{

CompareTask::CompareTask(std::shared_ptr<o2::base::GRPGeomRequest> req) : mCcdbRequest(req) {}

void CompareTask::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (mCcdbRequest && base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    if (matcher == framework::ConcreteDataMatcher("GLO", "GRPMAGFIELD", 0)) {
      TrackExtrap::setField();
    }
  }
}

void outputToPdfClearDivide(const std::string fileName,
                            TCanvas& c,
                            int nPadsx, int nPadsy)
{
  c.Print(fileName.c_str());
  c.Clear();
  c.Divide(nPadsx, nPadsy);
}

void CompareTask::pdfOutput()
{
  int nPadsx, nPadsy;
  TCanvas* c = autoCanvas("c", "c", mHistosAtVertex[0], &nPadsx, &nPadsy);
  TCanvas* c2 = autoCanvas("c2", "c2", mHistosAtVertex[0], &nPadsx, &nPadsy);
  c->Print(fmt::format("{}[", mOutputPdfFileName).c_str());

  auto toPdf2 = [&](std::function<void(const std::array<std::vector<TH1*>, 2>&, TCanvas*)> func) {
    func(mHistosAtVertex, c);
    outputToPdfClearDivide(mOutputPdfFileName, *c, nPadsx, nPadsy);
  };

  auto toPdf5 = [&](std::function<void(const std::array<std::vector<TH1*>, 5>&, TCanvas*)> func) {
    func(mComparisonsAtVertex, c);
    outputToPdfClearDivide(mOutputPdfFileName, *c, nPadsx, nPadsy);
  };

  toPdf2(drawPlainHistosAtVertex);
  toPdf2(drawDiffHistosAtVertex);
  toPdf2(drawRatioHistosAtVertex);

  toPdf5(drawComparisonsAtVertex);

  c->Clear();
  drawTrackResiduals(mTrackResidualsAtFirstCluster, c);
  c->Print(mOutputPdfFileName.c_str());

  c->Clear();
  drawClusterClusterResiduals(mClusterResiduals[4], "ClCl", c);
  c->Print(mOutputPdfFileName.c_str());

  c->Clear();
  drawClusterTrackResiduals(mClusterResiduals[0], mClusterResiduals[1], "AllTracks", c);
  c->Print(mOutputPdfFileName.c_str());

  c->Clear();
  c2->Clear();
  drawClusterTrackResidualsSigma(mClusterResiduals[0], mClusterResiduals[1], "AllTracks", c, c2);
  c->Print(mOutputPdfFileName.c_str());
  c2->Print(mOutputPdfFileName.c_str());

  c->Clear();
  drawClusterTrackResiduals(mClusterResiduals[2], mClusterResiduals[3], "SimilarTracks", c);
  c->Print(mOutputPdfFileName.c_str());

  c->Clear();
  c2->Clear();
  drawClusterTrackResidualsSigma(mClusterResiduals[2], mClusterResiduals[3], "SimilarTracks", c, c2);
  c->Print(mOutputPdfFileName.c_str());
  c2->Print(mOutputPdfFileName.c_str());

  c->Clear();
  drawClusterTrackResidualsRatio(mClusterResiduals[0], mClusterResiduals[1], "AllTracks", c);
  c->Print(mOutputPdfFileName.c_str());

  c->Clear();
  drawClusterTrackResidualsRatio(mClusterResiduals[2], mClusterResiduals[3], "SimilarTracks", c);
  c->Print(mOutputPdfFileName.c_str());

  c->Clear();

  c->Print(fmt::format("{}]", mOutputPdfFileName).c_str());
}

void CompareTask::init(InitContext& ic)
{
  auto outputFileName = ic.options().get<std::string>("outfile");
  mOutputPdfFileName = ic.options().get<std::string>("pdf-outfile");
  mPrintDiff = ic.options().get<bool>("print-diff");
  mPrintAll = ic.options().get<bool>("print-all");
  mApplyTrackSelection = ic.options().get<bool>("apply-track-selection");
  mPrecision = ic.options().get<double>("precision");
  o2::base::GRPGeomHelper::instance().setRequest(mCcdbRequest);

  mOutputRootFile = std::make_unique<TFile>(outputFileName.c_str(), "RECREATE");

  createHistosAtVertex(mHistosAtVertex[0], "1");
  createHistosAtVertex(mHistosAtVertex[1], "2");
  createHistosAtVertex(mComparisonsAtVertex[0], "identical");
  createHistosAtVertex(mComparisonsAtVertex[1], "similar1");
  createHistosAtVertex(mComparisonsAtVertex[2], "similar2");
  createHistosAtVertex(mComparisonsAtVertex[3], "additional");
  createHistosAtVertex(mComparisonsAtVertex[4], "missing");
  createHistosForClusterResiduals(mClusterResiduals[0], "AllTracks1", 2.);
  createHistosForClusterResiduals(mClusterResiduals[1], "AllTracks2", 2.);
  createHistosForClusterResiduals(mClusterResiduals[2], "SimilarTracks1", 2.);
  createHistosForClusterResiduals(mClusterResiduals[3], "SimilarTracks2", 2.);
  createHistosForClusterResiduals(mClusterResiduals[4], "ClCl", 0.2);
  createHistosForTrackResiduals(mTrackResidualsAtFirstCluster);

  mNofDifferences = 0;

  auto stop = [this]() {
    LOGP(warning, "Number of differences in ROF-by-ROF comparison: {}", mNofDifferences);
    printStat();
    if (!mOutputPdfFileName.empty()) {
      pdfOutput();
    }
    mOutputRootFile->cd();
    mOutputRootFile->WriteObject(&mHistosAtVertex, "histosAtVertex");
    mOutputRootFile->WriteObject(&mComparisonsAtVertex, "comparisonsAtVertex");
    mOutputRootFile->WriteObject(&mTrackResidualsAtFirstCluster, "trackResidualsAtFirstCluster");
    mOutputRootFile->WriteObject(&mClusterResiduals, "clusterResiduals");
    mOutputRootFile->Close();
  };
  ic.services().get<CallbackService>().set<CallbackService::Id::Stop>(stop);
}

std::list<ExtendedTrack> CompareTask::convert(gsl::span<const TrackMCH> mchTracks,
                                              gsl::span<const Cluster> clusters)
{
  std::list<ExtendedTrack> tracks;
  constexpr double vx{0.0};
  constexpr double vy{0.0};
  constexpr double vz{0.0};
  for (const auto& mchTrack : mchTracks) {
    tracks.emplace_back(mchTrack, clusters, vx, vy, vz);
  }
  return tracks;
}

void CompareTask::dump(std::string prefix,
                       const std::list<ExtendedTrack>& tracks1,
                       const std::list<ExtendedTrack>& tracks2)
{
  if (tracks1.size() > 0 || tracks2.size() > 0) {
    LOGP(warning, "{} tracks1: {} tracks2: {}", prefix, tracks1.size(), tracks2.size());
    for (const auto& t : tracks1) {
      LOGP(warning, "Track1 {}", t.asString());
    }
    for (const auto& t : tracks2) {
      LOGP(warning, "Track2 {}", t.asString());
    }
  }
}

std::list<ExtendedTrack> CompareTask::getExtendedTracks(const ROFRecord& rof,
                                                        gsl::span<const TrackMCH> tfTracks,
                                                        gsl::span<const Cluster> tfClusters)
{
  const auto mchTracks = tfTracks.subspan(rof.getFirstIdx(), rof.getNEntries());
  return convert(mchTracks, tfClusters);
}

void CompareTask::run(ProcessingContext& pc)
{
  static int tf{0};
  LOGP(info, "TF {}", tf);

  if (mCcdbRequest) {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  }

  auto rofs1 = pc.inputs().get<gsl::span<ROFRecord>>("rofs1");
  auto itracks1 = pc.inputs().get<gsl::span<TrackMCH>>("tracks1");
  auto iclusters1 = pc.inputs().get<gsl::span<Cluster>>("clusters1");
  auto rofs2 = pc.inputs().get<gsl::span<ROFRecord>>("rofs2");
  auto itracks2 = pc.inputs().get<gsl::span<TrackMCH>>("tracks2");
  auto iclusters2 = pc.inputs().get<gsl::span<Cluster>>("clusters2");

  bool areSameRofs = std::equal(rofs1.begin(), rofs1.end(),
                                rofs2.begin(), rofs2.end(),
                                [](const ROFRecord& r1, const ROFRecord& r2) {
                                  return r1.getBCData() == r2.getBCData() && r1.getBCWidth() == r2.getBCWidth();
                                }); // just compare BC id, orbit and BC width, not number of found tracks
  if (!areSameRofs) {
    LOGP(warning, "ROFs are different --> cannot perform ROF-by-ROF comparison");
  }

  auto nROFs = std::max(rofs1.size(), rofs2.size());
  for (auto i = 0; i < nROFs; ++i) {

    std::list<ExtendedTrack> tracks1{};
    if (i < rofs1.size()) {
      tracks1 = getExtendedTracks(rofs1[i], itracks1, iclusters1);
      mNTracksAll[0] += tracks1.size();
      if (mApplyTrackSelection) {
        selectTracks(tracks1);
      }
      fillHistosAtVertex(tracks1, mHistosAtVertex[0]);
      fillClusterTrackResiduals(tracks1, mClusterResiduals[0], false);
    }

    std::list<ExtendedTrack> tracks2{};
    if (i < rofs2.size()) {
      tracks2 = getExtendedTracks(rofs2[i], itracks2, iclusters2);
      mNTracksAll[1] += tracks2.size();
      if (mApplyTrackSelection) {
        selectTracks(tracks2);
      }
      fillHistosAtVertex(tracks2, mHistosAtVertex[1]);
      fillClusterTrackResiduals(tracks2, mClusterResiduals[1], false);
    }

    if (areSameRofs) {
      int nDiff = compareEvents(tracks1, tracks2,
                                mPrecision,
                                mPrintDiff,
                                mPrintAll,
                                mTrackResidualsAtFirstCluster,
                                mClusterResiduals[4]);
      fillClusterTrackResiduals(tracks1, mClusterResiduals[2], true);
      fillClusterTrackResiduals(tracks2, mClusterResiduals[3], true);
      if (nDiff > 0) {
        if (mPrintDiff) {
          LOG(warning) << "--> " << nDiff << " differences found in ROF " << rofs1[i];
        }
        mNofDifferences += nDiff;
      }
      fillComparisonsAtVertex(tracks1, tracks2, mComparisonsAtVertex);
    }
  }

  ++tf;
}

void CompareTask::printStat()
{
  /// print some statistics and the relative difference (in %) between the 2 inputs
  /// the uncertainty on the difference is the normal approximation of the binomial error

  auto print = [](TString selection, int n1, int n2) {
    if (n1 == 0) {
      printf("%s | %8d | %8d | %7s ± %4s %%\n", selection.Data(), n1, n2, "nan", "nan");
    } else {
      double eff = double(n2) / n1;
      double diff = 100. * (eff - 1.);
      double err = 100. * std::max(1. / n1, std::sqrt(eff * std::abs(1. - eff) / n1));
      printf("%s | %8d | %8d | %7.2f ± %4.2f %%\n", selection.Data(), n1, n2, diff, err);
    }
  };

  printf("\n");
  printf("-------------------------------------------------------\n");
  printf("selection      |  file 1  |  file 2  |       diff\n");
  printf("-------------------------------------------------------\n");

  print("all           ", mNTracksAll[0], mNTracksAll[1]);

  print("matched       ", mNTracksMatch[0], mNTracksMatch[1]);

  print("selected      ", mHistosAtVertex[0][0]->GetEntries(), mHistosAtVertex[1][0]->GetEntries());

  double pTRange[6] = {0., 0.5, 1., 2., 4., 1000.};
  for (int i = 0; i < 5; ++i) {
    TString selection = (i == 0) ? TString::Format("pT < %.1f GeV/c", pTRange[1])
                                 : ((i == 4) ? TString::Format("pT > %.1f      ", pTRange[4])
                                             : TString::Format("%.1f < pT < %.1f", pTRange[i], pTRange[i + 1]));
    int n1 = mHistosAtVertex[0][0]->Integral(mHistosAtVertex[0][0]->GetXaxis()->FindBin(pTRange[i] + 0.01),
                                             mHistosAtVertex[0][0]->GetXaxis()->FindBin(pTRange[i + 1] - 0.01));
    int n2 = mHistosAtVertex[1][0]->Integral(mHistosAtVertex[1][0]->GetXaxis()->FindBin(pTRange[i] + 0.01),
                                             mHistosAtVertex[1][0]->GetXaxis()->FindBin(pTRange[i + 1] - 0.01));
    print(selection, n1, n2);
  }

  double pRange[6] = {0., 5., 10., 20., 40., 10000.};
  for (int i = 0; i < 5; ++i) {
    TString selection = (i == 0) ? TString::Format("p < %02.0f GeV/c  ", pRange[1])
                                 : ((i == 4) ? TString::Format("p > %02.0f        ", pRange[4])
                                             : TString::Format("%02.0f < p < %02.0f   ", pRange[i], pRange[i + 1]));
    int n1 = mHistosAtVertex[0][4]->Integral(mHistosAtVertex[0][4]->GetXaxis()->FindBin(pRange[i] + 0.01),
                                             mHistosAtVertex[0][4]->GetXaxis()->FindBin(pRange[i + 1] - 0.01));
    int n2 = mHistosAtVertex[1][4]->Integral(mHistosAtVertex[1][4]->GetXaxis()->FindBin(pRange[i] + 0.01),
                                             mHistosAtVertex[1][4]->GetXaxis()->FindBin(pRange[i + 1] - 0.01));
    print(selection, n1, n2);
  }

  printf("-------------------------------------------------------\n");
  printf("\n");
}
} // namespace o2::mch::eval
