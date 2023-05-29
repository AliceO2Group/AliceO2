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
  drawClusterTrackResidualsSigma(mClusterResiduals[0], mClusterResiduals[1], "AllTracks", c);
  c->Print(mOutputPdfFileName.c_str());

  c->Clear();
  drawClusterTrackResiduals(mClusterResiduals[2], mClusterResiduals[3], "SimilarTracks", c);
  c->Print(mOutputPdfFileName.c_str());

  c->Clear();
  drawClusterTrackResidualsSigma(mClusterResiduals[2], mClusterResiduals[3], "SimilarTracks", c);
  c->Print(mOutputPdfFileName.c_str());

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
    LOGP(info, "Number of differences: {}", mNofDifferences);
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
                                [](const ROFRecord& r1, const ROFRecord& r2) { return r1.getBCData() == r2.getBCData(); }); // just compare BCid and orbit, not number of found tracks
  if (!areSameRofs) {
    LOGP(fatal, "Can only work with identical ROFs {} vs {}",
         rofs1.size(), rofs2.size());
  }

  LOGP(warning, "TF {}", tf);

  // fill the internal track structure based on the MCH tracks
  for (auto i = 0; i < rofs1.size(); i++) {
    auto tracks1 = getExtendedTracks(rofs1[i], itracks1, iclusters1);
    auto tracks2 = getExtendedTracks(rofs2[i], itracks2, iclusters2);
    //  if requested should select tracks here
    if (mApplyTrackSelection) {
      selectTracks(tracks1);
      selectTracks(tracks2);
    }

    fillHistosAtVertex(tracks1, mHistosAtVertex[0]);
    fillHistosAtVertex(tracks2, mHistosAtVertex[1]);

    fillClusterTrackResiduals(tracks1, mClusterResiduals[0], false);
    fillClusterTrackResiduals(tracks2, mClusterResiduals[1], false);

    int nDiff = compareEvents(tracks1, tracks2,
                              mPrecision,
                              mPrintAll,
                              mTrackResidualsAtFirstCluster,
                              mClusterResiduals[4]);
    fillClusterTrackResiduals(tracks1, mClusterResiduals[2], true);
    fillClusterTrackResiduals(tracks2, mClusterResiduals[3], true);
    if (nDiff > 0) {
      LOG(warning) << "--> " << nDiff << " differences found in ROF " << rofs1[i];
      mNofDifferences += nDiff;
    }
    fillComparisonsAtVertex(tracks1, tracks2, mComparisonsAtVertex);
  }
  ++tf;
}
} // namespace o2::mch::eval
