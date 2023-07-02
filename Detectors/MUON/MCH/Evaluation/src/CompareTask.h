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

#ifndef O2_MCH_EVALUATION_COMPARE_TASK_H__
#define O2_MCH_EVALUATION_COMPARE_TASK_H__

#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/InitContext.h"
#include "MCHEvaluation/ExtendedTrack.h"
#include <TFile.h>
#include <array>
#include <gsl/span>
#include <list>
#include <memory>
#include <string>
#include <vector>

class TH1;

namespace o2::mch::eval
{
class CompareTask
{
 public:
  CompareTask(std::shared_ptr<o2::base::GRPGeomRequest> req);

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj);

  void init(o2::framework::InitContext& ic);

  std::list<ExtendedTrack> convert(gsl::span<const TrackMCH> mchTracks,
                                   gsl::span<const Cluster> clusters);

  void dump(std::string prefix,
            const std::list<ExtendedTrack>& tracks1,
            const std::list<ExtendedTrack>& tracks2);

  std::list<ExtendedTrack> getExtendedTracks(const ROFRecord& rof,
                                             gsl::span<const TrackMCH> tfTracks,
                                             gsl::span<const Cluster> tfClusters);

  void run(o2::framework::ProcessingContext& pc);

 private:
  void pdfOutput();
  void printStat();

 private:
  std::shared_ptr<o2::base::GRPGeomRequest> mCcdbRequest;
  std::unique_ptr<TFile> mOutputRootFile;
  std::string mOutputPdfFileName;
  std::array<std::vector<TH1*>, 2> mHistosAtVertex;
  std::vector<TH1*> mTrackResidualsAtFirstCluster{};
  std::array<std::vector<TH1*>, 5> mComparisonsAtVertex;
  std::array<std::vector<TH1*>, 5> mClusterResiduals;
  std::array<int, 2> mNTracksAll{};
  std::array<int, 2> mNTracksMatch{};
  bool mApplyTrackSelection;
  double mPrecision;
  bool mPrintDiff;
  bool mPrintAll;
  int mNofDifferences;
};
} // namespace o2::mch::eval

#endif
