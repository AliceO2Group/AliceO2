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
#include "Framework/HistogramRegistry.h"
#include <TH1F.h>

#include <cmath>

using namespace o2;
using namespace o2::framework;

// This is a very simple example showing how to create an histogram
// FIXME: this should really inherit from AnalysisTask but
//        we need GCC 7.4+ for that
struct ATask {
  /// Construct a registry object with direct declaration
  HistogramRegistry registry{"registry", true, {{"eta", "#eta", {HistogramType::kTH1F, {{102, -2.01, 2.01}}}}, {"phi", "#varphi", {HistogramType::kTH1F, {{100, 0., 2. * M_PI}}}}}};

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      registry.get("eta")->Fill(track.eta());
      registry.get("phi")->Fill(track.phi());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("eta-and-phi-histograms")};
}
