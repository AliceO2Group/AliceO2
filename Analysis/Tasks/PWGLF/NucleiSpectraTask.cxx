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

#include "PID/PIDResponse.h"

#include <TH1F.h>
#include <TH2F.h>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct NucleiSpecraTask {

  OutputObj<TH2F> hTPCsignal{TH2F("hTPCsignal", ";#it{p} (GeV/#it{c}); d#it{E}/d#it{X} (a. u.)", 600, 0., 3, 1400, 0, 1400)};
  OutputObj<TH1F> hMomentum{TH1F("hMomentum", ";#it{p} (GeV/#it{c});", 600, 0., 3.)};

  void process(soa::Join<aod::Tracks, aod::TracksExtra> const& tracks)
  {
    for (auto& track : tracks) {
      hTPCsignal->Fill(track.p(), track.tpcSignal());
      hMomentum->Fill(track.p());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<NucleiSpecraTask>("nuclei-spectra")};
}
