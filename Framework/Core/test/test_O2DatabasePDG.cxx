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
///
/// \brief FullTracks is a join of Tracks, TracksCov, and TracksExtra.
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/O2DatabasePDGPlugin.h"

#include <TDatabasePDG.h>
#include <cmath>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

#define ASSERT_ERROR(condition)                                   \
  if ((condition) == false) {                                     \
    LOG(fatal) << R"(Test condition ")" #condition R"(" failed)"; \
  }

struct PdgTest {
  Service<o2::framework::O2DatabasePDG> pdgNew;
  Service<TDatabasePDG> pdgOld;
  Service<o2::framework::ControlService> control;

  void process(Enumeration<0, 1>& e)
  {
    // Hardcoded from DataFormats/simulation/include/SimulationDataFormat/O2DatabasePDG.h
    TParticlePDG* p = pdgOld->GetParticle(300553);
    ASSERT_ERROR(p != nullptr);
    ASSERT_ERROR(p->Mass() == 10.580);
    ASSERT_ERROR(p->Stable() == kFALSE);
    ASSERT_ERROR(p->Charge() == 0);
    ASSERT_ERROR(p->Width() == 0.000);

    TParticlePDG* pNew = pdgNew->GetParticle(300553);
    ASSERT_ERROR(pNew != nullptr);
    ASSERT_ERROR(pNew->Mass() == 10.580);
    ASSERT_ERROR(pNew->Stable() == kFALSE);
    ASSERT_ERROR(pNew->Charge() == 0);
    ASSERT_ERROR(pNew->Width() == 0.000);
    control->readyToQuit(QuitRequest::Me);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<PdgTest>(cfgc),
  };
}
