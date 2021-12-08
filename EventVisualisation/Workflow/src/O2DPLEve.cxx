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
/// \file    O2DPLEve.cxx
/// \author  Piotr Nowakowski
/// \author p.nowakowski@cern.ch

#include "EventVisualisationView/Initializer.h"
#include "EventVisualisationBase/ConfigurationManager.h"
#include "EventVisualisationBase/DataSourceOffline.h"
#include "EventVisualisationView/Options.h"

#include "EveWorkflow/DetectorData.h"
#include "EveWorkflow/EveWorkflowHelper.h"

#include "Framework/AnalysisTask.h"
#include "Framework/ConfigParamSpec.h"

#include "FairLogger.h"

#include <TApplication.h>
#include <TEveBrowser.h>
#include <TEveManager.h>
#include <TEnv.h>
#include <TSystem.h>

#include <ctime>

using namespace std;
using namespace o2;
using namespace o2::event_visualisation;
using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
}

struct O2DPLEve {
  DetectorData mData;
  std::shared_ptr<EveWorkflowHelper> mHelper;
  o2::framework::ControlService* mCoS;

  void init(o2::framework::InitContext& ic);
  void process(o2::aod::Collisions const& collision, soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra> const& tracks);
  void endOfStream(o2::framework::EndOfStreamContext& ec);

  static void updateCallback();
};

void O2DPLEve::init(o2::framework::InitContext& ic)
{
  srand(static_cast<unsigned int>(time(nullptr)));

  TEnv settings;
  ConfigurationManager::getInstance().getConfig(settings);

  std::array<const char*, 7> keys = {"Gui.DefaultFont", "Gui.MenuFont", "Gui.MenuHiFont",
                                     "Gui.DocFixedFont", "Gui.DocPropFont", "Gui.IconFont", "Gui.StatusFont"};
  for (const auto& key : keys) {
    if (settings.Defined(key)) {
      gEnv->SetValue(key, settings.GetValue(key, ""));
    }
  }

  // TODO: fix this ugly hack
  int argc = 0;
  char* argv[1];

  gApplication = new TApplication("o2eve", &argc, argv);

  LOG(info) << "Initializing TEveManager";
  if (!TEveManager::Create(kTRUE, "FI")) {
    LOG(fatal) << "Could not create TEveManager!!";
    exit(0);
  }

  // Initialize o2 Event Visualisation
  Initializer::setup();

  auto cbS = ic.services().get<o2::framework::CallbackService>();

  cbS.set(CallbackService::Id::ClockTick, O2DPLEve::updateCallback);

  //Initialize track propagation
  mData.init();
  mHelper = std::make_shared<EveWorkflowHelper>();

  // TODO: connect mCoS->readyToQuit(QuitRequest::All) to eve browser close
  mCoS = &ic.services().get<o2::framework::ControlService>();
}

void O2DPLEve::process(o2::aod::Collisions const& collisions, soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra> const& tracks)
{
  auto& eventManager = EventManager::getInstance();
  auto offlineSource = dynamic_cast<DataSourceOffline*>(eventManager.getDataSource());

  for (auto const& c : collisions) {
    LOGF(info, "COLLISION %d", c.globalIndex());

    auto const tracksCol = tracks.sliceBy(aod::track::collisionId, c.globalIndex());

    for (auto const& track : tracksCol) {
      mHelper->drawAOD(track, c.collisionTime());
    }

    offlineSource->addEvent(mHelper->currentEvent());

    mHelper->clear();
  }

  offlineSource->refresh();
}

void O2DPLEve::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  TEveManager::Terminate();
  gApplication->Terminate(0);
}

void O2DPLEve::updateCallback()
{
  gSystem->ProcessEvents();
  gEve->Redraw3D(true);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<O2DPLEve>(cfgc, TaskName{"o2-dpl-eve"}),
  };
}
