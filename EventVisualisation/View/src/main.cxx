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
/// \file    main.cxx
/// \author  Jeremi Niedziela
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch

#include "EventVisualisationView/Initializer.h"
#include "EventVisualisationBase/ConfigurationManager.h"

#include "EventVisualisationBase/DataInterpreter.h"
#include "EventVisualisationView/Options.h"

#include "FairLogger.h"

#include <TApplication.h>
#include <TEveBrowser.h>
#include <TEveManager.h>
#include <TEnv.h>

#include <ctime>

using namespace std;
using namespace o2::event_visualisation;

int main(int argc, char** argv)
{
  LOG(INFO) << "Welcome in O2 event visualisation tool";

  if (!Options::Instance()->processCommandLine(argc, argv)) {
    exit(-1);
  }

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

  // create ROOT application environment
  TApplication* app = new TApplication("o2eve", &argc, argv);
  app->Connect("TEveBrowser", "CloseWindow()", "TApplication", app, "Terminate()");

  LOG(INFO) << "Initializing TEveManager";
  if (!TEveManager::Create(kTRUE, "FI")) {
    LOG(FATAL) << "Could not create TEveManager!!";
    exit(0);
  }

  //gEve->SpawnNewViewer("3D View", "");  exit(0);

  // Initialize o2 Event Visualisation
  Initializer::setup(Options::Instance()->online() ? EventManager::SourceOnline : EventManager::SourceOffline);

  // Start the application
  app->Run(kTRUE);

  //DataInterpreter::removeInstances();
  // Terminate application
  TEveManager::Terminate();
  app->Terminate(0);

  LOG(INFO) << "O2 event visualisation tool terminated properly";
  return 0;
}
