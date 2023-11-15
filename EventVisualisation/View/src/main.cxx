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
/// \file    main.cxx
/// \author  Jeremi Niedziela
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch

#include "EventVisualisationView/Initializer.h"
#include "EventVisualisationBase/ConfigurationManager.h"
#include "EventVisualisationView/Options.h"

#include <fairlogger/Logger.h>

#include <TApplication.h>
#include <TEveBrowser.h>
#include <TEveManager.h>
#include <TEnv.h>

#include <ctime>

using namespace std;
using namespace o2::event_visualisation;

int main(int argc, char** argv)
{
  LOGF(info, "Welcome in O2 event visualisation tool (", o2_eve_version, ")");

  if (!Options::Instance()->processCommandLine(argc, argv)) {
    exit(-1);
  }

  srand(static_cast<unsigned int>(time(nullptr)));

  TEnv settings;
  ConfigurationManager::setOptionsFileName(Options::Instance()->optionsFileName());
  ConfigurationManager::getInstance().getConfig(settings);

  std::array<const char*, 7> keys = {"Gui.DefaultFont", "Gui.MenuFont", "Gui.MenuHiFont",
                                     "Gui.DocFixedFont", "Gui.DocPropFont", "Gui.IconFont", "Gui.StatusFont"};
  for (const auto& key : keys) {
    if (settings.Defined(key)) {
      gEnv->SetValue(key, settings.GetValue(key, ""));
    }
  }

  // create ROOT application environment
  auto app = new TApplication("o2eve", &argc, argv);
  gApplication = app;
  //app->Connect("TEveBrowser", "CloseWindow()", "TApplication", app, "Terminate()");

  LOGF(info, "Initializing TEveManager");
  if (!TEveManager::Create(kTRUE, "FI")) {
    LOGF(fatal, "Could not create TEveManager!!");
    exit(0);
  }

  // Initialize o2 Event Visualisation
  Initializer::setup();

  // Start the application
  app->Run(kTRUE);

  // Terminate application
  TEveManager::Terminate();
  app->Terminate(0);

  LOGF(info, "O2 event visualisation tool terminated properly");
  return 0;
}
