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

#include <TApplication.h>
#include <TEveBrowser.h>
#include <TEveManager.h>
#include <TEnv.h>

#include <unistd.h>
#include <ctime>
#include <iostream>

using namespace std;
using namespace o2::event_visualisation;

std::string printOptions(Options* o)
{
  std::string res;
  res.append(std::string("randomTracks: ") + (o->randomTracks ? "true" : "false") + "\n");
  res.append(std::string("vds         : ") + (o->vsd ? "true" : "false") + "\n");
  res.append(std::string("itc         : ") + (o->itc ? "true" : "false") + "\n");
  return res;
}

Options* processCommandLine(int argc, char* argv[])
{
  static Options options;
  int opt;

  // put ':' in the starting of the
  // string so that program can
  //distinguish between '?' and ':'
  while ((opt = getopt(argc, argv, ":if:rv")) != -1) {
    switch (opt) {
      case 'r':
        options.randomTracks = true;
        break;
      case 'i':
        options.itc = true;
        break;
      case 'v':
        options.vsd = true;
        break;
      case 'f':
        options.fileName = optarg;
        break;
      case ':':
        printf("option needs a value\n");
        return nullptr;
      case '?':
        printf("unknown option: %c\n", optopt);
        return nullptr;
    }
  }

  // optind is for the extra arguments
  // which are not parsed
  for (; optind < argc; optind++) {
    printf("extra arguments: %s\n", argv[optind]);
    return nullptr;
  }

  return &options;
}

int main(int argc, char** argv)
{
  cout << "Welcome in O2 event visualisation tool" << endl;
  Options* options = processCommandLine(argc, argv);
  if (options == nullptr)
    exit(-1);

  srand(static_cast<unsigned int>(time(nullptr)));

  TEnv settings;
  ConfigurationManager::getInstance().getConfig(settings);

  std::array<const char*, 7> keys = {"Gui.DefaultFont", "Gui.MenuFont", "Gui.MenuHiFont",
                                     "Gui.DocFixedFont", "Gui.DocPropFont", "Gui.IconFont", "Gui.StatusFont"};
  for (const auto& key : keys) {
    if (settings.Defined(key))
      gEnv->SetValue(key, settings.GetValue(key, ""));
  }

  // create ROOT application environment
  TApplication* app = new TApplication("o2eve", &argc, argv);
  app->Connect("TEveBrowser", "CloseWindow()", "TApplication", app, "Terminate()");

  cout << "Initializing TEveManager" << endl;
  if (!TEveManager::Create(kTRUE, "FI")) {
    cout << "FATAL -- Could not create TEveManager!!" << endl;
    exit(0);
  }

  // Initialize o2 Event Visualisation
  Initializer::setup(*options);

  // Start the application
  app->Run(kTRUE);

  DataInterpreter::removeInstances();
  // Terminate application
  TEveManager::Terminate();
  app->Terminate(0);

  return 0;
}
