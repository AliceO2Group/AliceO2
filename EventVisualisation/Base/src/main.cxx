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
///

#include "Initializer.h"

#include <TApplication.h>
#include <TEveBrowser.h>
#include <TEveManager.h>

#include <iostream>

using namespace std;
using namespace o2::EventVisualisation;

//#include <AliLog.h>  // logging will come soon to O2

int main(int argc, char **argv)
{
  cout<<"Welcome in O2 event visualisation tool"<<endl;
  
  // create ROOT application environment
  TApplication *app = new TApplication("o2eve", &argc, argv);
  app->Connect("TEveBrowser", "CloseWindow()", "TApplication", app, "Terminate()");
  
  cout<<"Initializing TEveManager"<<endl;
  if(!TEveManager::Create()){
    cout<<"FATAL -- Could not create TEveManager!!"<<endl;
    exit(0);
  }
  
  // Initialize o2 Event Visualisation
  auto initializer(new Initializer());
  
  // Start the application
  app->Run(kTRUE);
  
  // Terminate application
  TEveManager::Terminate();
  app->Terminate();
  
  return 0;
}
