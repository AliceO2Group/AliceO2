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

#include "EventVisualisationView/Initializer.h"

#include <TApplication.h>
#include <TEveBrowser.h>
#include <TEveManager.h>

#include <ctime>
#include <iostream>

using namespace std;
using namespace o2::event_visualisation;


/*ITS */
#include <iostream>
#include <array>
#include <algorithm>
#include <fstream>

#include <TFile.h>
#include <TTree.h>
#include <TEveManager.h>
#include <TEveBrowser.h>
#include <TGButton.h>
#include <TGNumberEntry.h>
#include <TGFrame.h>
#include <TGTab.h>
#include <TGLCameraOverlay.h>
#include <TEveFrameBox.h>
#include <TEveQuadSet.h>
#include <TEveTrans.h>
#include <TEvePointSet.h>
#include <TEveTrackPropagator.h>
#include <TEveTrack.h>
#include <Rtypes.h>

#include "EventVisualisationView/MultiView.h"

#include <TEnv.h>
#include <EventVisualisationBase/ConfigurationManager.h>

#include "EventVisualisationView/MultiView.h"
extern TEveManager* gEve;
void drawEvent(TEveElementList* mEvent) {
    auto multi = o2::event_visualisation::MultiView::getInstance();
    multi->registerEvent(mEvent);
    gEve->Redraw3D(kFALSE);
}


int main(int argc, char **argv)
{
    cout<<"Welcome in O2 event visualisation tool"<<endl;

    srand(static_cast<unsigned int>(time(nullptr)));

    TEnv settings;
    ConfigurationManager::getInstance().getConfig(settings);

    std::array<const char*, 7> keys = {"Gui.DefaultFont", "Gui.MenuFont", "Gui.MenuHiFont",
                                    "Gui.DocFixedFont", "Gui.DocPropFont", "Gui.IconFont", "Gui.StatusFont"};
    for(const auto& key : keys) {
        if(settings.Defined(key))
            gEnv->SetValue(key,  settings.GetValue(key, ""));
    }

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
    app->Terminate(0);

    return 0;
}

int old(int argc, char **argv)
{
    cout<<"Welcome in O2 event visualisation tool"<<endl;

    srand(static_cast<unsigned int>(time(nullptr)));

    TEnv settings;
    ConfigurationManager::getInstance().getConfig(settings);

    std::array<const char*, 7> keys = {"Gui.DefaultFont", "Gui.MenuFont", "Gui.MenuHiFont",
                                       "Gui.DocFixedFont", "Gui.DocPropFont", "Gui.IconFont", "Gui.StatusFont"};
    for(const auto& key : keys) {
        if(settings.Defined(key))
            gEnv->SetValue(key,  settings.GetValue(key, ""));
    }

    // create ROOT application environment
    TApplication *app = new TApplication("o2eve", &argc, argv);
    app->Connect("TEveBrowser", "CloseWindow()", "TApplication", app, "Terminate()");

    cout<<"Initializing TEveManager"<<endl;
    if(!(gEve=TEveManager::Create())){
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