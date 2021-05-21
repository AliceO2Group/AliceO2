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
/// \file    Initializer.cxx
/// \author  Jeremi Niedziela
/// \author  julian.myrcha@cern.ch
/// \author  p.nowakowski@cern.ch
///

#include "EventVisualisationView/Initializer.h"

#include "EventVisualisationBase/ConfigurationManager.h"
#include "EventVisualisationBase/GeometryManager.h"
#include "EventVisualisationView/EventManager.h"
#include "EventVisualisationView/MultiView.h"
#include "EventVisualisationBase/VisualisationConstants.h"
#include "EventVisualisationBase/DataSourceOffline.h"
#include "EventVisualisationView/EventManagerFrame.h"
#include "EventVisualisationView/Options.h"

#include "EventVisualisationDetectors/DataInterpreterITS.h"
#include "EventVisualisationDetectors/DataReaderITS.h"
#include "EventVisualisationDetectors/DataInterpreterTPC.h"
#include "EventVisualisationDetectors/DataReaderTPC.h"

#include "EventVisualisationDetectors/DataReaderJSON.h"

#include "FairLogger.h"

#include <TGTab.h>
#include <TEnv.h>
#include <TEveBrowser.h>
#include <TEveManager.h>
#include <TRegexp.h>
#include <TSystem.h>
#include <TEveWindowManager.h>
using namespace std;

namespace o2
{
namespace event_visualisation
{

void Initializer::setup(EventManager::EDataSource defaultDataSource)
{
  TEnv settings;
  ConfigurationManager::getInstance().getConfig(settings);

  const bool fullscreen = settings.GetValue("fullscreen.mode", false);                           // hide left and bottom tabs
  const string ocdbStorage = settings.GetValue("OCDB.default.path", "local://$ALICE_ROOT/OCDB"); // default path to OCDB
  LOG(INFO) << "Initializer -- OCDB path:" << ocdbStorage;

  auto& eventManager = EventManager::getInstance();
  eventManager.setCdbPath(ocdbStorage);

  eventManager.setDataSourceType(defaultDataSource);
  eventManager.Open();
  if (Options::Instance()->tpc()) {
    eventManager.getDataSource()->registerDetector(new DataReaderTPC(new DataInterpreterTPC()), EVisualisationGroup::TPC);
  }
  if (Options::Instance()->its()) {
    eventManager.getDataSource()->registerDetector(new DataReaderITS(new DataInterpreterITS()), EVisualisationGroup::ITS);
  }
  if (Options::Instance()->json()) {
    eventManager.getDataSource()->registerDetector(new DataReaderJSON(nullptr), EVisualisationGroup::JSON);
  }


  GeometryManager::getInstance().setR2Geometry(std::string(settings.GetValue("simple.geom.default", "R3")).compare("R2") == 0);

  setupGeometry();

  gSystem->ProcessEvents();
  gEve->Redraw3D(true);

  setupBackground();

  // Setup windows size, fullscreen and focus
  TEveBrowser* browser = gEve->GetBrowser();
  browser->GetTabRight()->SetTab(1);
  browser->MoveResize(0, 0, gClient->GetDisplayWidth(), gClient->GetDisplayHeight() - 32);

  browser->StartEmbedding(TRootBrowser::kBottom);
  EventManagerFrame* frame = new EventManagerFrame(eventManager);
  browser->StopEmbedding("EventCtrl Balbinka");

  if (fullscreen) {
    ((TGWindow*)gEve->GetBrowser()->GetTabLeft()->GetParent())->Resize(1, 0);
    ((TGWindow*)gEve->GetBrowser()->GetTabBottom()->GetParent())->Resize(0, 1);
  }
  gEve->GetBrowser()->Layout();
  gSystem->ProcessEvents();

  setupCamera();

  // Temporary:
  // For the time being we draw single random event on startup.
  // Later this will be triggered by button, and finally moved to configuration.
  gEve->AddEvent(&EventManager::getInstance());
  eventManager.getDataSource()->refresh();
  frame->DoFirstEvent();
  frame->StartTimer();
}

void Initializer::setupGeometry()
{
  // read path to geometry files from config file
  TEnv settings;
  ConfigurationManager::getInstance().getConfig(settings);

  // get geometry from Geometry Manager and register in multiview
  auto multiView = MultiView::getInstance();

  //auto geometry_enabled = GeometryManager::getInstance().getR2Geometry()? R2Visualisation:R3Visualisation;
  for (int iDet = 0; iDet < NvisualisationGroups; ++iDet) {

    if (GeometryManager::getInstance().getR2Geometry()) {
      if (!R2Visualisation[iDet]) {
        continue;
      }
    }

    if (!GeometryManager::getInstance().getR2Geometry()) {
      if (!R3Visualisation[iDet]) {
        continue;
      }
    }

    EVisualisationGroup det = static_cast<EVisualisationGroup>(iDet);
    string detName = gVisualisationGroupName[det];
    LOG(INFO) << detName;

    if (settings.GetValue((detName + ".draw").c_str(), false)) {
      if (detName == "TPC" || detName == "MCH" || detName == "MID" || detName == "MFT") { // don't load MUON+MFT and AD and standard TPC to R-Phi view

        multiView->drawGeometryForDetector(detName, true, false);
      } else if (detName == "RPH") { // special TPC geom from R-Phi view

        multiView->drawGeometryForDetector(detName, false, true, false);
      } else { // default
        if (detName != "ACO") {
          multiView->drawGeometryForDetector(detName);
        }
      }
    }
  }
}

void Initializer::setupCamera()
{
  // move and rotate sub-views
  TEnv settings;
  ConfigurationManager::getInstance().getConfig(settings);

  // read settings from config file
  const double angleHorizontal = settings.GetValue("camera.3D.rotation.horizontal", -0.4);
  const double angleVertical = settings.GetValue("camera.3D.rotation.vertical", 1.0);

  double zoom[MultiView::NumberOfViews];
  zoom[MultiView::View3d] = settings.GetValue("camera.3D.zoom", 1.0);
  zoom[MultiView::ViewRphi] = settings.GetValue("camera.R-Phi.zoom", 1.0);
  zoom[MultiView::ViewZrho] = settings.GetValue("camera.Rho-Z.zoom", 1.0);

  // get necessary elements of the multiview and set camera position
  auto multiView = MultiView::getInstance();

  for (int viewIter = 0; viewIter < MultiView::NumberOfViews; ++viewIter) {
    TGLViewer* glView = multiView->getView(static_cast<MultiView::EViews>(viewIter))->GetGLViewer();
    glView->CurrentCamera().Reset();

    if (viewIter == 0) {
      glView->CurrentCamera().RotateRad(angleHorizontal, angleVertical);
    }
    glView->CurrentCamera().Dolly(zoom[viewIter], kFALSE, kTRUE);
  }
}

void Initializer::setupBackground()
{
  // get viewers of multiview and change color to the value from config file
  TEnv settings;
  ConfigurationManager::getInstance().getConfig(settings);
  Color_t col = settings.GetValue("background.color", 1);

  for (int viewIter = 0; viewIter < MultiView::NumberOfViews; ++viewIter) {
    TEveViewer* view = MultiView::getInstance()->getView(static_cast<MultiView::EViews>(viewIter));
    view->GetGLViewer()->SetClearColor(col);
  }
}

} // namespace event_visualisation
} // namespace o2
