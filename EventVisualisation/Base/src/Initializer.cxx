// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    Initializer.cxx
/// \author  Jeremi Niedziela
///

#include "Initializer.h"

#include "EventManager.h"
#include "GeometryManager.h"
#include "MultiView.h"

#include <TGTab.h>
#include <TEnv.h>
#include <TEveBrowser.h>
#include <TEveManager.h>
#include <TRegexp.h>
#include <TSystem.h>
#include <TSystemDirectory.h>

#include <iostream>

using namespace std;

namespace o2  {
namespace EventVisualisation {

Initializer::Initializer(EventManager::EDataSource defaultDataSource)
{
  TEnv settings;
  getConfig(settings);
  
  const bool fullscreen      = settings.GetValue("fullscreen.mode",false);       // hide left and bottom tabs
  const string ocdbStorage   = settings.GetValue("OCDB.default.path","local://$ALICE_ROOT/OCDB");// default path to OCDB
  cout<<"Initializer -- OCDB path:"<<ocdbStorage<<endl;
  
  EventManager *eventManager = EventManager::getInstance();
  
  eventManager->setDataSourceType(defaultDataSource);
  eventManager->setCdbPath(ocdbStorage);
  
//  gEve->AddEvent(eventManager);
  
  setupGeometry();
  gSystem->ProcessEvents();
  gEve->Redraw3D(true);
  
  setupBackground();
  
  // Setup windows size, fullscreen and focus
  TEveBrowser *browser = gEve->GetBrowser();
  browser->GetTabRight()->SetTab(1);
  browser->MoveResize(0, 0, gClient->GetDisplayWidth(),gClient->GetDisplayHeight() - 32);
  
  if(fullscreen){
    ((TGWindow*)gEve->GetBrowser()->GetTabLeft()->GetParent())->Resize(1,0);
    ((TGWindow*)gEve->GetBrowser()->GetTabBottom()->GetParent())->Resize(0,1);
    
  }
  gEve->GetBrowser()->Layout();
  gSystem->ProcessEvents();
  
  setupCamera();
}
 
Initializer::~Initializer()
{
  
}
  
void Initializer::setupGeometry()
{
  // read path to geometry files from config file
  TEnv settings;
  getConfig(settings);
  
  string geomPath = settings.GetValue("simple.geom.path","${ALICEO2_ROOT}/EventVisualisation/resources/geometry/run3/");
  const string o2basePath = "";//gSystem->Getenv("ALICEO2_ROOT"); // this variable is not set in O2, to be fixed
  const size_t o2pos = geomPath.find("${ALICEO2_ROOT}");

  if(o2pos != string::npos){
    geomPath.replace(o2pos,o2pos+13,o2basePath);
  }

  // open root files matching "simple_geom_XYZ.txt"
  TSystemDirectory dir(geomPath.c_str(),geomPath.c_str());
  TList *files(dir.GetListOfFiles());
  vector<TString> detectorsList;
  
  if (files){
    TRegexp geomNamePattern("simple_geom_[A-Z,0-9][A-Z,0-9][A-Z,0-9].root");
    TRegexp detectorNamePattern("[A-Z,0-9][A-Z,0-9][A-Z,0-9]");
    
    TSystemFile *file = nullptr;
    TString fileName;
    TIter next(files);
    
    while ((file=static_cast<TSystemFile*>(next()))){
      fileName = file->GetName();
      
      if(fileName.Contains(geomNamePattern)){
        TString detectorName = fileName(detectorNamePattern);
        detectorName.Resize(3);
        detectorsList.push_back(detectorName);
      }
    }
  }
  else{
    cout<<"\n\nInitializer -- geometry files not found!!!"<<endl;
    cout<<"Searched directory was:"<<endl;
    dir.Print();
  }

  // get geometry from Geometry Manager and register in multiview
  auto geoManager = GeometryManager::getInstance();
  
  for(int i=0;i<detectorsList.size();++i){
    if(settings.GetValue(detectorsList[i]+".draw", true))
    {
      if(   detectorsList[i]=="TPC" || detectorsList[i]=="MCH" || detectorsList[i]=="MTR"
         || detectorsList[i]=="MID" || detectorsList[i]=="MFT" || detectorsList[i]=="AD0"
         || detectorsList[i]=="FMD"){// don't load MUON+MFT and AD and standard TPC to R-Phi view
        
        geoManager->drawGeometryForDetector(detectorsList[i].Data(),true,false);
      }
      else if(detectorsList[i]=="RPH"){// special TPC geom from R-Phi view
        
        geoManager->drawGeometryForDetector("RPH",false,true,false);
      }
      else{// default
        geoManager->drawGeometryForDetector(detectorsList[i].Data());
      }
    }
  }
}
 
void Initializer::setupCamera()
{
  // move and rotate sub-views
  TEnv settings;
  getConfig(settings);
  
  // read settings from config file
  const double angleHorizontal = settings.GetValue("camera.3D.rotation.horizontal",-0.4);
  const double angleVertical   = settings.GetValue("camera.3D.rotation.vertical",1.0);
  
  double zoom[MultiView::NumberOfViews];
  zoom[MultiView::View3d]   = settings.GetValue("camera.3D.zoom",1.0);
  zoom[MultiView::ViewRphi] = settings.GetValue("camera.R-Phi.zoom",1.0);
  zoom[MultiView::ViewZrho] = settings.GetValue("camera.Rho-Z.zoom",1.0);
  
  // get necessary elements of the multiview and set camera position
  auto multiView = MultiView::getInstance();
  
  for(int viewIter=0;viewIter<MultiView::NumberOfViews;++viewIter){
    TGLViewer *glView = multiView->getView(static_cast<MultiView::EViews>(viewIter))->GetGLViewer();
    glView->CurrentCamera().Reset();
    
    if(viewIter==0){
        glView->CurrentCamera().RotateRad(angleHorizontal, angleVertical);
    }
    glView->CurrentCamera().Dolly(zoom[viewIter], kFALSE, kTRUE);
  }
}

void Initializer::setupBackground()
{
  // get viewers of multiview and change color to the value from config file
  TEnv settings;
  getConfig(settings);
  
  for(int viewIter=0;viewIter<MultiView::NumberOfViews;++viewIter){
    TEveViewer *view = MultiView::getInstance()->getView(static_cast<MultiView::EViews>(viewIter));
    view->GetGLViewer()->SetClearColor(settings.GetValue("background.color",1));
  }
}
  
void Initializer::getConfig(TEnv &settings)
{
  if(settings.ReadFile(Form("%s/.eve_config",gSystem->Getenv("HOME")), kEnvUser) < 0)
  {
    if(settings.ReadFile(Form("%s/eve_config",gSystem->Getenv("HOME")), kEnvUser) < 0)
    {
      cout<<"WARNING -- could not find eve_config in home directory! Trying default one in O2/EventVisualisation/Base/"<<endl;
      if(settings.ReadFile(Form("%s/EventVisualisation/Base/src/eve_config",gSystem->Getenv("ALICEO2_INSTALL_PATH")), kEnvUser) < 0)
      {
        cout<<"ERROR -- could not find eve_config file!."<<endl;
        exit(0);
      }
    }
  }
}

}
}
