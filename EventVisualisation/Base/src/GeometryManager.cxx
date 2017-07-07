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
/// \file    GeometryManager.cxx
/// \author  Jeremi Niedziela

#include "GeometryManager.h"

#include "Initializer.h"
#include "MultiView.h"

#include <TFile.h>
#include <TGLViewer.h>
#include <TEnv.h>
#include <TEveGeoShapeExtract.h>
#include <TEveManager.h>
#include <TEveProjectionManager.h>
#include <TSystem.h>

#include <iostream>

using namespace std;

namespace o2  {
namespace EventVisualisation {

GeometryManager* GeometryManager::sInstance = nullptr;
  
GeometryManager* GeometryManager::getInstance()
{
  if(!sInstance){
    new GeometryManager();
  }
  return sInstance;
}
 
void GeometryManager::drawGeometryForDetector(string detectorName,bool threeD, bool rPhi, bool zRho)
{
  TEveGeoShape *shape = getGeometryForDetector(detectorName);
  registerGeometry(shape, threeD, rPhi, zRho);
}
  
void GeometryManager::destroyAllGeometries()
{
  for(int i=0;i<mGeomVector.size();++i)
  {
    if(mGeomVector[i])
    {
      mGeomVector[i]->DestroyElements();
      gEve->RemoveElement(mGeomVector[i],MultiView::getInstance()->getScene(MultiView::Scene3dGeom));
      mGeomVector[i] = 0;
    }
  }
}
  
GeometryManager::GeometryManager()
{
  cout<<"Creating geometry manager"<<endl;
  sInstance = this;
}

GeometryManager::~GeometryManager()
{
}

TEveGeoShape* GeometryManager::getGeometryForDetector(string detectorName)
{
  TEnv settings;
  Initializer::getConfig(settings);
 
  // read geometry path from config file
  string geomPath = settings.GetValue("simple.geom.path","");
  const string o2basePathSpecifier = "${ALICE_ROOT}";
  const string o2basePath = "";//gSystem->Getenv("ALICE_ROOT");
  const size_t o2pos = geomPath.find(o2basePathSpecifier);

  if(o2pos != string::npos){
    geomPath.replace(o2pos,o2pos+o2basePathSpecifier.size(),o2basePath);
  }
  
  // load ROOT file with geometry
  TFile *f = TFile::Open(Form("%s/simple_geom_%s.root",geomPath.c_str(),detectorName.c_str()));
  if(!f){
    cout<<"GeometryManager::GetSimpleGeom -- no file with geometry found!"<<endl;
    return nullptr;
  }
  
  TEveGeoShapeExtract *geomShapreExtract = static_cast<TEveGeoShapeExtract*>(f->Get(detectorName.c_str()));
  TEveGeoShape *geomShape = TEveGeoShape::ImportShapeExtract(geomShapreExtract);
  f->Close();
  
  // tricks for different R-Phi geom of TPC:
  if(detectorName=="RPH"){  // use all other parameters of regular TPC geom
    detectorName = "TPC";
  }

  // prepare geometry to be drawn including all children
  drawDeep(geomShape,
           settings.GetValue(Form("%s.color",detectorName.c_str()),-1),
           settings.GetValue(Form("%s.trans",detectorName.c_str()),-1),
           settings.GetValue(Form("%s.line.color",detectorName.c_str()),-1));

  gEve->GetDefaultGLViewer()->UpdateScene();

  return geomShape;
}

void GeometryManager::drawDeep(TEveGeoShape *geomShape,Color_t color, Char_t transparency, Color_t lineColor)
{
  if(geomShape->HasChildren()){
    geomShape->SetRnrSelf(false);
    
    if(strcmp(geomShape->GetElementName(),"TPC_Drift_1")==0){// hack for TPC drift chamber
      geomShape->SetRnrSelf(kTRUE);
      if(color>=0) geomShape->SetMainColor(color);
      if(lineColor>=0){
        geomShape->SetLineColor(lineColor);
        geomShape->SetLineWidth(0.1);
        geomShape->SetDrawFrame(true);
      }
      else{
        geomShape->SetDrawFrame(false);
      }
      if(transparency>=0){
        geomShape->SetMainTransparency(transparency);
      }
    }
    
    for(TEveElement::List_i i = geomShape->BeginChildren(); i != geomShape->EndChildren(); ++i){
      drawDeep(static_cast<TEveGeoShape*>(*i),color,transparency,lineColor);
    }
  }
  else
  {
    geomShape->SetRnrSelf(true);
    if(color>=0) geomShape->SetMainColor(color);
    if(lineColor>=0){
      geomShape->SetLineColor(lineColor);
      geomShape->SetLineWidth(0.1);
      geomShape->SetDrawFrame(true);
    }
    else{
      geomShape->SetDrawFrame(false);
    }
    if(transparency>=0){
      geomShape->SetMainTransparency(transparency);
    }
      
    if(strcmp(geomShape->GetElementName(),"PHOS_5")==0){// hack for PHOS module which is not installed
      geomShape->SetRnrSelf(false);
    }
  }
}

void GeometryManager::registerGeometry(TEveGeoShape *geom, bool threeD, bool rPhi, bool zRho)
{
  if(!geom){
    cout<<"GeometryManager::InitSimpleGeom -- geometry is NULL!"<<endl;
    return;
  }
  mGeomVector.push_back(geom);
  
  auto multiView = MultiView::getInstance();
  TEveProjectionManager *projection;
  
  if(threeD){
    gEve->AddElement(geom,multiView->getScene(MultiView::Scene3dGeom));
  }
  if(rPhi){
    projection = multiView->getProjection(MultiView::ProjectionRphi);
    projection->SetCurrentDepth(-10);
    projection->ImportElements(geom, multiView->getScene(MultiView::SceneRPhiGeom));
    projection->SetCurrentDepth(0);
  }
  if(zRho){
    projection = multiView->getProjection(MultiView::ProjectionZrho);
    projection->SetCurrentDepth(-10);
    projection->ImportElements(geom, multiView->getScene(MultiView::SceneZrhoGeom));
    projection->SetCurrentDepth(0);
  }
}
  
}
}
