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
/// \file    MultiView.cxx
/// \author  Jeremi Niedziela

#include "MultiView.h"

#include "Initializer.h"

#include <TBrowser.h>
#include <TEnv.h>
#include <TEveBrowser.h>
#include <TEveManager.h>
#include <TEveProjectionAxes.h>
#include <TEveProjectionManager.h>

#include <iostream>

namespace o2  {
namespace EventVisualisation {

MultiView *MultiView::sInstance = nullptr;

MultiView::MultiView()
{
  // set scene names and descriptions
  mSceneNames[Scene3dGeom]    = "3D Geometry";
  mSceneNames[SceneRPhiGeom]  = "R-Phi Geometry";
  mSceneNames[SceneZrhoGeom]  = "Rho-Z Geometry";
  mSceneNames[Scene3dEvent]   = "3D Event";
  mSceneNames[SceneRphiEvent] = "R-Phi Event";
  mSceneNames[SceneZrhoEvent] = "Rho-Z Event";
  
  mSceneDescriptions[Scene3dGeom]    = "Scene holding 3D geometry.";
  mSceneDescriptions[SceneRPhiGeom]  = "Scene holding projected geometry for the R-Phi view.";
  mSceneDescriptions[SceneZrhoGeom]  = "Scene holding projected geometry for the Rho-Z view.";
  mSceneDescriptions[Scene3dEvent]   = "Scene holding 3D event.";
  mSceneDescriptions[SceneRphiEvent] = "Scene holding projected event for the R-Phi view.";
  mSceneDescriptions[SceneZrhoEvent] = "Scene holding projected event for the Rho-Z view.";
  
  // spawn scenes
  for(int i=0;i<NumberOfScenes;++i){
    mScenes[i] = gEve->SpawnNewScene(mSceneNames[i].c_str(), mSceneDescriptions[i].c_str());
  }
  
  // Projection managers
  mProjections[ProjectionRphi] = new TEveProjectionManager();
  mProjections[ProjectionZrho] = new TEveProjectionManager();
  
  mProjections[ProjectionRphi]->SetProjection(TEveProjection::kPT_RPhi);
  mProjections[ProjectionZrho]->SetProjection(TEveProjection::kPT_RhoZ);
  
  gEve->AddToListTree(static_cast<TEveElement*>(mProjections[ProjectionRphi]),false);
  gEve->AddToListTree(static_cast<TEveElement*>(mProjections[ProjectionZrho]),false);
  
  // add axes
  TEnv settings;
  Initializer::getConfig(settings);
  const bool showAxes = settings.GetValue("axes.show", false);
  if(showAxes)
  {
    for(int i=0;i<NumberOfProjections;++i)
    {
      TEveProjectionAxes axes(mProjections[static_cast<EProjections>(i)]);
      axes.SetMainColor(kWhite);
      axes.SetTitle("R-Phi");
      axes.SetTitleSize(0.05);
      axes.SetTitleFont(102);
      axes.SetLabelSize(0.025);
      axes.SetLabelFont(102);
      mScenes[getSceneOfProjection(static_cast<EProjections>(i))]->AddElement(&axes);
    }
  }
  
  setupMultiview();
  
  sInstance = this;
}

MultiView::~MultiView()
{
}

MultiView* MultiView::getInstance()
{
  if(!sInstance){
    new MultiView();
  }
  return sInstance;
}

void MultiView::setupMultiview()
{
  // Split window in packs for 3D and projections, create viewers and add scenes to them
  TEveWindowSlot *slot = TEveWindow::CreateWindowInTab(gEve->GetBrowser()->GetTabRight());
  TEveWindowPack *pack = slot->MakePack();
  
  pack->SetElementName("Multi View");
  pack->SetHorizontal();
  pack->SetShowTitleBar(kFALSE);
  
  pack->NewSlotWithWeight(2)->MakeCurrent(); // new slot is created from pack
  mViews[View3d] = gEve->SpawnNewViewer("3D View", "");
  mViews[View3d]->AddScene(mScenes[Scene3dGeom]);
  mViews[View3d]->AddScene(mScenes[Scene3dEvent]);
  
  pack =  pack->NewSlot()->MakePack();
  pack->SetShowTitleBar(kFALSE);
  pack->NewSlot()->MakeCurrent();
  mViews[ViewRphi] = gEve->SpawnNewViewer("R-Phi View", "");
  mViews[ViewRphi]->GetGLViewer()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
  mViews[ViewRphi]->AddScene(mScenes[SceneRPhiGeom]);
  mViews[ViewRphi]->AddScene(mScenes[SceneRphiEvent]);
  
  pack->NewSlot()->MakeCurrent();
  mViews[ViewZrho] = gEve->SpawnNewViewer("Rho-Z View", "");
  mViews[ViewZrho]->GetGLViewer()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
  mViews[ViewZrho]->AddScene(mScenes[SceneZrhoGeom]);
  mViews[ViewZrho]->AddScene(mScenes[SceneZrhoEvent]);
}
  
MultiView::EScenes MultiView::getSceneOfProjection(EProjections projection)
{
  if(projection == ProjectionRphi){
    return SceneRPhiGeom;
  }
  else if(projection == ProjectionZrho){
    return SceneZrhoGeom;
  }
  return NumberOfScenes;
}
  
  
}
}
