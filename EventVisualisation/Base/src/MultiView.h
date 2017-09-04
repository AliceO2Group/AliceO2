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
/// \file    MultiView.h
/// \author  Jeremi Niedziela

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_MULTIVIEW_H
#define ALICE_O2_EVENTVISUALISATION_BASE_MULTIVIEW_H

#include <TGLViewer.h>
#include <TEveGeoShape.h>
#include <TEveScene.h>
#include <TEveViewer.h>

#include <vector>

namespace o2  {
namespace EventVisualisation {

/// This singleton class manages views and scenes of the event display.
///
/// MultiView will create all necessary views and scenes, give them proper names
/// and descriptions and provide pointers to these objects. 
  
class MultiView
{
  public:
    enum EViews{
      View3d,       ///< 3D view
      ViewRphi,     ///< R-Phi view
      ViewZrho ,    ///< Z-Rho view
      NumberOfViews ///< Total number of views
    };
    
    enum EScenes{
      Scene3dGeom,    ///< 3D scene of geometry
      SceneRPhiGeom,  ///< R-Phi scene of geometry
      SceneZrhoGeom , ///< Z-Pho scene of geometry
      Scene3dEvent,   ///< 3D scene of event
      SceneRphiEvent, ///< R-Phi scene of event
      SceneZrhoEvent ,///< Z-Rho scene of event
      NumberOfScenes  ///< Total number of scenes
    };
    enum EProjections{
      ProjectionRphi,     ///< R-Phi projection
      ProjectionZrho,     ///< Z-Rho projection
      NumberOfProjections ///< Total number of projections
    };
    
    /// Returns an instance of the MultiView
    static MultiView* getInstance();
    
    /// Returns pointer to specific view
    inline TEveViewer *getView(EViews view){return mViews[view];}
    /// Returns pointer to specific scene
    inline TEveScene* getScene(EScenes scene){return mScenes[scene];}
    /// Returns pointer to specific projection manager
    inline TEveProjectionManager* getProjection(EProjections projection){return mProjections[projection];}
    
  private:
    /// Default constructor
    MultiView();
    /// Default destructor
    ~MultiView();
    
    static MultiView *sInstance;                              ///< Single instance of the multiview

    TEveViewer *mViews[NumberOfViews];                        ///< Array of all views
    TEveScene  *mScenes[NumberOfScenes];                      ///< Array of all geometry and event scenes
    TEveProjectionManager *mProjections[NumberOfProjections]; ///< Array of all projection managers
    
    std::string mSceneNames[NumberOfScenes];                  ///< Names of event and geometry scenes
    std::string mSceneDescriptions[NumberOfScenes];           ///< Descriptions of event and geometry scenes

    /// Splits the window into sectors for 3D view and projections
    void setupMultiview();
    /// Returns geometry scene for given projection manager
    EScenes getSceneOfProjection(EProjections projection);
};

}
}

#endif
