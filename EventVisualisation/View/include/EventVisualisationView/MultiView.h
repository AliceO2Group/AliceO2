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
/// \file    MultiView.h
/// \author  Jeremi Niedziela
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_VIEW_MULTIVIEW_H
#define ALICE_O2_EVENTVISUALISATION_VIEW_MULTIVIEW_H

#include <TGLAnnotation.h>
#include <TGLViewer.h>
#include <TEveGeoShape.h>
#include <TEveScene.h>
#include <TEveViewer.h>

#include <vector>

namespace o2
{
namespace event_visualisation
{

/// This singleton class manages views and scenes of the event display.
///
/// MultiView will create all necessary views and scenes, give them proper names
/// and descriptions and provide pointers to these objects. It also allows to draw
/// or remove simplified geometries. One can also register visualisation objects for
/// drawing in the MultiView, which will be imported to 3D view and projections.

class MultiView
{
 public:
  enum EViews {
    View3d,       ///< 3D view
    ViewRphi,     ///< R-Phi view
    ViewZY,       ///< Z-Y view
    NumberOfViews ///< Total number of views
  };

  enum EScenes {
    Scene3dGeom,    ///< 3D scene of geometry
    Scene3dEvent,   ///< 3D scene of event
    SceneRphiGeom,  ///< R-Phi scene of geometry
    SceneZYGeom,    ///< Z-Y scene of geometry
    SceneRphiEvent, ///< R-Phi scene of event
    SceneZYEvent,   ///< Z-Y scene of event
    NumberOfScenes  ///< Total number of scenes
  };
  enum EProjections {
    ProjectionRphi,     ///< R-Phi projection
    ProjectionZY,       ///< Z-Y projection
    NumberOfProjections ///< Total number of projections
  };

  /// Returns an instance of the MultiView
  static MultiView* getInstance();

  /// Returns pointer to specific view
  inline TEveViewer* getView(EViews view) { return mViews[view]; }
  /// Returns pointer to specific scene
  inline TEveScene* getScene(EScenes scene) { return mScenes[scene]; }
  /// Returns pointer to specific projection manager
  inline TEveProjectionManager* getProjection(EProjections projection) { return mProjections[projection]; }

  /// Returns the detector geometry for a given name
  /// \param detectorName The name of the requested detector
  TEveGeoShape* getDetectorGeometry(const std::string& detectorName);

  /// Draws geometry for given detector
  /// \param detectorName  The name of the detector to draw geometry of
  /// \param threeD Should 3D view be drawn
  /// \param rPhi Should R-Phi projection be drawn
  /// \param zy Should Z-Y projection be drawn
  void drawGeometryForDetector(std::string detectorName, bool threeD = true, bool rPhi = true, bool zy = true);
  /// Registers geometry to be drawn in appropriate views
  void registerGeometry(TEveGeoShape* geom, bool threeD = true, bool rPhi = true, bool zy = true);
  /// Removes all geometries
  void destroyAllGeometries();

  /// Registers an elements to be drawn
  void registerElements(TEveElementList* elements[], TEveElementList* phiElements[]);

  /// Registers an element to be drawn
  void registerElement(TEveElement* event);
  void registerEvent(TEveElement* event) { return registerElement(event); }

  /// Get annotation pointer
  TGLAnnotation* getAnnotationTop() { return mAnnotationTop.get(); }
  TGLAnnotation* getAnnotationBottom() { return mAnnotationBottom.get(); }

  /// Removes all shapes representing current event
  void destroyAllEvents(); // override;
  void redraw3D();

 private:
  /// Default constructor
  MultiView();
  /// Default destructor
  ~MultiView();

  static MultiView* sInstance; ///< Single instance of the multiview

  TEveViewer* mViews[NumberOfViews];                        ///< Array of all views
  TEveScene* mScenes[NumberOfScenes];                       ///< Array of all geometry and event scenes
  TEveProjectionManager* mProjections[NumberOfProjections]; ///< Array of all projection managers
  std::vector<TEveGeoShape*> mDetectors;                    ///< Vector of detector geometries
  std::unique_ptr<TGLAnnotation> mAnnotationTop;            ///< 3D view annotation (top)
  std::unique_ptr<TGLAnnotation> mAnnotationBottom;         ///< 3D view annotation (bottom)

  std::string mSceneNames[NumberOfScenes];        ///< Names of event and geometry scenes
  std::string mSceneDescriptions[NumberOfScenes]; ///< Descriptions of event and geometry scenes

  /// Splits the window into sectors for 3D view and projections
  void setupMultiview();
  /// Returns geometry scene for given projection manager
  EScenes getSceneOfProjection(EProjections projection);
};

} // namespace event_visualisation
} // namespace o2

#endif // ALICE_O2_EVENTVISUALISATION_VIEW_MULTIVIEW_H
