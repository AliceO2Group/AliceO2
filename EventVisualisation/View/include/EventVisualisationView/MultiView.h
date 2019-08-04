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

#include <EventVisualisationBase/EventRegistration.h>
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

class MultiView : public EventRegistration
{
 public:
  enum EViews {
    View3d,       ///< 3D view
    ViewRphi,     ///< R-Phi view
    ViewZrho,     ///< Z-Rho view
    NumberOfViews ///< Total number of views
  };

  enum EScenes {
    Scene3dGeom,    ///< 3D scene of geometry
    Scene3dEvent,   ///< 3D scene of event
    SceneRPhiGeom,  ///< R-Phi scene of geometry
    SceneZrhoGeom,  ///< Z-Pho scene of geometry
    SceneRphiEvent, ///< R-Phi scene of event
    SceneZrhoEvent, ///< Z-Rho scene of event
    NumberOfScenes  ///< Total number of scenes
  };
  enum EProjections {
    ProjectionRphi,     ///< R-Phi projection
    ProjectionZrho,     ///< Z-Rho projection
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

  /// Draws geometry for given detector
  /// \param detectorName  The name of the detector to draw geometry of
  /// \param threeD Should 3D view be drawn
  /// \param rPhi Should R-Phi projection be drawn
  /// \param zRho Should Z-Rho projection be drawn
  void drawGeometryForDetector(std::string detectorName, bool threeD = true, bool rPhi = true, bool zRho = true);
  /// Registers geometry to be drawn in appropriate views
  void registerGeometry(TEveGeoShape* geom, bool threeD = true, bool rPhi = true, bool zRho = true);
  /// Removes all geometries
  void destroyAllGeometries();

  /// Registers an element to be drawn
  void registerElement(TEveElement* event) override;

  ///
  void registerEvent(TEveElement* event) { return registerElement(event); }
  /// Removes all shapes representing current event
  void destroyAllEvents() override;

  void drawRandomEvent();

 private:
  /// Default constructor
  MultiView();
  /// Default destructor
  ~MultiView() = default;

  static MultiView* sInstance; ///< Single instance of the multiview

  TEveViewer* mViews[NumberOfViews];                        ///< Array of all views
  TEveScene* mScenes[NumberOfScenes];                       ///< Array of all geometry and event scenes
  TEveProjectionManager* mProjections[NumberOfProjections]; ///< Array of all projection managers

  std::string mSceneNames[NumberOfScenes];        ///< Names of event and geometry scenes
  std::string mSceneDescriptions[NumberOfScenes]; ///< Descriptions of event and geometry scenes

  /// Splits the window into sectors for 3D view and projections
  void setupMultiview();
  /// Returns geometry scene for given projection manager
  EScenes getSceneOfProjection(EProjections projection);

  /// Vector keeping all geometries
  ///
  /// This is used just to know what to remove
  /// when destroying of all geometries is requested
  std::vector<TEveGeoShape*> mGeomVector;
};

} // namespace event_visualisation
} // namespace o2

#endif
