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
/// \file    GeometryManager.h
/// \author  Jeremi Niedziela

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_GEOMETRYMANAGER_H
#define ALICE_O2_EVENTVISUALISATION_BASE_GEOMETRYMANAGER_H

#include <TEveGeoShape.h>

#include <string>

namespace o2  {
namespace EventVisualisation {

/// GeometryManager is responsible for drawing geometry of detectors.
///
/// GeometryManager is a singleton class which opens ROOT files with
/// simplified geometries, reads drawing parameters (such as color or transparency)
/// from the config file, prepares and draws the volumes. If needed, it can remove
/// and redraw geometry for given detector or all geometries.
  
class GeometryManager
{
  public:
    /// Returns an instance of GeometryManager
    static GeometryManager* getInstance();
    
    /// Draws geometry for given detector
    /// \param detectorName  The name of the detector to draw geometry of
    /// \param threeD Should 3D view be drawn
    /// \param rPhi Should R-Phi projection be drawn
    /// \param zRho Should Z-Rho projection be drawn
    void drawGeometryForDetector(std::string detectorName,bool threeD=true, bool rPhi=true, bool zRho=true);
    /// Removes all geometries
    void destroyAllGeometries();
    
  private:
    /// Default constructor
    GeometryManager();
    /// Default destructor
    ~GeometryManager();
    
    static GeometryManager *sInstance;        ///< Static instance of GeometryManager
    
    /// Vector keeping all geometries
    ///
    /// This is used just to know what to remove
    /// when destroying of all geometries is requested
    std::vector<TEveGeoShape*> mGeomVector;
    
    /// Returns ROOT shapes describing simplified geometry of given detector
    TEveGeoShape* getGeometryForDetector(std::string detectorName);
    /// Goes through all children nodes of geometry shape and sets drawing options
    void drawDeep(TEveGeoShape *geomShape, Color_t color, Char_t transparency, Color_t lineColor);
    /// Registers geometry to be drawn in appropriate views
    void registerGeometry(TEveGeoShape *geom,bool threeD=true, bool rPhi=true, bool zRho=true);
};
  
}
}

#endif
