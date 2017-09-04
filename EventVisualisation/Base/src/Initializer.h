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
/// \file    Initializer.h
/// \author  Jeremi Niedziela
///

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_INITIALIZER_H
#define ALICE_O2_EVENTVISUALISATION_BASE_INITIALIZER_H

#include "EventManager.h"

#include <TEnv.h>

namespace o2  {
namespace EventVisualisation {

/// This class initializes a core of the visualisation system.
///
/// Initializer should be created only once when starting
/// event display. It will create all scenes, load geometries,
/// setup camera and background and finally resize and position
/// the main window.

class Initializer
{
  public:
    /// Default constructor
    explicit Initializer(const EventManager::EDataSource defaultDataSource = EventManager::SourceOffline);// default data source will be moved to a config file
    /// Default destructor
    ~Initializer();
    
    /// Returns current event display configuration
    static void getConfig(TEnv &settings); // this will be moved to different class
  private:
    /// Loads geometry for all detectors
    void setupGeometry();
    /// Sets up background color
    void setupBackground();
    /// Sets up camera position
    void setupCamera();
};

#endif

}
}
