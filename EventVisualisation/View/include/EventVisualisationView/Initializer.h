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
/// \file    Initializer.h
/// \author  Jeremi Niedziela
///

#ifndef ALICE_O2_EVENTVISUALISATION_VIEW_INITIALIZER_H
#define ALICE_O2_EVENTVISUALISATION_VIEW_INITIALIZER_H

#include "EventVisualisationView/EventManager.h"

namespace o2
{
namespace event_visualisation
{

/// This class initializes a core of the visualisation system.
///
/// Initializer should be created only once when starting
/// event display. It will create MultiView, load geometries,
/// setup camera and background and finally resize and position
/// the main window.

class Initializer
{
 public:
  /// Default constructor
  static void setup(const EventManager::EDataSource defaultDataSource);

 private:
  /// Loads geometry for all detectors
  static void setupGeometry();
  /// Sets up background color
  static void setupBackground();
  /// Sets up camera position
  static void setupCamera();
};
} // namespace event_visualisation
} // namespace o2

#endif // ALICE_O2_EVENTVISUALISATION_VIEW_INITIALIZER_H
