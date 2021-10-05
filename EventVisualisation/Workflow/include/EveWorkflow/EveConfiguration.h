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

/// \file
/// \author Julian Myrcha

#ifndef ALICE_O2_EVENTVISUALISATION_WORKFLOW_EVECONFIGURATION_H
#define ALICE_O2_EVENTVISUALISATION_WORKFLOW_EVECONFIGURATION_H

#include "DataFormatsTPC/Constants.h"
#include "TRDBase/GeometryFlat.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"

#include <memory>
#include <array>
#include <vector>
#include <functional>
#include <gsl/gsl>

class TH1F;

namespace o2
{
namespace event_visualisation
{
template <class T>
struct DefaultPtr {
  typedef T type;
};
template <class T>
struct ConstPtr {
  typedef const T type;
};

template <template <typename T> class S>
struct CalibObjectsTemplate {
  typename S<o2::trd::GeometryFlat>::type* trdGeometry = nullptr;
  typename S<o2::itsmft::TopologyDictionary>::type* itsPatternDict = nullptr;
  typename S<o2::itsmft::TopologyDictionary>::type* mftPatternDict = nullptr;
};

typedef CalibObjectsTemplate<DefaultPtr> CalibObjects; // NOTE: These 2 must have identical layout since they are memcopied
typedef CalibObjectsTemplate<ConstPtr> CalibObjectsConst;

struct SettingsGRP {
  // All new members must be sizeof(int) resp. sizeof(float) for alignment reasons!
  float solenoidBz = -5.00668;  // solenoid field strength
  int constBz = 0;              // for test-MC events with constant Bz
  int homemadeEvents = 0;       // Toy-MC events
  int continuousMaxTimeBin = 0; // 0 for triggered events, -1 for default of 23ms
  int needsClusterer = 0;       // Set to true if the data requires the clusterizer
};

struct SettingsProcessing {
  bool runMC;
};

struct EveConfiguration {
  EveConfiguration() = default;
  ~EveConfiguration() = default;
  EveConfiguration(const EveConfiguration&) = default;
  SettingsGRP configGRP;
  CalibObjectsConst configCalib;
  SettingsProcessing configProcessing;
  void ReadConfigurableParam();
};

} // namespace event_visualisation
} // namespace o2

#endif
