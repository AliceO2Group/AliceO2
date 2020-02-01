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
/// \file    DataInterpreter.h
/// \author  Jeremi Niedziela
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_DATAINTERPRETER_H
#define ALICE_O2_EVENTVISUALISATION_BASE_DATAINTERPRETER_H

#include "EventVisualisationBase/VisualisationConstants.h"
#include "EventVisualisationDataConverter/VisualisationEvent.h"

#include <TEveElement.h>

namespace o2
{
namespace event_visualisation
{

/// DataInterpreter is a template class for detector-specific visualisation classes.
///
/// Each detector should override this template class, providing method(s)
/// to interpret detector-specific data (such as hits, digits, clusters)
/// as a set of visualisation objects (points, lines, boxes).

class DataInterpreter
{
 private:
  static DataInterpreter* instance[EVisualisationGroup::NvisualisationGroups];

 public:
  // Default constructor
  DataInterpreter() = default;
  // Virtual destructor
  virtual ~DataInterpreter() = default;
  static void removeInstances()
  {
    for (int i = 0; i < EVisualisationGroup::NvisualisationGroups; i++)
      if (instance[i] != nullptr) {
        delete instance[i];
        instance[i] = nullptr;
      }
  }

  // Should return visualisation objects for required data type
  virtual std::unique_ptr<VisualisationEvent> interpretDataForType(TObject* data, EVisualisationDataType type) = 0;

  static DataInterpreter* getInstance(EVisualisationGroup type) { return instance[type]; }
  //static void setInstance(DataInterpreter* instance, EVisualisationGroup type) { DataInterpreter::instance[type] = instance; }
};

} // namespace event_visualisation
} // namespace o2

#endif // ALICE_O2_EVENTVISUALISATION_BASE_DATAINTERPRETER_H
