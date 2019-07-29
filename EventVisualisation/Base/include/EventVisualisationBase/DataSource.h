// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataReaderITS.h
/// \brief VSD specific reading from file(s) (Visualisation Summary Data)
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCE_H
#define ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCE_H

#include <EventVisualisationBase/VisualisationConstants.h>

class TObject;

namespace o2
{
namespace event_visualisation
{

class DataSource
{
 public:
  virtual TObject* getEventData(int /*no*/, EVisualisationGroup /*purpose*/) { return nullptr; };
  virtual int GetEventCount() { return 0; };

  DataSource() = default;

  /// Default destructor
  virtual ~DataSource() = default;

  /// Deleted copy constructor
  DataSource(DataSource const&) = delete;

  /// Deleted assignemt operator
  void operator=(DataSource const&) = delete;
};

} // namespace event_visualisation
} // namespace o2

#endif //ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCE_H
