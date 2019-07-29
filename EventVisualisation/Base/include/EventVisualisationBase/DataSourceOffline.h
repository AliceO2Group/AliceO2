// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataSourceOffline.h
/// \brief Grouping reading from file(s)
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCEOFFLINE_H
#define ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCEOFFLINE_H

#include <EventVisualisationBase/DataSource.h>
#include <EventVisualisationBase/DataReader.h>

class TObject;

namespace o2
{
namespace event_visualisation
{

class DataSourceOffline : public DataSource
{
 protected:
  static DataReader* instance[EVisualisationGroup::NvisualisationGroups];

 public:
  DataSourceOffline() = default;

  ~DataSourceOffline() override = default;
  DataSourceOffline(DataSourceOffline const&) = delete;

  /// Deleted assigment operator
  void operator=(DataSourceOffline const&) = delete;

  int GetEventCount() override;

  void registerReader(DataReader* reader, EVisualisationGroup purpose)
  {
    instance[purpose] = reader;
  }

  TObject* getEventData(int no, EVisualisationGroup purpose) override;
};

} // namespace event_visualisation
} // namespace o2

#endif //ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCEOFFLINE_H
