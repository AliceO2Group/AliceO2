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
  Int_t mCurrentEvent = 0;
  DataReader* mDataReaders[EVisualisationGroup::NvisualisationGroups];
  VisualisationEvent getEventData(int no, EVisualisationGroup purpose, EVisualisationDataType dataType);

 public:
  DataSourceOffline();

  ~DataSourceOffline() override = default;
  DataSourceOffline(DataSourceOffline const&) = delete;
  void setCurrentEvent(Int_t currentEvent) override;
  Int_t getCurrentEvent() override;

  /// Deleted assigment operator
  void operator=(DataSourceOffline const&) = delete;

  int getEventCount() override;

  void registerReader(DataReader* reader, EVisualisationGroup type);

  std::vector<std::pair<VisualisationEvent, std::string>> getVisualisationList(int no) override;
};

} // namespace event_visualisation
} // namespace o2

#endif //ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCEOFFLINE_H
