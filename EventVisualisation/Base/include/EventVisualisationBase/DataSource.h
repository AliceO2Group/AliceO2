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
/// \brief reading from file(s)
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCE_H
#define ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCE_H

#include <EventVisualisationBase/VisualisationConstants.h>
#include <EventVisualisationBase/DataReader.h>
#include <EventVisualisationDataConverter/VisualisationEvent.h>
#include <utility>

class TObject;

namespace o2
{
namespace event_visualisation
{

class DataSource
{
 public:
  virtual Int_t getCurrentEvent() { return 0; };
  virtual void setCurrentEvent(Int_t /*currentEvent*/){};
  virtual int getEventCount() { return 0; };
  virtual bool refresh() { return false; }; // recompute
  DataSource() = default;

  /// Default destructor
  virtual ~DataSource() = default;

  /// Deleted copy constructor
  DataSource(DataSource const&) = delete;

  /// Deleted assignemt operator
  void operator=(DataSource const&) = delete;

  virtual std::vector<std::pair<VisualisationEvent, std::string>> getVisualisationList(int no) = 0;
  virtual void registerDetector(DataReader* /*reader*/, EVisualisationGroup /*type*/){};
};

} // namespace event_visualisation
} // namespace o2

#endif //ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCE_H
