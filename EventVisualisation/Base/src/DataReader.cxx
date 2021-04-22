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
/// \file    DataInterpreter.cxx
/// \author julian.myrcha@cern.ch

#include "EventVisualisationBase/DataReader.h"
#include "FairLogger.h"

using namespace std;

namespace o2
{
namespace event_visualisation
{

DataReader::DataReader(DataInterpreter* interpreter) : mInterpreter(interpreter)
{
}

VisualisationEvent DataReader::getEvent(int no, EVisualisationDataType dataType)
{
  TObject* data = this->getEventData(no);
  VisualisationEvent event = mInterpreter->interpretDataForType(data, dataType);
  return event;
}

} // namespace event_visualisation
} // namespace o2
