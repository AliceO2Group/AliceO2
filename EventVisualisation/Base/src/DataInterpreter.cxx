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
/// \author  Jeremi Niedziela

#include "EventVisualisationBase/DataInterpreter.h"

#include <iostream>

using namespace std;

namespace o2  {
namespace event_visualisation {

DataInterpreter* DataInterpreter::instance[EVisualisationGroup::NvisualisationGroups];
DataInterpreter::DataInterpreter() = default;

TEveElement* DataInterpreter::interpretDataForType(TObject* /*data*/, EVisualisationDataType /*type*/)
{
  cout<<"Virtual method interpretDataForType(EventManager::EDataType type) -- should be implemented in deriving class!!"<<endl;
  
  return nullptr;
}
  
}
}
