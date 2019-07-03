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
/// \file    DataInterpreterVSD.cxx
/// \author  Julian Myrcha

#include "EventVisualisationDetectors/DataInterpreterVSD.h"

#include "EventVisualisationBase/ConfigurationManager.h"
#include "EventVisualisationBase/Track.h"

#include "EventVisualisationDataConverter/MinimalisticEvent.h"

#include <TEveManager.h>
#include <TEveTrackPropagator.h>
#include <TGListTree.h>

#include <iostream>

using namespace std;

namespace o2 {
namespace event_visualisation {

DataInterpreterITS::DataInterpreterITS() = default;

DataInterpreterITS::~DataInterpreterITS() = default;

TEveElement* DataInterpreterITS::interpretDataForType(EDataType type) {
    int multiplicity = 500*((double)rand()/RAND_MAX)+100;
    MinimalisticEvent *minEvent = new MinimalisticEvent(
            15,             // event number
            123456,         // run number
            7000,           // energy
            multiplicity,   // multiplicity
            "p-p",          // event system
            12736563        // timestamp
            );


    return nullptr;
}

}
}