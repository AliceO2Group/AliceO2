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
/// \file    VisualisationConstants.h
/// \author  Jeremi Niedziela
/// \author julian.myrcha@cern.ch
///

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_VISUALISATIONCONSTANTS_H
#define ALICE_O2_EVENTVISUALISATION_BASE_VISUALISATIONCONSTANTS_H

#include <string>

namespace o2  {
namespace event_visualisation {

enum EVisualisationGroup {
  ACO,
  EMC,
  HMP,
  MCH,
  PHS,
  RPH,
  SDD,
  SPD,
  SSD,
  TOF,
  TPC,
  TRD,
  RND,
  VSD,
  NvisualisationGroups
};

const std::string gVisualisationGroupName[NvisualisationGroups] = {
  "ACO",
  "EMC",
  "HMP",
  "MCH",
  "PHS",
  "RPH",
  "SDD",
  "SPD",
  "SSD",
  "TOF",
  "TPC",
  "TRD",
  "RND"
};

enum EVisualisationDataType {
  Raw,       ///< Raw data
  Hits,      ///< Hits
  Digits,    ///< Digits
  Clusters,  ///< Reconstructed clusters (RecPoints)
  ESD,       ///< Event Summary Data
  AOD,       ///< Analysis Object Data
  NoData,    ///< no data was loaded
  NdataTypes ///< number of supported data types
};

const std::string gDataTypeNames[NdataTypes] = {
    "Raw",
    "Hits",
    "Digits",
    "Clusters",
    "ESD",
    "AOD",
    "NoData"
};

  
}
}

#endif
