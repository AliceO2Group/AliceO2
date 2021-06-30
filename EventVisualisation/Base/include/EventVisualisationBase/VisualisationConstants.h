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

///
/// \file    VisualisationConstants.h
/// \author  Jeremi Niedziela
/// \author julian.myrcha@cern.ch
///

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_VISUALISATIONCONSTANTS_H
#define ALICE_O2_EVENTVISUALISATION_BASE_VISUALISATIONCONSTANTS_H

#include <string>

namespace o2
{
namespace event_visualisation
{

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
  ITS,
  TPC,
  TRD,
  RND,
  VSD,
  JSON,
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
  "ITS",
  "TPC",
  "TRD",
  "RND",
  "VSD",
  "JSON"};

const bool R2Visualisation[NvisualisationGroups] = {
  true,  //"ACO",
  true,  //"EMC",
  true,  //"HMP",
  true,  //"MCH",
  true,  //"PHS",
  true,  //"RPH",
  true,  //"SDD",
  true,  //"SPD",
  true,  //"SSD",
  true,  //"TOF",
  false, //ITS
  true,  //"TPC",
  true,  //"TRD",
  true,  //"RND",
  true,  //"VSD"
  true   //"JSON"
};

const bool R3Visualisation[NvisualisationGroups] = {
  true,  //"ACO",
  true,  //"EMC",
  true,  //"HMP",
  false, //"MCH",
  true,  //"PHS",
  false, //"RPH",
  false, //"SDD",
  false, //"SPD",
  false, //"SSD",
  false, //"TOF",
  true,  // ITS
  true,  //"TPC",
  false, //"TRD",
  false, //"RND",
  false, //"VSD"
  true   //"JSON"
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
  "NoData"};

} // namespace event_visualisation
} // namespace o2

#endif
