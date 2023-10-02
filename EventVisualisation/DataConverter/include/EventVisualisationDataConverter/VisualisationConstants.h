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
  ITS,
  TPC,
  TRD,
  TOF,
  MFT,
  MCH,
  MID,
  EMC,
  PHS,
  CPV,
  HMP,
  NvisualisationGroups
};

const std::string gVisualisationGroupName[NvisualisationGroups] = {
  "ITS",
  "TPC",
  "TRD",
  "TOF",
  "MFT",
  "MCH",
  "MID",
  "EMC",
  "PHS",
  "CPV",
  "HMP"};

const bool R3Visualisation[NvisualisationGroups] = {
  true, //"ITS",
  true, //"TPC",
  true, //"TRD",
  true, //"TOF",
  true, // "MFT"
  true, //"MCH",
  true, //"MID",
  true, //"EMC",
  true, //"PHS",
  true, //"CPV"
  true, //"HMP"
};

enum EVisualisationDataType {
  Clusters,     ///< Reconstructed clusters (RecPoints)
  Tracks,       ///< Event Summary Data
  Calorimeters, ///< Calorimeters
  NdataTypes    ///< number of supported data types
};

const std::string gDataTypeNames[NdataTypes] = {
  "Clusters",
  "Tracks",
  "Calorimeters"};

static int findGroupIndex(const std::string& name)
{
  for (int i = 0; i < NvisualisationGroups; i++) {
    if (name == gVisualisationGroupName[i]) {
      return i;
    }
  }
  return -1;
};

} // namespace event_visualisation
} // namespace o2

#endif
