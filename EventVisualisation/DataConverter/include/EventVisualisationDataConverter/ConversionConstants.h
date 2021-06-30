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
/// \file    ConversionConstants.h
/// \author  Jeremi Niedziela
/// \author  Maciej Grochowicz
///

#ifndef ALICE_O2_DATACONVERTER_CONVERSIONCONSTANTS_H
#define ALICE_O2_DATACONVERTER_CONVERSIONCONSTANTS_H

#include <string>

///
/// Which visualisation import method was used to obtain track
///
enum ETrackSource {
  CosmicSource,
  TPCSource,
  ITSSource
};

enum ETrackType {
  Standard,

  KinkMother,
  KinkDaughter,

  V0NegativeDaughter,
  V0PositiveDaughter,
  V0Mother,

  CascadePrimaryMother,
  CascadePrimaryDaughter,
  CascadeSecondaryMother,
  CascadeNegativeDaughter,
  CascadePositiveDaughter,

  MuonMatched,
  MuonNotMatched,
  MuonGhost
};

const int nDetectorTypes = 23;

const std::string gDetectorTypes[nDetectorTypes] = {
  "Invalid Layer",
  "First Layer",

  "SPD1",
  "SPD2",
  "SDD1",
  "SDD2",
  "SSD1",
  "SSD2",

  "TPC1",
  "TPC2",

  "TRD1",
  "TRD2",
  "TRD3",
  "TRD4",
  "TRD5",
  "TRD6",

  "TOF",

  "PHOS1",
  "PHOS2",

  "HMPID",
  "MUON",
  "EMCAL",
  "LastLayer"};

const int nTrackTypes = 14;

const std::string gTrackTypes[nTrackTypes] = {
  "standard",

  "kink_mother",
  "kink_daughter",

  "V0_negative_daughter",
  "V0_positive_daughter",
  "V0_mother",

  "cascade_primary_mother",
  "cascade_primary_daughter",
  "cascade_secondary_mother",
  "cascade_negative_daughter",
  "cascade_positive_daughter",

  "muon_matched",
  "muon_not_matched",
  "muon_ghost"};

#endif //ALICE_O2_DATACONVERTER_CONVERSIONCONSTANTS_H
