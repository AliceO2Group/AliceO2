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
/// \file    VisualisationEventSerializer.h
/// \author  Julian Myrcha
///

#ifndef O2EVE_STATISTIC_H
#define O2EVE_STATISTIC_H

#include <string>
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "rapidjson/document.h"

namespace o2
{
namespace event_visualisation
{
class VisualisationEvent;

const std::string gDetectorSources[o2::dataformats::GlobalTrackID::NSources] = {
  "ITS", // standalone detectors
  "TPC",
  "TRD",
  "TOF",
  "PHS", // FIXME Not sure PHS ... FDD should be kept here, at the moment
  "CPV", // they are here for completeness
  "EMC",
  "HMP",
  "MFT",
  "MCH",
  "MID",
  "ZDC",
  "FT0",
  "FV0",
  "FDD",
  "ITSTPC", // 2-detector tracks
  "TPCTOF",
  "TPCTRD",
  "MFTMCH",
  "ITSTPCTRD", // 3-detector tracks
  "ITSTPCTOF",
  "TPCTRDTOF",
  "MFTMCHMID",
  "ITSTPCTRDTOF", // full barrel track
  "ITSAB",        // ITS AfterBurner tracklets
  "CTP",
  //
  "MCHMID"};

class Statistic
{
 public:
  rapidjson::Document tree;
  rapidjson::Document::AllocatorType* allocator;

  Statistic();
  void save(std::string fileName);

  void toFile(const VisualisationEvent::Statistic& statistic);
};

} // namespace event_visualisation
} // namespace o2

#endif // O2EVE_STATISTIC_H
