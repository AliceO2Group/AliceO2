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
/// \file Digitizer.h
/// \brief Definition of the ALICE3 TOF digitizer
/// \author Nicolò Jacazio, Università del Piemonte Orientale (IT)
/// \since 2026-03-17
///

#ifndef ALICEO2_IOTOF_DIGITIZER_H
#define ALICEO2_IOTOF_DIGITIZER_H

#include "ITSMFTSimulation/Hit.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsIOTOF/Digit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "IOTOFBase/GeometryTGeo.h"

namespace o2::iotof
{

class Digitizer
{
 public:
  void setDigits(std::vector<o2::iotof::Digit>* dig) { mDigits = dig; }
  void setMCLabels(o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mclb) { mMCLabels = mclb; }
  void setROFRecords(std::vector<o2::itsmft::ROFRecord>* rec) { mROFRecords = rec; }

  void init();

  /// Steer conversion of hits to digits
  void process(const std::vector<o2::itsmft::Hit>* hits, int evID, int srcID);

  // provide the common iotof::GeometryTGeo to access matrices and segmentation
  void setGeometry(const o2::iotof::GeometryTGeo* gm) { mGeometry = gm; }

 private:
  void processHit(const o2::itsmft::Hit& hit, uint32_t& maxFr, int evID, int srcID);

  const o2::iotof::GeometryTGeo* mGeometry = nullptr; ///< IOTOF geometry

  // std::vector<o2::iotof::ChipDigitsContainer> mChips; ///< Array of chips digits containers

  std::vector<o2::iotof::Digit>* mDigits = nullptr;                        //! output digits
  std::vector<o2::itsmft::ROFRecord>* mROFRecords = nullptr;               //! output ROF records
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mMCLabels = nullptr; //! output labels
};
} // namespace o2::iotof

#endif