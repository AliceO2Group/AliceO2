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
/// @file   DCSPTemperature.h
/// @author Thomas Klemenz, thomas.klemenz@tum.de
///

#ifndef AliceO2_TPC_QC_DCSPTEMPERATURE_H
#define AliceO2_TPC_QC_DCSPTEMPERATURE_H

#include <vector>

// o2 includes
#include "DataFormatsTPC/DCS.h"

class TCanvas;

namespace o2::tpc::qc
{

/// @brief This class helps visualizing the data from the temperature sensors inside the TPC
///
/// origin: TPC
/// @author Thomas Klemenz, thomas.klemenz@tum.de
class DCSPTemperature
{
 public:
  /// default constructor
  DCSPTemperature() = default;

  /// destructor
  ~DCSPTemperature();

  /// initializes the TCanvases that will later be shown on the QCG
  void initializeCanvases();

  /// fill graphs with temperature data and populate the canvases
  void processData(const std::vector<std::unique_ptr<o2::tpc::dcs::Temperature>>& data);

  /// returns the canvases showing the temperature data
  std::vector<TCanvas*>& getCanvases() { return mCanVec; };

 private:
  std::vector<TCanvas*> mCanVec{}; // holds the canvases which are to be published in the QCG

  ClassDefNV(DCSPTemperature, 1)
};
} // namespace o2::tpc::qc

#endif