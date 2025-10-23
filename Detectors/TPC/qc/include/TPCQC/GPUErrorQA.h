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
/// @file   GPUErrorQA.h
/// @author Anton Riedel, anton.riedel@cern.ch
///

#ifndef AliceO2_TPC_QC_GPUERRORQA_H
#define AliceO2_TPC_QC_GPUERRORQA_H

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

// root includes

// o2 includes
// #include "DataFormatsTPC/Defs.h"

class TH1;
namespace o2::tpc::qc
{

/// @brief TPC QC task for errors from GPU reconstruction
///
/// This class is used to retrieve and visualize GPU errors
/// according to corresponding error code and location.
///
/// origin: TPC
/// @author Anton Riedel, anton.riedel@cern.ch
class GPUErrorQA
{
 public:
  /// \brief Constructor.
  GPUErrorQA() = default;

  /// process gpu error reported by the reconstruction workflow
  void processErrors(std::vector<std::array<uint32_t, 4>> errors);

  /// Initialize all histograms
  void initializeHistograms();

  /// Reset all histograms
  void resetHistograms();

  /// return histograms
  const std::unordered_map<std::string, std::unique_ptr<TH1>>& getMapHist() const { return mMapHist; }

  /// Dump results to a file
  void dumpToFile(std::string filename);

 private:
  std::unordered_map<std::string, std::unique_ptr<TH1>> mMapHist;

  ClassDefNV(GPUErrorQA, 2);
};
} // namespace o2::tpc::qc

#endif // AliceO2_TPC_QC_GPUERRORQA_H
