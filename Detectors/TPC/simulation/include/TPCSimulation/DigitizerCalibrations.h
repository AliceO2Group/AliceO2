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

/// \file DigitizerCalibrations.h
/// \brief Calibration objects of the TPC digitizer, shared read-only between digitizers

#ifndef ALICEO2_TPC_DIGITIZERCALIBRATIONS_H_
#define ALICEO2_TPC_DIGITIZERCALIBRATIONS_H_

#include "TPCBase/CalDet.h"
#include <memory>

namespace o2::tpc
{

/// Calibration objects of the TPC digitizer, read from the CDBInterface by one thread and then only read
struct DigitizerCalibrations {
  const CalPad* gainMap = nullptr;
  const CalPad* pedestals = nullptr;
  const CalPad* pedestalsCRU = nullptr;
  const CalPad* noise = nullptr;
  const CalPad* zeroSuppression = nullptr;
  const CalPad* padParams[3] = {nullptr, nullptr, nullptr}; ///< ion tail fraction, ion tail slope and common mode k-values per pad, if used
  std::unique_ptr<const CalDet<bool>> deadMap;              ///< own copy of the dead channel map, if used, since the CDBInterface rebuilds it on each access
  bool cmcEnabled = false;                                  ///< common mode correction enabled in the FEE config (DigiMode Auto)

  /// Read the calibration objects from the CDBInterface, which is not thread safe
  static std::shared_ptr<const DigitizerCalibrations> fetch();
};

} // namespace o2::tpc

#endif
