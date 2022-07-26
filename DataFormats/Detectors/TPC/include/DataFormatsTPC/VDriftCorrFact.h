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

/// \file VDriftCorrFact.h
/// \brief calibration data from laser track calibration
///
/// This class holds the calibration output data ITS/TPC tgl difference calibration
///
/// \author ruben.shahoyan@cern.ch

#ifndef AliceO2_TPC_VDRIFT_CORRFACT_H
#define AliceO2_TPC_VDRIFT_CORRFACT_H

#include "GPUCommonRtypes.h"

namespace o2::tpc
{

struct VDriftCorrFact {
  long firstTime{};       ///< first time stamp of processed TFs
  long lastTime{};        ///< last time stamp of processed TFs
  long creationTime{};    ///< time of creation
  float corrFact{1.0};    ///< drift velocity correction factor (multiplicative)
  float corrFactErr{0.0}; ///< stat error of correction factor
  float refVDrift{0.};    ///< reference vdrift for which factor was extracted

  float getVDrift() const { return refVDrift * corrFact; }
  float getVDriftError() const { return refVDrift * corrFactErr; }

  // renormalize reference and correction either to provided new reference (if >0) or to correction 1 wrt current reference
  void normalize(float newVRef = 0.f)
  {
    if (newVRef == 0.f) {
      newVRef = refVDrift * corrFact;
    }
    float fact = refVDrift / newVRef;
    refVDrift = newVRef;
    corrFactErr *= fact;
    corrFact *= fact;
  }

  ClassDefNV(VDriftCorrFact, 1);
};

} // namespace o2::tpc
#endif
