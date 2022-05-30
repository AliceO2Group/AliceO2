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

/// \file DcsCcdbObjects.h
/// \brief Objects which are created from DCS DPs and stored in the CCDB
/// \author Ole Schmidt, ole.schmidt@cern.ch

#ifndef ALICEO2_DCSCCDBOBJECTSTRD_H
#define ALICEO2_DCSCCDBOBJECTSTRD_H

#include "DataFormatsTRD/Constants.h"
#include "Rtypes.h"
#include <array>

namespace o2
{
namespace trd
{

struct TRDDCSMinMaxMeanInfo {
  float minValue{0.f};  // min value seen by the TRD DCS processor
  float maxValue{0.f};  // max value seen by the TRD DCS processor
  float meanValue{0.f}; // mean value seen by the TRD DCS processor
  int nPoints{0};       // number of values seen by the TRD DCS processor

  void print() const;
  void addPoint(float value);

  ClassDefNV(TRDDCSMinMaxMeanInfo, 1);
};

} // namespace trd
} // namespace o2

#endif // ALICEO2_DCSCCDBOBJECTSTRD_H
