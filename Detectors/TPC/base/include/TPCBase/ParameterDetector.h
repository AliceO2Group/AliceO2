// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ParameterDetector.h
/// \brief Definition of the parameter class for the detector
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_ParameterDetector_H_
#define ALICEO2_TPC_ParameterDetector_H_

#include <array>
#include "DataFormatsTPC/Defs.h"
#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"

namespace o2
{
namespace tpc
{

struct ParameterDetector : public o2::conf::ConfigurableParamHelper<ParameterDetector> {

  float TPClength = 250.f;     ///< Length of the TPC [cm]
  float PadCapacitance = 0.1f; ///< Capacitance of a single pad [pF]
  TimeBin TmaxTriggered = 550; ///< Maximum time bin in case of triggered readout mode

  O2ParamDef(ParameterDetector, "TPCDetParam");
};
} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_ParameterDetector_H_
