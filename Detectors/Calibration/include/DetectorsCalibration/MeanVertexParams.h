// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author Chiara.Zampolli@cern.ch

#ifndef ALICEO2_MEANVERTEX_PARAMS_H
#define ALICEO2_MEANVERTEX_PARAMS_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace calibration
{

// There are configurable params for TPC-ITS matching
struct MeanVertexParams : public o2::conf::ConfigurableParamHelper<MeanVertexParams> {

  int minEntries = 100;
  int nbinsX = 100;
  float rangeX = 1.f;
  int nbinsY = 100;
  float rangeY = 1.f;
  int nbinsZ = 100;
  float rangeZ = 20.f;
  int nSlots4SMA = 5;
  bool useFit = false;
  int tfPerSlot = 5;
  int maxTFdelay = 3;

  O2ParamDef(MeanVertexParams, "MeanVertexCalib");
};

} // namespace calibration
} // end namespace o2

#endif
