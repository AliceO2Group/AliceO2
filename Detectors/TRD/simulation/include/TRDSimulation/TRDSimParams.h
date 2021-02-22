// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CONF_DIGIPARAMS_H_
#define O2_CONF_DIGIPARAMS_H_

// Global parameters for TRD simulation / digitization

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace trd
{

// Global parameters for TRD simulation / digitization
/*
  See https://github.com/AliceO2Group/AliceO2/blob/dev/Common/SimConfig/doc/ConfigurableParam.md
*/
struct TRDSimParams : public o2::conf::ConfigurableParamHelper<TRDSimParams> {
  int digithreads = 4;       // number of digitizer threads
  float maxMCStepSize = 0.1; // maximum size of MC steps
  bool doTR = true;          // switch for transition radiation
  O2ParamDef(TRDSimParams, "TRDSimParams");
};

} // namespace trd
} // namespace o2

#endif
