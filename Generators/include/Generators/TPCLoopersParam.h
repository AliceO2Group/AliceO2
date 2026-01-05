// Copyright 2024-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author M+Giacalone - September 2025

#ifndef ALICEO2_EVENTGEN_TPCLOOPERSPARAM_H_
#define ALICEO2_EVENTGEN_TPCLOOPERSPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace eventgen
{

/**
 ** a parameter class/struct to keep the settings of
 ** the tpc loopers event-generator and
 ** allow the user to modify them
 **/
struct GenTPCLoopersParam : public o2::conf::ConfigurableParamHelper<GenTPCLoopersParam> {
  bool loopersVeto = false; // if true, no loopers are generated
  std::string model_pairs = "ccdb://Users/m/mgiacalo/WGAN_ExtGenPair";  // ONNX model for e+e- pair production
  std::string model_compton = "ccdb://Users/m/mgiacalo/WGAN_ExtGenCompton"; // ONNX model for Compton scattering
  std::string poisson = "${O2_ROOT}/share/Generators/egconfig/poisson_params.csv"; // file with Poissonian parameters
  std::string gauss = "${O2_ROOT}/share/Generators/egconfig/gaussian_params.csv"; // file with Gaussian parameters
  std::string scaler_pair = "${O2_ROOT}/share/Generators/egconfig/ScalerPairParams.json"; // file with scaler parameters for e+e- pair production
  std::string scaler_compton = "${O2_ROOT}/share/Generators/egconfig/ScalerComptonParams.json"; // file with scaler parameters for Compton scattering
  std::string nclxrate = "ccdb://Users/m/mgiacalo/ClustersTrackRatio";                          // file with clusters/rate information per orbit
  std::string colsys = "PbPb";                                                                  // collision system  (PbPb or pp)
  int intrate = -1;                                                                             // Automatic IR from collision context if -1, else user-defined interaction rate in Hz
  bool flat_gas = true; // if true, the gas density is considered flat in the TPC volume
  unsigned int nFlatGasLoopers = 500;  // number of loopers to be generated per event in case of flat gas
  float fraction_pairs = 0.08; // fraction of loopers
  float multiplier[2] = {1., 1.}; // multiplier for pairs and compton loopers for Poissonian and Gaussian sampling
  unsigned int fixedNLoopers[2] = {1, 1}; // fixed number of loopers coming from pairs and compton electrons - valid if flat gas is false and both Poisson and Gaussian params files are empty
  float adjust_flatgas = 0.f; // adjustment for the number of flat gas loopers per orbit (in percentage, e.g. -0.1 = -10%) [-1, inf)]
  O2ParamDef(GenTPCLoopersParam, "GenTPCLoopers");
};

} // end namespace eventgen
} // end namespace o2

#endif // ALICEO2_EVENTGEN_TPCLOOPERSPARAM_H_
