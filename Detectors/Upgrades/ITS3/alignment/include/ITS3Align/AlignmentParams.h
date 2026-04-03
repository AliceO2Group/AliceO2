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
#ifndef ALICEO2_ITS3_ALIGNMENTPARAMS_H_
#define ALICEO2_ITS3_ALIGNMENTPARAMS_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "DetectorsBase/Propagator.h"

namespace o2::its3::align
{
struct AlignmentParams : public o2::conf::ConfigurableParamHelper<AlignmentParams> {
  // Track selection
  float minPt = 1.f;         // minimum pt required
  int minITSCls = 7;         // minimum number of ITS clusters
  float maxITSChi2Ndf = 1.2; // maximum ITS track chi2

  // propagation opt
  double maxSnp = 0.85;
  double maxStep = 2.0;
  // o2::base::PropagatorD::MatCorrType matCorrType = o2::base::PropagatorD::MatCorrType::USEMatCorrTGeo;
  o2::base::PropagatorD::MatCorrType corrType = o2::base::PropagatorD::MatCorrType::USEMatCorrLUT;

  bool useStableRef = true; // use input tracks as linearization point
  float minMS = 1e-6f;      //  minimum scattering to account for
  float maxChi2Ndf = 10;    // maximum Chi2/Ndf allowed for GBL fit

  // per chip extra error
  float extraClsErrY[6] = {0};
  float extraClsErrZ[6] = {0};

  // misalignment simulation
  bool doMisalignmentLeg = false;  // simulate Legendre deformation on ITS3 layers
  bool doMisalignmentRB = false;   // simulate rigid body misalignment on ITS3 layers
  bool doMisalignmentInex = false; // simulate in-extensional deformation on ITS3 layers
  std::string misAlgJson;          // JSON file with deformation and/or rigid body params

  // DOF configuration (JSON file defining which volumes have which DOFs)
  std::string dofConfigJson; // if empty, no DOFs are configured

  // Ridder options
  int ridderMaxExtrap = 10;
  double ridderRelIniStep[5] = {0.01, 0.01, 0.02, 0.02, 0.02};
  double ridderMaxIniStep[5] = {0.1, 0.1, 0.05, 0.05, 0.05};
  double ridderShrinkFac = 2.0;
  double ridderEps = 1e-16;

  // MillePede output
  std::string milleBinFile = "mp2data.bin";
  std::string milleConFile = "mp2con.txt";
  std::string milleParamFile = "mp2param.txt";
  std::string milleTreeFile = "mp2tree.txt";
  std::string milleResFile = "millepede.res";
  std::string milleResOutJson = "result.json";

  O2ParamDef(AlignmentParams, "ITS3AlignmentParams");
};
} // namespace o2::its3::align

#endif
