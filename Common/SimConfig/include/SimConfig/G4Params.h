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

#ifndef O2_SIMCONFIG_G4PARAM_H_
#define O2_SIMCONFIG_G4PARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace conf
{

// enumerating the possible G4 physics settings
enum class EG4Physics {
  kFTFP_BERT_optical = 0,             /* just ordinary */
  kFTFP_BERT_optical_biasing = 1,     /* with biasing enabled */
  kFTFP_INCLXX_optical = 2,           /* special INCL++ version */
  kFTFP_BERT_HP_optical = 3,          /* enable low energy neutron transport */
  kFTFP_BERT_EMV_optical = 4,         /* just ordinary with faster electromagnetic physics */
  kFTFP_BERT_EMV_optical_biasing = 5, /* with biasing enabled with faster electromagnetic physics */
  kFTFP_INCLXX_EMV_optical = 6,       /* special INCL++ version */
  kFTFP_BERT_EMV_HP_optical = 7,      /* enable low energy neutron transport */
  kUSER = 8                           /* allows to give own string combination */
};

// enumerating possible geometry navigation modes
// (understanding that geometry description is always done with TGeo)
enum class EG4Nav {
  kTGeo = 0,   /* navigate with TGeo */
  kG4 = 1,     /* navigate with G4 native geometry */
  kVecGeom = 2 /* navigate with VecGeom, on the G4 geometry built from TGeo */
};

// the Geant4 navigator used with navmode kVecGeom
enum class EVecGeomNav {
  kRelocating = 0, /* relocates at the boundary locate, blocking the volume just left (default) */
  kPropagated = 1  /* adopts the state VecGeom propagated during the step; less work per crossing */
};

// parameters to influence the G4 engine
struct G4Params : public o2::conf::ConfigurableParamHelper<G4Params> {
  EG4Physics physicsmode = EG4Physics::kFTFP_BERT_EMV_optical; // default physics mode with which to configure G4

  std::string configMacroFile = ""; // a user provided g4Config.in file (otherwise standard one fill be taken)
  std::string userPhysicsList = ""; // possibility to directly give physics list as string

  EG4Nav navmode = EG4Nav::kTGeo; // geometry navigation mode (default TGeo)

  // Settings for navmode == kVecGeom; ignored otherwise.
  // which of the two VecGeom navigators
  EVecGeomNav vecgeomNavigator = EVecGeomNav::kRelocating;
  double vecgeomPushDepth = 1.e-9;      // cm; how far past a face, measured across it, a boundary
                                        // point is pushed before it is located
  bool vecgeomZeroSafety = false;       // answer zero to every safety query; conservative, but it
                                        // shortens steps and so changes the random history
  bool vecgeomFlattenAssemblies = true; // dissolve TGeo assemblies into their content when converting
                                        // to VecGeom; the Geant4 touchable keeps the assembly levels
  int vecgeomCheckRays = 0;             // if > 0, step this many rays out of the interaction point with
                                        // TGeo and VecGeom and report the volumes they enter differently
  int vecgeomCheckLocation = 0;         // if > 0, locate this many random points with both and report
                                        // the volumes they disagree on
  std::string vecgeomCheckVolumes = ""; // comma-separated volumes to cross-check by sampling inside
                                        // their placements

  std::string fluenceWeightFile = ""; // file containing the scoring weights (pdg, ekin, weight)
  std::string const& getPhysicsConfigString() const;

  bool g4scoring = false;
  bool g4fluenceweight = false;

  // Fast simulation. Empty fastSimModels (the default) disables the feature
  // entirely; see Detectors/gconfig/include/SimSetup/G4FastSimulation.h.
  std::string fastSimModels = "";   // comma-separated model names to activate
  std::string fastSimEnvelope = ""; // volume a model stands in for, e.g. AFaM; the media of its
                                    // subtree are collected automatically
  std::string fastSimRegions = "";  // optional explicit media, overriding the subtree walk
  float fastSimMinEnergy = 1.f;     // GeV; below this the detailed transport runs
  O2ParamDef(G4Params, "G4");
};

} // namespace conf
} // namespace o2

#endif /* O2_SIMCONFIG_G4PARAM_H_ */
