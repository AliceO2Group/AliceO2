// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file FastSimModel.h
/// \brief CCDB object holding the ZDC fast simulation ONNX model together with its usage parameters

#ifndef O2_ZDC_FASTSIMMODEL_H_
#define O2_ZDC_FASTSIMMODEL_H_

#include <Rtypes.h>
#include <string>
#include <vector>

class TParticle;

namespace o2
{
namespace zdc
{

/// The fast simulation maps a spectator nucleon (given by its generator level kinematics)
/// directly to the number of photons seen by the 5 ZN and the 5 ZP channels.
///
/// ONNX signature expected by the code:
///  - input  "cond_phys" [N, NConditions] : PDG, E, vx, vy, vz, px, py, pz (GeV, cm, GeV/c; raw, unscaled)
///  - input  "x0"        [N, NOutputs]    : normal distributed random numbers
///  - input  "u"         [N, NOutputs]    : uniform random numbers in [0,1) used by the model for the stochastic
///                                          rounding of the output (optional: only if inputNameUniform is not empty)
///  - output              [N, NOutputs]    : photons in ZN (common, 1, 2, 3, 4) followed by ZP (common, 1, 2, 3, 4)
/// The model is side agnostic and trained for particles going towards positive z:
/// particles with pz < 0 are mirrored in z before the inference and the result is assigned to the C side.
struct FastSimModel {
  static constexpr int NConditions = 8;                 ///< PDG, E, vx, vy, vz, px, py, pz
  static constexpr int NChannelsPerCalo = 5;            ///< common + 4 towers
  static constexpr int NOutputs = 2 * NChannelsPerCalo; ///< ZN followed by ZP

  std::vector<char> onnx;                        ///< serialized ONNX model
  std::string inputNameConditions = "cond_phys"; ///< name of the input tensor with the particle kinematics
  std::string inputNameNoise = "x0";             ///< name of the input tensor with the gaussian noise
  std::string inputNameUniform = "u";            ///< name of the input tensor with uniform random numbers (empty: not used by the model)
  std::string outputName = "x_phys";             ///< name of the output tensor (empty: first output of the model)
  std::string version = "";                      ///< free text identifying the model

  // domain of validity of the model: everything outside is ignored by the fast simulation
  std::vector<int> pdgCodes = {2212, 2112}; ///< supported particle species
  float eMin = 0.f;                         ///< minimal energy (GeV)
  float eMax = 1.e9f;                       ///< maximal energy (GeV)
  float absEtaMin = 0.f;                    ///< minimal |pseudorapidity|
  float vtxRMax = 1.e9f;                    ///< maximal transverse distance of production vertex from nominal IP (cm)
  float vtxZMax = 1.e9f;                    ///< maximal |z| of production vertex from nominal IP (cm)

  /// load the ONNX model from a file; returns false on failure
  bool loadONNX(const std::string& fileName);

  /// is this particle to be treated by the model?
  bool accepts(const TParticle& p) const;

  void print() const;

  ClassDefNV(FastSimModel, 1);
};

} // namespace zdc
} // namespace o2

#endif
