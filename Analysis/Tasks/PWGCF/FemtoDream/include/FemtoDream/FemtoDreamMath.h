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

/// \file FemtoDreamMath.h
/// \brief Definition of the FemtoDreamMath Container for math calculations of quantities related to pairs
/// \author Valentina Mantovani Sarti, TU München, valentina.mantovani-sarti@tum.de

#ifndef ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMMATH_H_
#define ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMMATH_H_

#include "Math/Vector4D.h"
#include "Math/Boost.h"
#include "TLorentzVector.h"
#include "TMath.h"

#include <iostream>

namespace o2::analysis
{
namespace femtoDream
{

/// \class FemtoDreamMath
/// \brief Container for math calculations of quantities related to pairs

class FemtoDreamMath
{
 public:
  template <typename T>
  static float getkstar(const T& part1, const float mass1, const T& part2, const float mass2)
  {
    ROOT::Math::PtEtaPhiMVector vecpart1(part1.pt(), part1.eta(), part1.phi(), mass1);
    ROOT::Math::PtEtaPhiMVector vecpart2(part2.pt(), part2.eta(), part2.phi(), mass2);

    ROOT::Math::PtEtaPhiMVector trackSum = vecpart1 + vecpart2;

    float beta = trackSum.Beta();
    float betax = beta * std::cos(trackSum.Phi()) * std::sin(trackSum.Theta());
    float betay = beta * std::sin(trackSum.Phi()) * std::sin(trackSum.Theta());
    float betaz = beta * std::cos(trackSum.Theta());

    ROOT::Math::PxPyPzMVector PartOneCMS(vecpart1);
    ROOT::Math::PxPyPzMVector PartTwoCMS(vecpart2);

    ROOT::Math::Boost boostPRF = ROOT::Math::Boost(-betax, -betay, -betaz);
    PartOneCMS = boostPRF(PartOneCMS);
    PartTwoCMS = boostPRF(PartTwoCMS);

    ROOT::Math::PxPyPzMVector trackRelK = PartOneCMS - PartTwoCMS;
    return 0.5 * trackRelK.P();
  }

  template <typename T>
  static float getkT(const T& part1, const float mass1, const T& part2, const float mass2)
  {
    ROOT::Math::PtEtaPhiMVector vecpart1(part1.pt(), part1.eta(), part1.phi(), mass1);
    ROOT::Math::PtEtaPhiMVector vecpart2(part2.pt(), part2.eta(), part2.phi(), mass2);

    ROOT::Math::PtEtaPhiMVector trackSum = vecpart1 + vecpart2;
    return 0.5 * trackSum.Pt();
  }

  template <typename T>
  static float getmT(const T& part1, const float mass1, const T& part2, const float mass2)
  {
    ROOT::Math::PtEtaPhiMVector vecpart1(part1.pt(), part1.eta(), part1.phi(), mass1);
    ROOT::Math::PtEtaPhiMVector vecpart2(part2.pt(), part2.eta(), part2.phi(), mass2);

    ROOT::Math::PtEtaPhiMVector trackSum = vecpart1 + vecpart2;
    float averageMass = 0.5 * (mass1 + mass2);
    return std::sqrt(std::pow(getkT(part1, mass1, part2, mass2), 2.) + std::pow(averageMass, 2.));
  }
};

} /* namespace femtoDream */
} /* namespace o2::analysis */

#endif /* ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMMATH_H_ */
