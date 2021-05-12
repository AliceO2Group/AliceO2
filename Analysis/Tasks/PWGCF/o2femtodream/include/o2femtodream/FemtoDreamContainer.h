// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file FemtoDreamContainer.h
/// \brief Definition of the FemtoDreamContainer
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMCONTAINER_H_
#define ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMCONTAINER_H_

#include "Math/Vector4D.h"
#include "TLorentzVector.h"
#include "TMath.h"

#include "Framework/HistogramRegistry.h"

#include <iostream>

namespace o2::analysis
{
namespace femtoDream
{

/// \class FemtoDreamContainer
/// \brief Container for all histogramming related to the correlation function. The two
/// particles of the pair are passed here, and the correlation function and QA histograms
/// are filled according to the specified observable

class FemtoDreamContainer
{

 public:
  enum class Observable { kstar };

  void init(o2::framework::HistogramRegistry* registry, Observable obs = Observable::kstar);

  void setMasses(const float m1, const float m2)
  {
    mMassOne = m1;
    mMassTwo = m2;
  }

  template <typename T>
  void setPair(o2::framework::HistogramRegistry* registry, T& part1, T& part2);

  template <typename T>
  static float getkstar(const T& part1, const float mass1, const T& part2, const float mass2)
  {
    TLorentzVector vecpart1, vecpart2;
    vecpart1.SetXYZM(part1.pt(), part1.eta(), part1.phi(), mass1);
    vecpart2.SetXYZM(part2.pt(), part2.eta(), part2.phi(), mass2);
    return getkstar(vecpart1, vecpart2);
    //    return getkstar({part1.pt(), part1.eta(), part1.phi(), mass1}, {part2.pt(), part2.eta(), part2.phi(), mass2});
  }

  /// \todo improve performance of the computation by replacing TLorentzVector
  //  static float getkstar(const ROOT::Math::PtEtaPhiMVector& part1, const ROOT::Math::PtEtaPhiMVector& part2);
  static float getkstar(const TLorentzVector& part1, const TLorentzVector& part2);

 protected:
  Observable mFemtoObs = Observable::kstar;

  float mMassOne = 0.f;
  float mMassTwo = 0.f;
};

void FemtoDreamContainer::init(o2::framework::HistogramRegistry* registry, Observable obs)
{
  std::string femtoObs;
  if (mFemtoObs == Observable::kstar) {
    femtoObs = "#it{k}^{*} (GeV/#it{c})";
  }
  registry->add("relPairDist", ("; " + femtoObs + "; Entries").c_str(), o2::framework::kTH1F, {{5000, 0, 5}});
}

template <typename T>
inline void FemtoDreamContainer::setPair(o2::framework::HistogramRegistry* registry, T& part1, T& part2)
{
  float femtoObs;
  if (mFemtoObs == Observable::kstar) {
    femtoObs = getkstar(part1, mMassOne, part2, mMassTwo);
  }
  registry->fill(HIST("relPairDist"), femtoObs);
}

//inline float FemtoDreamContainer::getkstar(const ROOT::Math::PtEtaPhiMVector& part1, const ROOT::Math::PtEtaPhiMVector& part2)
inline float FemtoDreamContainer::getkstar(const TLorentzVector& part1, const TLorentzVector& part2)
{
  TLorentzVector trackSum = part1 + part2;

  float beta = trackSum.Beta();
  float betax = beta * std::cos(trackSum.Phi()) * std::sin(trackSum.Theta());
  float betay = beta * std::sin(trackSum.Phi()) * std::sin(trackSum.Theta());
  float betaz = beta * std::cos(trackSum.Theta());

  TLorentzVector PartOneCMS = part1;
  TLorentzVector PartTwoCMS = part2;

  PartOneCMS.Boost(-betax, -betay, -betaz);
  PartTwoCMS.Boost(-betax, -betay, -betaz);

  TLorentzVector trackRelK = PartOneCMS - PartTwoCMS;

  return 0.5 * trackRelK.P();
}

} /* namespace femtoDream */
} /* namespace o2::analysis */

#endif /* ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMCONTAINER_H_ */
