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

/// \file FemtoDreamContainer.h
/// \brief Definition of the FemtoDreamContainer
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de
/// \author Valentina Mantovani Sarti, valentina.mantovani-sarti@tum.de

#ifndef ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMCONTAINER_H_
#define ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMCONTAINER_H_

#include "Framework/HistogramRegistry.h"
#include "FemtoDreamMath.h"

#include "Math/Vector4D.h"
#include "TLorentzVector.h"
#include "TMath.h"
#include "TDatabasePDG.h"

namespace o2::analysis::femtoDream
{

namespace femtoDreamContainer
{
enum Observable { kstar };
}

/// \class FemtoDreamContainer
/// \brief Container for all histogramming related to the correlation function. The two
/// particles of the pair are passed here, and the correlation function and QA histograms
/// are filled according to the specified observable
class FemtoDreamContainer
{
 public:
  template <typename T>
  void init(o2::framework::HistogramRegistry* registry, femtoDreamContainer::Observable obs, T& kstarBins, T& multBins, T& kTBins, T& mTBins)
  {
    mHistogramRegistry = registry;
    std::string femtoObs;
    if (mFemtoObs == femtoDreamContainer::Observable::kstar) {
      femtoObs = "#it{k*} (GeV/#it{c})";
    }
    std::vector<double> tmpVecMult = multBins;
    framework::AxisSpec multAxis = {tmpVecMult, "Multiplicity"};
    framework::AxisSpec femtoObsAxis = {kstarBins, femtoObs.c_str()};
    framework::AxisSpec kTAxis = {kTBins, "#it{k}_{T} (GeV/#it{c})"};
    framework::AxisSpec mTAxis = {mTBins, "#it{m}_{T} (GeV/#it{c}^{2})"};

    mHistogramRegistry->add("relPairDist", ("; " + femtoObs + "; Entries").c_str(), o2::framework::kTH1F, {femtoObsAxis});
    mHistogramRegistry->add("relPairkT", "; #it{k}_{T} (GeV/#it{c}); Entries", o2::framework::kTH1F, {kTAxis});
    mHistogramRegistry->add("relPairkstarkT", ("; " + femtoObs + "; #it{k}_{T} (GeV/#it{c})").c_str(), o2::framework::kTH2F, {femtoObsAxis, kTAxis});
    mHistogramRegistry->add("relPairkstarmT", ("; " + femtoObs + "; #it{m}_{T} (GeV/#it{c}^{2})").c_str(), o2::framework::kTH2F, {femtoObsAxis, mTAxis});
    mHistogramRegistry->add("relPairkstarMult", ("; " + femtoObs + "; Multiplicity").c_str(), o2::framework::kTH2F, {femtoObsAxis, multAxis});
  }

  void setPDGCodes(const int pdg1, const int pdg2)
  {
    mMassOne = TDatabasePDG::Instance()->GetParticle(pdg1)->Mass();
    mMassTwo = TDatabasePDG::Instance()->GetParticle(pdg2)->Mass();
  }

  template <typename T>
  void setPair(T const& part1, T const& part2, const int mult)
  {
    float femtoObs;
    if (mFemtoObs == femtoDreamContainer::Observable::kstar) {
      femtoObs = FemtoDreamMath::getkstar(part1, mMassOne, part2, mMassTwo);
    }
    const float kT = FemtoDreamMath::getkT(part1, mMassOne, part2, mMassTwo);
    const float mT = FemtoDreamMath::getmT(part1, mMassOne, part2, mMassTwo);

    if (mHistogramRegistry) {
      mHistogramRegistry->fill(HIST("relPairDist"), femtoObs);
      mHistogramRegistry->fill(HIST("relPairkT"), kT);
      mHistogramRegistry->fill(HIST("relPairkstarkT"), femtoObs, kT);
      mHistogramRegistry->fill(HIST("relPairkstarmT"), femtoObs, mT);
      mHistogramRegistry->fill(HIST("relPairkstarMult"), femtoObs, mult);
    }
  }

 protected:
  femtoDreamContainer::Observable mFemtoObs = femtoDreamContainer::Observable::kstar;
  o2::framework::HistogramRegistry* mHistogramRegistry = nullptr;

  float mMassOne = 0.f;
  float mMassTwo = 0.f;
};

} // namespace o2::analysis::femtoDream

#endif /* ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMCONTAINER_H_ */
