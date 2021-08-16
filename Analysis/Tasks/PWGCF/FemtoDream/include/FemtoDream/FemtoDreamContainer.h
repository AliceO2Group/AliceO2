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
#include "TMath.h"
#include "TDatabasePDG.h"

using namespace o2::framework;

namespace o2::analysis::femtoDream
{

namespace femtoDreamContainer
{
/// Femtoscopic observable to be computed
enum Observable { kstar ///< kstar
};

/// Type of the event processind
enum EventType { same, ///< Pair from same event
                 mixed ///< Pair from mixed event
};
}; // namespace femtoDreamContainer

/// \class FemtoDreamContainer
/// \brief Container for all histogramming related to the correlation function. The two
/// particles of the pair are passed here, and the correlation function and QA histograms
/// are filled according to the specified observable
/// \tparam eventType Type of the event (same/mixed)
/// \tparam obs Observable to be computed (k*/Q_inv/...)
template <femtoDreamContainer::EventType eventType, femtoDreamContainer::Observable obs>
class FemtoDreamContainer
{
 public:
  /// Destructor
  virtual ~FemtoDreamContainer() = default;

  /// Initializes histograms for the task
  /// \tparam T Type of the configurable for the axis configuration
  /// \param registry Histogram registry to be passed
  /// \param kstarBins k* binning for the histograms
  /// \param multBins multiplicity binning for the histograms
  /// \param kTBins kT binning for the histograms
  /// \param mTBins mT binning for the histograms
  template <typename T>
  void init(HistogramRegistry* registry, T& kstarBins, T& multBins, T& kTBins, T& mTBins)
  {
    mHistogramRegistry = registry;
    std::string femtoObs;
    if constexpr (mFemtoObs == femtoDreamContainer::Observable::kstar) {
      femtoObs = "#it{k*} (GeV/#it{c})";
    }
    std::vector<double> tmpVecMult = multBins;
    framework::AxisSpec multAxis = {tmpVecMult, "Multiplicity"};
    framework::AxisSpec femtoObsAxis = {kstarBins, femtoObs.c_str()};
    framework::AxisSpec kTAxis = {kTBins, "#it{k}_{T} (GeV/#it{c})"};
    framework::AxisSpec mTAxis = {mTBins, "#it{m}_{T} (GeV/#it{c}^{2})"};

    std::string folderName = static_cast<std::string>(mFolderSuffix[mEventType]);
    mHistogramRegistry->add((folderName + "relPairDist").c_str(), ("; " + femtoObs + "; Entries").c_str(), kTH1F, {femtoObsAxis});
    mHistogramRegistry->add((folderName + "relPairkT").c_str(), "; #it{k}_{T} (GeV/#it{c}); Entries", kTH1F, {kTAxis});
    mHistogramRegistry->add((folderName + "relPairkstarkT").c_str(), ("; " + femtoObs + "; #it{k}_{T} (GeV/#it{c})").c_str(), kTH2F, {femtoObsAxis, kTAxis});
    mHistogramRegistry->add((folderName + "relPairkstarmT").c_str(), ("; " + femtoObs + "; #it{m}_{T} (GeV/#it{c}^{2})").c_str(), kTH2F, {femtoObsAxis, mTAxis});
    mHistogramRegistry->add((folderName + "relPairkstarMult").c_str(), ("; " + femtoObs + "; Multiplicity").c_str(), kTH2F, {femtoObsAxis, multAxis});
  }

  /// Set the PDG codes of the two particles involved
  /// \param pdg1 PDG code of particle one
  /// \param pdg2 PDG code of particle two
  void setPDGCodes(const int pdg1, const int pdg2)
  {
    mMassOne = TDatabasePDG::Instance()->GetParticle(pdg1)->Mass();
    mMassTwo = TDatabasePDG::Instance()->GetParticle(pdg2)->Mass();
  }

  /// Pass a pair to the container and compute all the relevant observables
  /// \tparam T type of the femtodreamparticle
  /// \param part1 Particle one
  /// \param part2 Particle two
  /// \param mult Multiplicity of the event
  template <typename T>
  void setPair(T const& part1, T const& part2, const int mult)
  {
    float femtoObs;
    if constexpr (mFemtoObs == femtoDreamContainer::Observable::kstar) {
      femtoObs = FemtoDreamMath::getkstar(part1, mMassOne, part2, mMassTwo);
    }
    const float kT = FemtoDreamMath::getkT(part1, mMassOne, part2, mMassTwo);
    const float mT = FemtoDreamMath::getmT(part1, mMassOne, part2, mMassTwo);

    if (mHistogramRegistry) {
      mHistogramRegistry->fill(HIST(mFolderSuffix[mEventType]) + HIST("relPairDist"), femtoObs);
      mHistogramRegistry->fill(HIST(mFolderSuffix[mEventType]) + HIST("relPairkT"), kT);
      mHistogramRegistry->fill(HIST(mFolderSuffix[mEventType]) + HIST("relPairkstarkT"), femtoObs, kT);
      mHistogramRegistry->fill(HIST(mFolderSuffix[mEventType]) + HIST("relPairkstarmT"), femtoObs, mT);
      mHistogramRegistry->fill(HIST(mFolderSuffix[mEventType]) + HIST("relPairkstarMult"), femtoObs, mult);
    }
  }

 protected:
  HistogramRegistry* mHistogramRegistry = nullptr;                                    ///< For QA output
  static constexpr std::string_view mFolderSuffix[2] = {"SameEvent/", "MixedEvent/"}; ///< Folder naming for the output according to mEventType
  static constexpr femtoDreamContainer::Observable mFemtoObs = obs;                   ///< Femtoscopic observable to be computed (according to femtoDreamContainer::Observable)
  static constexpr int mEventType = eventType;                                        ///< Type of the event (same/mixed, according to femtoDreamContainer::EventType)
  float mMassOne = 0.f;                                                               ///< PDG mass of particle 1
  float mMassTwo = 0.f;                                                               ///< PDG mass of particle 2
};

} // namespace o2::analysis::femtoDream

#endif /* ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMCONTAINER_H_ */
