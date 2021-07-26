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

/// \file FemtoDreamParticleHisto.h
/// \brief FemtoDreamParticleHisto - Histogram class for tracks, V0s and cascades
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ANALYSIS_TASKS_PWGCF_O2FEMTODREAM_INCLUDE_O2FEMTODREAM_FEMTODREAMPARTICLEHISTO_H_
#define ANALYSIS_TASKS_PWGCF_O2FEMTODREAM_INCLUDE_O2FEMTODREAM_FEMTODREAMPARTICLEHISTO_H_

#include "FemtoDerived.h"
#include "Framework/HistogramRegistry.h"

using namespace o2::framework;

namespace o2::analysis::femtoDream
{

/// \class FemtoDreamParticleHisto
/// \brief Class for histogramming particle properties
/// \tparam particleType Type of the particle (Track/V0/Cascade/...)
/// \tparam suffixType (optional) Takes care of the suffix for the folder name in case of analyses of pairs of the same kind (T-T, V-V, C-C)
template <o2::aod::femtodreamparticle::ParticleType particleType, int suffixType = 0>
class FemtoDreamParticleHisto
{
 public:
  /// Destructor
  virtual ~FemtoDreamParticleHisto() = default;

  /// Initalization of the QA histograms
  /// \param registry HistogramRegistry
  void init(HistogramRegistry* registry)
  {
    if (registry) {
      mHistogramRegistry = registry;
      /// The folder names are defined by the type of the object and the suffix (if applicable)
      std::string folderName = static_cast<std::string>(o2::aod::femtodreamparticle::ParticleTypeName[mParticleType]);
      folderName += static_cast<std::string>(mFolderSuffix[mFolderSuffixType]);

      /// Histograms of the kinematic properties
      mHistogramRegistry->add((folderName + "/pThist").c_str(), "; #it{p}_{T} (GeV/#it{c}); Entries", kTH1F, {{1000, 0, 10}});
      mHistogramRegistry->add((folderName + "/etahist").c_str(), "; #eta; Entries", kTH1F, {{1000, -1, 1}});
      mHistogramRegistry->add((folderName + "/phihist").c_str(), "; #phi; Entries", kTH1F, {{1000, 0, 2. * M_PI}});

      /// Particle-type specific histograms
      if constexpr (mParticleType == o2::aod::femtodreamparticle::ParticleType::kTrack) {
        /// Track histograms
        mHistogramRegistry->add((folderName + "/dcaXYhist").c_str(), "; #it{p}_{T} (GeV/#it{c}); DCA_{xy} (cm)", kTH2F, {{100, 0, 10}, {501, -3, 3}});
      } else if constexpr (mParticleType == o2::aod::femtodreamparticle::ParticleType::kV0) {
        /// V0 histograms
        mHistogramRegistry->add((folderName + "/cpahist").c_str(), "; #it{p}_{T} (GeV/#it{c}); cos#alpha", kTH2F, {{100, 0, 10}, {500, 0, 1}});
      } else if constexpr (mParticleType == o2::aod::femtodreamparticle::ParticleType::kCascade) {
        /// Cascade histograms
      } else {
        LOG(FATAL) << "FemtoDreamParticleHisto: Histogramming for requested object not defined - quitting!";
      }
    }
  }

  /// Filling of the histograms
  /// \tparam T Data type of the particle
  /// \param part Particle
  template <typename T>
  void fillQA(T const& part)
  {
    if (mHistogramRegistry) {
      /// Histograms of the kinematic properties
      mHistogramRegistry->fill(HIST(o2::aod::femtodreamparticle::ParticleTypeName[mParticleType]) + HIST(mFolderSuffix[mFolderSuffixType]) + HIST("/pThist"), part.pt());
      mHistogramRegistry->fill(HIST(o2::aod::femtodreamparticle::ParticleTypeName[mParticleType]) + HIST(mFolderSuffix[mFolderSuffixType]) + HIST("/etahist"), part.eta());
      mHistogramRegistry->fill(HIST(o2::aod::femtodreamparticle::ParticleTypeName[mParticleType]) + HIST(mFolderSuffix[mFolderSuffixType]) + HIST("/phihist"), part.phi());

      /// Particle-type specific histograms
      if constexpr (mParticleType == o2::aod::femtodreamparticle::ParticleType::kTrack) {
        /// Track histograms
        mHistogramRegistry->fill(HIST(o2::aod::femtodreamparticle::ParticleTypeName[mParticleType]) + HIST(mFolderSuffix[mFolderSuffixType]) + HIST("/dcaXYhist"),
                                 part.pt(), part.tempFitVar());
      } else if constexpr (mParticleType == o2::aod::femtodreamparticle::ParticleType::kV0) {
        /// V0 histograms
        mHistogramRegistry->fill(HIST(o2::aod::femtodreamparticle::ParticleTypeName[mParticleType]) + HIST(mFolderSuffix[mFolderSuffixType]) + HIST("/cpahist"),
                                 part.pt(), part.tempFitVar());
      } else if constexpr (mParticleType == o2::aod::femtodreamparticle::ParticleType::kCascade) {
        /// Cascade histograms
      } else {
        LOG(FATAL) << "FemtoDreamParticleHisto: Histogramming for requested object not defined - quitting!";
      }
    }
  }

 private:
  HistogramRegistry* mHistogramRegistry;                                                   ///< For QA output
  static constexpr o2::aod::femtodreamparticle::ParticleType mParticleType = particleType; ///< Type of the particle under analysis
  static constexpr int mFolderSuffixType = suffixType;                                     ///< Counter for the folder suffix specified below
  static constexpr std::string_view mFolderSuffix[3] = {"", "_one", "_two"};               ///< Suffix for the folder name in case of analyses of pairs of the same kind (T-T, V-V, C-C)
};
} // namespace o2::analysis::femtoDream

#endif /* ANALYSIS_TASKS_PWGCF_O2FEMTODREAM_INCLUDE_O2FEMTODREAM_FEMTODREAMPARTICLEHISTO_H_ */
