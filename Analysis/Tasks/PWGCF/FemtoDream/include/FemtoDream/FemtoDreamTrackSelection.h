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

/// \file FemtoDreamTrackCuts.h
/// \brief Definition of the FemtoDreamTrackCuts
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMTRACKSELECTION_H_
#define ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMTRACKSELECTION_H_

#include "FemtoDerived.h"
#include "FemtoDreamObjectSelection.h"

#include "ReconstructionDataFormats/PID.h"
#include "Framework/HistogramRegistry.h"
#include <cmath>

using namespace o2::framework;

namespace o2::analysis::femtoDream
{
namespace femtoDreamTrackSelection
{
/// The different selections this task is capable of doing
enum TrackSel { kSign,        ///< Sign of the track
                kpTMin,       ///< Min. p_T (GeV/c)
                kpTMax,       ///< Max. p_T (GeV/c)
                kEtaMax,      ///< Max. |eta|
                kTPCnClsMin,  ///< Min. number of TPC clusters
                kTPCfClsMin,  ///< Min. fraction of crossed rows/findable TPC clusters
                kTPCcRowsMin, ///< Min. number of crossed TPC rows
                kTPCsClsMax,  ///< Max. number of shared TPC clusters
                kDCAxyMax,    ///< Max. DCA_xy (cm)
                kDCAzMax,     ///< Max. DCA_z (cm)
                kDCAMin,      ///< Min. DCA_xyz (cm)
                kPIDnSigmaMax ///< Max. |n_sigma| for PID
};
} // namespace femtoDreamTrackSelection

/// \class FemtoDreamTrackCuts
/// \brief Cut class to contain and execute all cuts applied to tracks
class FemtoDreamTrackSelection : public FemtoDreamObjectSelection<float, femtoDreamTrackSelection::TrackSel>
{
 public:
  /// Initializes histograms for the task
  /// \tparam part Type of the particle for proper naming of the folders for QA
  /// \tparam cutContainerType Data type of the bit-wise container for the selections
  /// \param registry HistogramRegistry for QA output
  template <o2::aod::femtodreamparticle::ParticleType part, typename cutContainerType>
  void init(HistogramRegistry* registry);

  /// Passes the species to the task for which PID needs to be stored
  /// \tparam T Data type of the configurable passed to the functions
  /// \param pids Configurable with the species
  template <typename T>
  void setPIDSpecies(T& pids)
  {
    std::vector<int> tmpPids = pids; /// necessary due to some features of the configurable
    for (const o2::track::PID& pid : tmpPids) {
      mPIDspecies.push_back(pid);
    }
  }

  /// Computes the n_sigma for a track and a particle-type hypothesis in the TPC
  /// \tparam T Data type of the track
  /// \param track Track for which PID is evaluated
  /// \param pid Particle species for which PID is evaluated
  /// \return Value of n_{sigma, TPC}
  template <typename T>
  auto getNsigmaTPC(T const& track, o2::track::PID pid);

  /// Computes the n_sigma for a track and a particle-type hypothesis in the TOF
  /// \tparam T Data type of the track
  /// \param track Track for which PID is evaluated
  /// \param pid Particle species for which PID is evaluated
  /// \return Value of n_{sigma, TOF}
  template <typename T>
  auto getNsigmaTOF(T const& track, o2::track::PID pid);

  /// Checks whether the most open combination of all selection criteria is fulfilled
  /// \tparam T Data type of the track
  /// \param track Track
  /// \return Whether the most open combination of all selection criteria is fulfilled
  template <typename T>
  bool isSelectedMinimal(T const& track);

  /// Obtain the bit-wise container for the selections
  /// \todo For the moment, PID is separated from the other selections, hence instead of a single value an std::array of size two is returned
  /// \tparam cutContainerType Data type of the bit-wise container for the selections
  /// \tparam T Data type of the track
  /// \param track Track
  /// \return The bit-wise container for the selections, separately with all selection criteria, and the PID
  template <typename cutContainerType, typename T>
  std::array<cutContainerType, 2> getCutContainer(T const& track);

  /// Some basic QA histograms
  /// \tparam part Type of the particle for proper naming of the folders for QA
  /// \tparam T Data type of the track
  /// \param track Track
  template <o2::aod::femtodreamparticle::ParticleType part, typename T>
  void fillQA(T const& track);

  /// Helper function to obtain the name of a given selection criterion for consistent naming of the configurables
  /// \param iSel Track selection variable to be examined
  /// \param prefix Additional prefix for the name of the configurable
  /// \param suffix Additional suffix for the name of the configurable
  static std::string getSelectionName(femtoDreamTrackSelection::TrackSel iSel, std::string_view prefix = "", std::string_view suffix = "")
  {
    std::string outString = static_cast<std::string>(prefix);
    outString += static_cast<std::string>(mSelectionNames[iSel]);
    outString += suffix;
    return outString;
  }

  /// Helper function to obtain the helper string of a given selection criterion for consistent description of the configurables
  /// \param iSel Track selection variable to be examined
  /// \param prefix Additional prefix for the output of the configurable
  static std::string getSelectionHelper(femtoDreamTrackSelection::TrackSel iSel, std::string_view prefix = "")
  {
    std::string outString = static_cast<std::string>(prefix);
    outString += static_cast<std::string>(mSelectionHelper[iSel]);
    return outString;
  }

 private:
  std::vector<o2::track::PID> mPIDspecies; ///< All the particle species for which the n_sigma values need to be stored
  static constexpr std::string_view mSelectionNames[12] = {"Sign",
                                                           "PtMin",
                                                           "PtMax",
                                                           "EtaMax",
                                                           "TPCnClsMin",
                                                           "TPCfClsMin",
                                                           "TPCcRowsMin",
                                                           "TPCsClsMax",
                                                           "DCAxyMax",
                                                           "DCAzMax",
                                                           "DCAMin",
                                                           "PIDnSigmaMax"}; ///< Name of the different selections
  static constexpr std::string_view mSelectionHelper[12] = {"Sign of the track",
                                                            "Minimal pT (GeV/c)",
                                                            "Maximal pT (GeV/c)",
                                                            "Maximal eta",
                                                            "Minimum number of TPC clusters",
                                                            "Minimum fraction of crossed rows/findable clusters",
                                                            "Minimum number of crossed TPC rows",
                                                            "Maximal number of shared TPC cluster",
                                                            "Maximal DCA_xy (cm)",
                                                            "Maximal DCA_z (cm)",
                                                            "Minimal DCA (cm)",
                                                            "Maximal PID (nSigma)"}; ///< Helper information for the different selections
};                                                                                   // namespace femtoDream

template <o2::aod::femtodreamparticle::ParticleType part, typename cutContainerType>
void FemtoDreamTrackSelection::init(HistogramRegistry* registry)
{
  if (registry) {
    mHistogramRegistry = registry;
    fillSelectionHistogram<part>();

    /// \todo this should be an automatic check in the parent class
    int nSelections = getNSelections() + mPIDspecies.size() * (getNSelections(femtoDreamTrackSelection::kPIDnSigmaMax) - 1);
    if (8 * sizeof(cutContainerType) < nSelections) {
      LOG(FATAL) << "FemtoDreamTrackCuts: Number of selections to large for your container - quitting!";
    }

    std::string folderName = static_cast<std::string>(o2::aod::femtodreamparticle::ParticleTypeName[part]);
    mHistogramRegistry->add((folderName + "/pThist").c_str(), "; #it{p}_{T} (GeV/#it{c}); Entries", kTH1F, {{1000, 0, 10}});
    mHistogramRegistry->add((folderName + "/etahist").c_str(), "; #eta; Entries", kTH1F, {{1000, -1, 1}});
    mHistogramRegistry->add((folderName + "/phihist").c_str(), "; #phi; Entries", kTH1F, {{1000, 0, 2. * M_PI}});
    mHistogramRegistry->add((folderName + "/tpcnclshist").c_str(), "; TPC Cluster; Entries", kTH1F, {{163, 0, 163}});
    mHistogramRegistry->add((folderName + "/tpcfclshist").c_str(), "; TPC ratio findable; Entries", kTH1F, {{100, 0.5, 1.5}});
    mHistogramRegistry->add((folderName + "/tpcnrowshist").c_str(), "; TPC crossed rows; Entries", kTH1F, {{163, 0, 163}});
    mHistogramRegistry->add((folderName + "/tpcnsharedhist").c_str(), "; TPC shared clusters; Entries", kTH1F, {{163, 0, 163}});
    mHistogramRegistry->add((folderName + "/dcaXYhist").c_str(), "; #it{p}_{T} (GeV/#it{c}); DCA_{xy} (cm)", kTH2F, {{100, 0, 10}, {301, -1.5, 1.5}});
    mHistogramRegistry->add((folderName + "/dcaZhist").c_str(), "; #it{p}_{T} (GeV/#it{c}); DCA_{z} (cm)", kTH2F, {{100, 0, 10}, {301, -1.5, 1.5}});
    mHistogramRegistry->add((folderName + "/dcahist").c_str(), "; #it{p}_{T} (GeV/#it{c}); DCA (cm)", kTH1F, {{301, 0., 1.5}});
    mHistogramRegistry->add((folderName + "/tpcdEdx").c_str(), "; #it{p} (GeV/#it{c}); TPC Signal", kTH2F, {{100, 0, 10}, {1000, 0, 1000}});
    mHistogramRegistry->add((folderName + "/tofSignal").c_str(), "; #it{p} (GeV/#it{c}); TOF Signal", kTH2F, {{100, 0, 10}, {1000, 0, 100e3}});
  }
}

template <typename T>
auto FemtoDreamTrackSelection::getNsigmaTPC(T const& track, o2::track::PID pid)
{
  switch (pid) {
    case o2::track::PID::Electron:
      return track.tpcNSigmaEl();
      break;
    case o2::track::PID::Muon:
      return track.tpcNSigmaMu();
      break;
    case o2::track::PID::Pion:
      return track.tpcNSigmaPi();
      break;
    case o2::track::PID::Kaon:
      return track.tpcNSigmaKa();
      break;
    case o2::track::PID::Proton:
      return track.tpcNSigmaPr();
      break;
    case o2::track::PID::Deuteron:
      return track.tpcNSigmaDe();
      break;
    default:
      return 999.f;
      break;
  }
}

template <typename T>
auto FemtoDreamTrackSelection::getNsigmaTOF(T const& track, o2::track::PID pid)
{
  /// skip tracks without TOF signal
  /// \todo not sure what the error flags mean...
  if (track.tofSignal() <= 0.f || std::abs(track.tofSignal() - 99998) < 0.01 || std::abs(track.tofSignal() - 99999) < 0.01) {
    return 999.f;
  }

  switch (pid) {
    case o2::track::PID::Electron:
      return track.tofNSigmaEl();
      break;
    case o2::track::PID::Muon:
      return track.tofNSigmaMu();
      break;
    case o2::track::PID::Pion:
      return track.tofNSigmaPi();
      break;
    case o2::track::PID::Kaon:
      return track.tofNSigmaKa();
      break;
    case o2::track::PID::Proton:
      return track.tofNSigmaPr();
      break;
    case o2::track::PID::Deuteron:
      return track.tofNSigmaDe();
      break;
    default:
      return 999.f;
      break;
  }
}

template <typename T>
bool FemtoDreamTrackSelection::isSelectedMinimal(T const& track)
{
  const auto pT = track.pt();
  const auto eta = track.eta();
  const auto tpcNClsF = track.tpcNClsFound();
  const auto tpcRClsC = track.tpcCrossedRowsOverFindableCls();
  const auto tpcNClsC = track.tpcNClsCrossedRows();
  const auto tpcNClsS = track.tpcNClsShared();
  const auto dcaXY = track.dcaXY();
  const auto dcaZ = track.dcaZ();
  const auto dca = std::sqrt(pow(dcaXY, 2.) + pow(dcaZ, 2.));
  std::vector<float> pidTPC, pidTOF;
  for (auto it : mPIDspecies) {
    pidTPC.push_back(getNsigmaTPC(track, it));
    pidTOF.push_back(getNsigmaTOF(track, it));
  }

  /// check whether the most open cuts are fulfilled
  const static int nPtMinSel = getNSelections(femtoDreamTrackSelection::kpTMin);
  const static int nPtMaxSel = getNSelections(femtoDreamTrackSelection::kpTMax);
  const static int nEtaSel = getNSelections(femtoDreamTrackSelection::kEtaMax);
  const static int nTPCnMinSel = getNSelections(femtoDreamTrackSelection::kTPCnClsMin);
  const static int nTPCfMinSel = getNSelections(femtoDreamTrackSelection::kTPCfClsMin);
  const static int nTPCcMinSel = getNSelections(femtoDreamTrackSelection::kTPCcRowsMin);
  const static int nTPCsMaxSel = getNSelections(femtoDreamTrackSelection::kTPCsClsMax);
  const static int nDCAxyMaxSel = getNSelections(femtoDreamTrackSelection::kDCAxyMax);
  const static int nDCAzMaxSel = getNSelections(femtoDreamTrackSelection::kDCAzMax);
  const static int nDCAMinSel = getNSelections(femtoDreamTrackSelection::kDCAMin);
  const static int nPIDnSigmaSel = getNSelections(femtoDreamTrackSelection::kPIDnSigmaMax);

  const static float pTMin = getMinimalSelection(femtoDreamTrackSelection::kpTMin, femtoDreamSelection::kLowerLimit);
  const static float pTMax = getMinimalSelection(femtoDreamTrackSelection::kpTMax, femtoDreamSelection::kUpperLimit);
  const static float etaMax = getMinimalSelection(femtoDreamTrackSelection::kEtaMax, femtoDreamSelection::kAbsUpperLimit);
  const static float nClsMin = getMinimalSelection(femtoDreamTrackSelection::kTPCnClsMin, femtoDreamSelection::kLowerLimit);
  const static float fClsMin = getMinimalSelection(femtoDreamTrackSelection::kTPCfClsMin, femtoDreamSelection::kLowerLimit);
  const static float cTPCMin = getMinimalSelection(femtoDreamTrackSelection::kTPCcRowsMin, femtoDreamSelection::kLowerLimit);
  const static float sTPCMax = getMinimalSelection(femtoDreamTrackSelection::kTPCsClsMax, femtoDreamSelection::kUpperLimit);
  const static float dcaXYMax = getMinimalSelection(femtoDreamTrackSelection::kDCAxyMax, femtoDreamSelection::kAbsUpperLimit);
  const static float dcaZMax = getMinimalSelection(femtoDreamTrackSelection::kDCAzMax, femtoDreamSelection::kAbsUpperLimit);
  const static float dcaMin = getMinimalSelection(femtoDreamTrackSelection::kDCAMin, femtoDreamSelection::kAbsLowerLimit);
  const static float nSigmaPIDMax = getMinimalSelection(femtoDreamTrackSelection::kPIDnSigmaMax, femtoDreamSelection::kAbsUpperLimit);

  if (nPtMinSel > 0 && pT < pTMin) {
    return false;
  }
  if (nPtMaxSel > 0 && pT > pTMax) {
    return false;
  }
  if (nEtaSel > 0 && std::abs(eta) > etaMax) {
    return false;
  }
  if (nTPCnMinSel > 0 && tpcNClsF < nClsMin) {
    return false;
  }
  if (nTPCfMinSel > 0 && tpcRClsC < fClsMin) {
    return false;
  }
  if (nTPCcMinSel > 0 && tpcNClsC < cTPCMin) {
    return false;
  }
  if (nTPCsMaxSel > 0 && tpcNClsS > sTPCMax) {
    return false;
  }
  if (nDCAxyMaxSel > 0 && std::abs(dcaXY) > dcaXYMax) {
    return false;
  }
  if (nDCAzMaxSel > 0 && std::abs(dcaZ) > dcaZMax) {
    return false;
  }
  if (nDCAMinSel > 0 && std::abs(dca) < dcaMin) {
    return false;
  }
  if (nPIDnSigmaSel > 0) {
    bool isFulfilled = false;
    for (size_t i = 0; i < pidTPC.size(); ++i) {
      auto pidTPCVal = pidTPC.at(i);
      auto pidTOFVal = pidTOF.at(i);
      auto pidComb = std::sqrt(pidTPCVal * pidTPCVal + pidTOFVal * pidTOFVal);
      if (std::abs(pidTPCVal) < nSigmaPIDMax || pidComb < nSigmaPIDMax) {
        isFulfilled = true;
      }
    }
    if (!isFulfilled) {
      return isFulfilled;
    }
  }
  return true;
}

template <typename cutContainerType, typename T>
std::array<cutContainerType, 2> FemtoDreamTrackSelection::getCutContainer(T const& track)
{
  cutContainerType output = 0;
  size_t counter = 0;
  cutContainerType outputPID = 0;
  size_t counterPID = 0;
  const auto sign = track.sign();
  const auto pT = track.pt();
  const auto eta = track.eta();
  const auto tpcNClsF = track.tpcNClsFound();
  const auto tpcRClsC = track.tpcCrossedRowsOverFindableCls();
  const auto tpcNClsC = track.tpcNClsCrossedRows();
  const auto tpcNClsS = track.tpcNClsShared();
  const auto dcaXY = track.dcaXY();
  const auto dcaZ = track.dcaZ();
  const auto dca = std::sqrt(pow(dcaXY, 2.) + pow(dcaZ, 2.));

  std::vector<float> pidTPC, pidTOF;
  for (auto it : mPIDspecies) {
    pidTPC.push_back(getNsigmaTPC(track, it));
    pidTOF.push_back(getNsigmaTOF(track, it));
  }

  float observable;
  for (auto& sel : mSelections) {
    const auto selVariable = sel.getSelectionVariable();
    if (selVariable == femtoDreamTrackSelection::kPIDnSigmaMax) {
      /// PID needs to be handled a bit differently since we may need more than one species
      for (size_t i = 0; i < pidTPC.size(); ++i) {
        auto pidTPCVal = pidTPC.at(i);
        auto pidTOFVal = pidTOF.at(i);
        sel.checkSelectionSetBit(pidTPCVal, outputPID, counterPID);
        auto pidComb = std::sqrt(pidTPCVal * pidTPCVal + pidTOFVal * pidTOFVal);
        sel.checkSelectionSetBit(pidComb, outputPID, counterPID);
      }

    } else {
      /// for the rest it's all the same
      switch (selVariable) {
        case (femtoDreamTrackSelection::kSign):
          observable = sign;
          break;
        case (femtoDreamTrackSelection::kpTMin):
        case (femtoDreamTrackSelection::kpTMax):
          observable = pT;
          break;
        case (femtoDreamTrackSelection::kEtaMax):
          observable = eta;
          break;
        case (femtoDreamTrackSelection::kTPCnClsMin):
          observable = tpcNClsF;
          break;
        case (femtoDreamTrackSelection::kTPCfClsMin):
          observable = tpcRClsC;
          break;
        case (femtoDreamTrackSelection::kTPCcRowsMin):
          observable = tpcNClsC;
          break;
        case (femtoDreamTrackSelection::kTPCsClsMax):
          observable = tpcNClsS;
          break;
        case (femtoDreamTrackSelection::kDCAxyMax):
          observable = dcaXY;
          break;
        case (femtoDreamTrackSelection::kDCAzMax):
          observable = dcaZ;
          break;
        case (femtoDreamTrackSelection::kDCAMin):
          observable = dca;
          break;
        case (femtoDreamTrackSelection::kPIDnSigmaMax):
          break;
      }
      sel.checkSelectionSetBit(observable, output, counter);
    }
  }
  return {output, outputPID};
}

template <o2::aod::femtodreamparticle::ParticleType part, typename T>
void FemtoDreamTrackSelection::fillQA(T const& track)
{
  if (mHistogramRegistry) {
    mHistogramRegistry->fill(HIST(o2::aod::femtodreamparticle::ParticleTypeName[part]) + HIST("/pThist"), track.pt());
    mHistogramRegistry->fill(HIST(o2::aod::femtodreamparticle::ParticleTypeName[part]) + HIST("/etahist"), track.eta());
    mHistogramRegistry->fill(HIST(o2::aod::femtodreamparticle::ParticleTypeName[part]) + HIST("/phihist"), track.phi());
    mHistogramRegistry->fill(HIST(o2::aod::femtodreamparticle::ParticleTypeName[part]) + HIST("/tpcnclshist"), track.tpcNClsFound());
    mHistogramRegistry->fill(HIST(o2::aod::femtodreamparticle::ParticleTypeName[part]) + HIST("/tpcfclshist"), track.tpcCrossedRowsOverFindableCls());
    mHistogramRegistry->fill(HIST(o2::aod::femtodreamparticle::ParticleTypeName[part]) + HIST("/tpcnrowshist"), track.tpcNClsCrossedRows());
    mHistogramRegistry->fill(HIST(o2::aod::femtodreamparticle::ParticleTypeName[part]) + HIST("/tpcnsharedhist"), track.tpcNClsShared());
    mHistogramRegistry->fill(HIST(o2::aod::femtodreamparticle::ParticleTypeName[part]) + HIST("/dcaXYhist"), track.pt(), track.dcaXY());
    mHistogramRegistry->fill(HIST(o2::aod::femtodreamparticle::ParticleTypeName[part]) + HIST("/dcaZhist"), track.pt(), track.dcaZ());
    mHistogramRegistry->fill(HIST(o2::aod::femtodreamparticle::ParticleTypeName[part]) + HIST("/dcahist"), std::sqrt(pow(track.dcaXY(), 2.) + pow(track.dcaZ(), 2.)));
    mHistogramRegistry->fill(HIST(o2::aod::femtodreamparticle::ParticleTypeName[part]) + HIST("/tpcdEdx"), track.tpcInnerParam(), track.tpcSignal());
    mHistogramRegistry->fill(HIST(o2::aod::femtodreamparticle::ParticleTypeName[part]) + HIST("/tofSignal"), track.p(), track.tofSignal());
  }
}

} // namespace o2::analysis::femtoDream

#endif /* ANALYSIS_TASKS_PWGCF_FEMTODREAM_FEMTODREAMTRACKSELECTION_H_ */
