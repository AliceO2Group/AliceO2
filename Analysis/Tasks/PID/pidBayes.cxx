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

///
/// \file   pidBayes.cxx
/// \author Nicolo' Jacazio
/// \brief  Task to produce PID tables for Bayes PID.
///         Only the tables for the mass hypotheses requested are filled, the others are sent empty.
///

// O2 includes
#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "Framework/RunningWorkflowInfo.h"
#include "Framework/Array2D.h"
#include <CCDB/BasicCCDBManager.h>
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/PID/PIDTOF.h"
#include "AnalysisDataModel/PID/PIDTPC.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::pid;
using namespace o2::framework::expressions;
using namespace o2::track;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{{"add-qa", VariantType::Int, 0, {"Produce Bayes PID QA histograms"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

struct bayesPid {
  using Trks = soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov>;
  using Coll = aod::Collisions;

  // Tables to produce
  Produces<o2::aod::pidBayesEl> tablePIDEl; /// Table for the Electron
  Produces<o2::aod::pidBayesMu> tablePIDMu; /// Table for the Muon
  Produces<o2::aod::pidBayesPi> tablePIDPi; /// Table for the Pion
  Produces<o2::aod::pidBayesKa> tablePIDKa; /// Table for the Kaon
  Produces<o2::aod::pidBayesPr> tablePIDPr; /// Table for the Proton
  Produces<o2::aod::pidBayesDe> tablePIDDe; /// Table for the Deuteron
  Produces<o2::aod::pidBayesTr> tablePIDTr; /// Table for the Triton
  Produces<o2::aod::pidBayesHe> tablePIDHe; /// Table for the Helium3
  Produces<o2::aod::pidBayesAl> tablePIDAl; /// Table for the Alpha
  Produces<o2::aod::pidBayes> tableBayes;   /// Table of the most probable particle type

  /// Types of probabilities that can be computed N.B. the order is important, detectors first!
  enum ProbType : int { kTOF = 0,  /// Probabilities with the TOF detector
                        kTPC,      /// Probabilities with the TPC detector
                        kNDet,     /// Number of detectors
                        kPrior,    /// Prior probabilities
                        kMerged,   /// Merged probabilities without the prior probability applied (not normalized)
                        kBayesian, /// Bayesian probability with the prior probability applied (normalized)
                        kNProb };

  // Detector response and input parameters
  std::array<DetectorResponse, kNDet> Response;
  static constexpr const char* detectorName[kNDet] = {"TOF", "TPC"};
  Service<o2::ccdb::BasicCCDBManager> ccdb;
  Configurable<std::string> paramfile{"param-file", "", "Path to the parametrization object, if emtpy the parametrization is not taken from file"};
  Configurable<std::string> TOFsigmaname{"param-tof-sigma", "TOFReso", "Name of the parametrization for the expected sigma, used in both file and CCDB mode"};
  Configurable<std::string> TPCsignalname{"param-tpc-signal", "BetheBloch", "Name of the parametrization for the expected signal, used in both file and CCDB mode"};
  Configurable<std::string> TPCsigmaname{"param-tpc-sigma", "TPCReso", "Name of the parametrization for the expected sigma, used in both file and CCDB mode"};
  Configurable<bool> enableTOF{"enableTOF", false, "Enabling TOF"};
  Configurable<bool> enableTPC{"enableTPC", false, "Enabling TPC"};
  /// Ordering has to respect the one in ProbType

  std::array<bool, kNDet> enabledDet{false};

  Configurable<std::string> url{"ccdb-url", "http://ccdb-test.cern.ch:8080", "url of the ccdb repository"};
  Configurable<std::string> ccdbPathTOF{"ccdbPathTOF", "Analysis/PID/TOF", "Path of the TOF parametrization on the CCDB"};
  Configurable<std::string> ccdbPathTPC{"ccdbPathTPC", "Analysis/PID/TPC", "Path of the TPC parametrization on the CCDB"};
  Configurable<long> timestamp{"ccdb-timestamp", -1, "timestamp of the object"};
  // Configuration flags to include and exclude particle hypotheses
  // Configurable<LabeledArray<int>> pid{"pid",
  //                                     {{-1, -1, -1, -1, -1, -1, -1, -1, -1}, 9, {"el", "mu", "pi", "ka", "pr", "de", "tr", "he", "al"}},
  //                                     "Produce PID information for the various mass hypotheses, overrides the automatic setup: the corresponding table can be set off (0) or on (1)"};
  Configurable<int> pidEl{"pid-el", -1, {"Produce PID information for the Electron mass hypothesis, overrides the automatic setup: the corresponding table can be set off (0) or on (1)"}};
  Configurable<int> pidMu{"pid-mu", -1, {"Produce PID information for the Muon mass hypothesis, overrides the automatic setup: the corresponding table can be set off (0) or on (1)"}};
  Configurable<int> pidPi{"pid-pi", -1, {"Produce PID information for the Pion mass hypothesis, overrides the automatic setup: the corresponding table can be set off (0) or on (1)"}};
  Configurable<int> pidKa{"pid-ka", -1, {"Produce PID information for the Kaon mass hypothesis, overrides the automatic setup: the corresponding table can be set off (0) or on (1)"}};
  Configurable<int> pidPr{"pid-pr", -1, {"Produce PID information for the Proton mass hypothesis, overrides the automatic setup: the corresponding table can be set off (0) or on (1)"}};
  Configurable<int> pidDe{"pid-de", -1, {"Produce PID information for the Deuterons mass hypothesis, overrides the automatic setup: the corresponding table can be set off (0) or on (1)"}};
  Configurable<int> pidTr{"pid-tr", -1, {"Produce PID information for the Triton mass hypothesis, overrides the automatic setup: the corresponding table can be set off (0) or on (1)"}};
  Configurable<int> pidHe{"pid-he", -1, {"Produce PID information for the Helium3 mass hypothesis, overrides the automatic setup: the corresponding table can be set off (0) or on (1)"}};
  Configurable<int> pidAl{"pid-al", -1, {"Produce PID information for the Alpha mass hypothesis, overrides the automatic setup: the corresponding table can be set off (0) or on (1)"}};

  std::array<std::array<float, PID::NIDs>, kNProb> Probability; /// Probabilities for all the cases defined in ProbType
  std::vector<PID::ID> enabledSpecies;                          /// Enabled species

  /// Checker of the species that are enabled and initializer of the probabilities
  template <ProbType detIndex, o2::track::PID::ID pid>
  bool checkEnabled()
  {
    static_assert(detIndex < kNDet && detIndex >= 0);
    if (!enabledDet[detIndex]) {
      LOG(DEBUG) << "Detector " << detectorName[detIndex] << " disabled";
      return false; // Setting the probability to 1 if the detector is disabled
    }
    LOG(DEBUG) << "Detector " << detectorName[detIndex] << " enabled with " << enabledSpecies.size() << " species";

    for (const auto enabledPid : enabledSpecies) { // Checking that the species is enabled
      LOG(DEBUG) << "Testing " << PID::getName(enabledPid) << " " << (int)enabledPid << " vs " << (int)pid;
      if (enabledPid == pid) {
        LOG(DEBUG) << "Particle " << PID::getName(enabledPid) << " enabled";
        Probability[detIndex][pid] = 1.f / enabledSpecies.size(); // set flat distribution (no decision yet)
        return true;
      }
    }
    return false;
  }

  float fRange = 5.f;

  void init(o2::framework::InitContext& initContext)
  {
    for (int i = 0; i < kNProb; i++) { // Setting all probabilities to the default value
      for (int j = 0; j < PID::NIDs; j++) {
        Probability[i][j] = 1.f;
      }
    }
    // Enabling detectors
    enabledDet[kTOF] = enableTOF;
    enabledDet[kTPC] = enableTPC;
    // Checking that at least one detectors is enabled
    bool atLeastOne = false;
    for (int i = 0; i < kNDet; i++) {
      if (enabledDet[i]) {
        atLeastOne = (atLeastOne || enabledDet[i]);
      }
    }
    if (!atLeastOne) {
      LOG(FATAL) << "All detectors are disabled";
    }

    // Checking the tables are requested in the workflow and enabling them
    auto& workflows = initContext.services().get<RunningWorkflowInfo const>();
    for (DeviceSpec device : workflows.devices) {
      for (auto input : device.inputs) {
        auto enableFlag = [&input, this](const PID::ID& id,
                                         Configurable<int>& flag) {
          const std::string particles[PID::NIDs] = {"El", "Mu", "Pi", "Ka", "Pr", "De", "Tr", "He", "Al"};
          const std::string particle = particles[id];
          const std::string table = "pidBayes" + particle;
          if (input.matcher.binding == table) {
            if (flag < 0) {
              flag.value = 1;
              LOG(INFO) << "Auto-enabling table: " + table;
            } else if (flag > 0) {
              flag.value = 1;
              LOG(INFO) << "Table enabled: " + table;
            } else {
              LOG(INFO) << "Table disabled: " + table;
            }
          }
        };
        enableFlag(PID::Electron, pidEl);
        enableFlag(PID::Muon, pidMu);
        enableFlag(PID::Pion, pidPi);
        enableFlag(PID::Kaon, pidKa);
        enableFlag(PID::Proton, pidPr);
        enableFlag(PID::Deuteron, pidDe);
        enableFlag(PID::Triton, pidTr);
        enableFlag(PID::Helium3, pidHe);
        enableFlag(PID::Alpha, pidAl);
      }
    }

    // Enabling species (only once) that are requested
    enabledSpecies.reserve(PID::NIDs); // Reserving a space for every particle species
    auto enableParticle = [this](const PID::ID& id,
                                 const Configurable<int>& flag) {
      if (flag.value == 1) {
        enabledSpecies.push_back(id);
        LOG(INFO) << "Enabling particle: " << (int)id << " " << PID::getName(enabledSpecies.back());
      }
    };

    enableParticle(PID::Electron, pidEl);
    enableParticle(PID::Muon, pidMu);
    enableParticle(PID::Pion, pidPi);
    enableParticle(PID::Kaon, pidKa);
    enableParticle(PID::Proton, pidPr);
    enableParticle(PID::Deuteron, pidDe);
    enableParticle(PID::Triton, pidTr);
    enableParticle(PID::Helium3, pidHe);
    enableParticle(PID::Alpha, pidAl);

    enabledSpecies.shrink_to_fit();
    std::sort(enabledSpecies.begin(), enabledSpecies.end());
    if (enabledSpecies.size() == 0) { // No enabled species
      LOG(FATAL) << "No species are enabled";
    } else if (enabledSpecies.size() > PID::NIDs) { // Too many enabled species
      LOG(FATAL) << "Too many enabled species! " << enabledSpecies.size() << " vs " << static_cast<int>(PID::NIDs) << " available";
    } else if (std::unique(enabledSpecies.begin(), enabledSpecies.end()) != enabledSpecies.end()) { // Duplicate enabled species
      LOG(FATAL) << "Duplicated enabled species!";
    } else { // All ok
      LOG(INFO) << enabledSpecies.size() << " species enabled for the Bayesian PID computation";
    }
    // Getting the parametrization parameters
    ccdb->setURL(url.value);
    ccdb->setTimestamp(timestamp.value);
    ccdb->setCaching(true);
    ccdb->setLocalObjectValidityChecking();
    // Not later than now objects
    ccdb->setCreatedNotAfter(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    //
    const std::vector<float> p = {0.008, 0.008, 0.002, 40.0};
    Response[kTOF].SetParameters(DetectorResponse::kSigma, p);
    const std::string fname = paramfile.value;
    if (!fname.empty()) { // Loading the parametrization from file
      LOG(INFO) << "Loading exp. sigma parametrization from file" << fname << ", using param: " << TOFsigmaname.value;
      Response[kTOF].LoadParamFromFile(fname.data(), TOFsigmaname.value, DetectorResponse::kSigma);

      LOG(INFO) << "Loading exp. signal parametrization from file" << fname << ", using param: " << TPCsignalname.value;
      Response[kTPC].LoadParamFromFile(fname.data(), TPCsignalname.value, DetectorResponse::kSignal);

      LOG(INFO) << "Loading exp. sigma parametrization from file" << fname << ", using param: " << TPCsigmaname.value;
      Response[kTPC].LoadParamFromFile(fname.data(), TPCsigmaname.value, DetectorResponse::kSigma);

    } else { // Loading it from CCDB
      std::string path = ccdbPathTOF.value + "/" + TOFsigmaname.value;
      LOG(INFO) << "Loading exp. sigma parametrization from CCDB, using path: " << path << " for timestamp " << timestamp.value;
      Response[kTOF].LoadParam(DetectorResponse::kSigma, ccdb->getForTimeStamp<Parametrization>(path, timestamp.value));

      path = ccdbPathTPC.value + "/" + TPCsignalname.value;
      LOG(INFO) << "Loading exp. signal parametrization from CCDB, using path: " << path << " for timestamp " << timestamp.value;
      Response[kTPC].LoadParam(DetectorResponse::kSignal, ccdb->getForTimeStamp<Parametrization>(path, timestamp.value));

      path = ccdbPathTPC.value + "/" + TPCsigmaname.value;
      LOG(INFO) << "Loading exp. sigma parametrization from CCDB, using path: " << path << " for timestamp " << timestamp.value;
      Response[kTPC].LoadParam(DetectorResponse::kSigma, ccdb->getForTimeStamp<Parametrization>(path, timestamp.value));
    }
  }

  template <o2::track::PID::ID pid>
  using respTPC = tpc::ELoss<Coll::iterator, Trks::iterator, pid>;

  /// Computes PID probabilities for the TPC
  template <o2::track::PID::ID pid>
  void ComputeTPCProbability(const Trks::iterator& track)
  {

    if (!checkEnabled<kTPC, pid>()) {
      return;
    }

    constexpr respTPC<pid> responseTPCPID;

    const float dedx = track.tpcSignal();
    bool mismatch = true;

    // if (fTuneMConData && ((fTuneMConDataMask & kDetTPC) == kDetTPC)){
    //   dedx = GetTPCsignalTunedOnData(track);
    // }

    const float bethe = responseTPCPID.GetExpectedSignal(Response[kTPC], track.collision(), track);
    const float sigma = responseTPCPID.GetExpectedSigma(Response[kTPC], track.collision(), track);
    LOG(INFO) << "For " << pid_constants::sNames[pid] << " computing bethe " << bethe << " and sigma " << sigma;
    // bethe = fTPCResponse.GetExpectedSignal(track, type, AliTPCPIDResponse::kdEdxDefault, fUseTPCEtaCorrection, fUseTPCMultiplicityCorrection, fUseTPCPileupCorrection);
    // sigma = fTPCResponse.GetExpectedSigma(track, type, AliTPCPIDResponse::kdEdxDefault, fUseTPCEtaCorrection, fUseTPCMultiplicityCorrection, fUseTPCPileupCorrection);

    if (abs(dedx - bethe) > fRange * sigma) {
      // Probability[kTPC][pid] = exp(-0.5 * fRange * fRange) / sigma; // BUG fix
      Probability[kTPC][pid] = exp(-0.5 * fRange * fRange);
    } else {
      // Probability[kTPC][pid] = exp(-0.5 * (dedx - bethe) * (dedx - bethe) / (sigma * sigma)) / sigma; //BUG fix
      Probability[kTPC][pid] = exp(-0.5 * (dedx - bethe) * (dedx - bethe) / (sigma * sigma));
      mismatch = false;
    }
    if (Probability[kTPC][pid] <= 0.f) {
      Probability[kTPC][pid] = 0.f;
    }
    if (mismatch) {
      Probability[kTPC][pid] = 1.f / Probability[kTPC].size();
    }
  }

  float fgTOFmismatchProb = 0.f;
  // float fgTOFmismatchProb = 1E-8;

  bool fNoTOFmism = true;
  float fTOFtail = 0.9;

  /// Response of the TOF detector
  template <o2::track::PID::ID pid>
  using respTOF = tof::ExpTimes<Coll::iterator, Trks::iterator, pid>;

  /// Compute PID probabilities for TOF
  template <o2::track::PID::ID pid>
  void ComputeTOFProbability(const Trks::iterator& track)
  {

    if (!checkEnabled<kTOF, pid>()) {
      return;
    }

    if (!track.hasTOF()) {
      return;
    }
    constexpr respTOF<pid> responseTOFPID;

    const float pt = track.pt();
    float mismPropagationFactor[10] = {1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
    // In the O2 this cannot be done because the cluster information is missing in the AOD
    // if (!fNoTOFmism) {                                                                  // this flag allows to disable mismatch for iterative procedure to get prior probabilities
    //   mismPropagationFactor[3] = 1 + exp(1 - 1.12 * pt);                                // it has to be alligned with the one in AliPIDCombined
    //   mismPropagationFactor[4] = 1 + 1. / (4.71114 - 5.72372 * pt + 2.94715 * pt * pt); // it has to be alligned with the one in AliPIDCombined

    //   int nTOFcluster = 0;
    //   if (track->GetTOFHeader() && track->GetTOFHeader()->GetTriggerMask() && track->GetTOFHeader()->GetNumberOfTOFclusters() > -1) { // N TOF clusters available
    //     nTOFcluster = track->GetTOFHeader()->GetNumberOfTOFclusters();
    //     if (fIsMC)
    //       nTOFcluster = int(nTOFcluster * 1.5); // +50% in MC
    //   } else {
    //     switch (fBeamTypeNum) {
    //       case kPP: // pp
    //         nTOFcluster = 80;
    //         break;
    //       case kPPB: // pPb 5.05 ATeV
    //         nTOFcluster = int(308 - 2.12 * fCurrCentrality + exp(4.917 - 0.1604 * fCurrCentrality));
    //         break;
    //       case kPBPB: // PbPb 2.76 ATeV
    //         nTOFcluster = int(exp(9.4 - 0.022 * fCurrCentrality));
    //         break;
    //     }
    //   }

    //   switch (fBeamTypeNum) { // matching window factors for 3 cm and 10 cm (about (10/3)^2)
    //     case kPP:             // pp 7 TeV
    //       nTOFcluster *= 10;
    //       break;
    //     case kPPB: // pPb 5.05 ATeV
    //       nTOFcluster *= 10;
    //       break;
    //     case kPBPB: // pPb 5.05 ATeV
    //       //       nTOFcluster *= 1;
    //       break;
    //   }

    //   if (nTOFcluster < 0)
    //     nTOFcluster = 10;

    //   fgTOFmismatchProb = fTOFResponse.GetMismatchProbability(track->GetTOFsignal(), track->Eta()) * nTOFcluster * 6E-6 * (1 + 2.90505e-01 / pt / pt); // mism weight * tof occupancy (including matching window factor) * pt dependence
    // }

    const float meanCorrFactor = 0.07 / fTOFtail; // Correction factor on the mean because of the tail (should be ~ 0.1 with tail = 1.1)

    const float nsigmas = responseTOFPID.GetSeparation(Response[kTOF], track.collision(), track) + meanCorrFactor;

    const float expTime = responseTOFPID.GetExpectedSignal(track.collision(), track);
    const float sig = responseTOFPID.GetExpectedSigma(Response[kTOF], track.collision(), track);

    if (nsigmas < fTOFtail) {
      Probability[kTOF][pid] = exp(-0.5 * nsigmas * nsigmas) / sig;
    } else {
      Probability[kTOF][pid] = exp(-(nsigmas - fTOFtail * 0.5) * fTOFtail) / sig;
    }

    Probability[kTOF][pid] += fgTOFmismatchProb * mismPropagationFactor[pid];
    LOG(DEBUG) << "For " << pid_constants::sNames[pid] << " with signal " << track.tofSignal() << " computing exp time " << expTime << " and sigma " << sig << " and nsigma " << nsigmas << " probability " << Probability[kTOF][pid];
  }

  /// Calculate probabilities from all enabled detectors and species
  void MergeProbabilities()
  {
    LOG(DEBUG) << "Merging probabilities";

    float probSum[kNDet + 1] = {0.f}; /// Summed probabilities for all detectors + 1 for the sum of all detectors

    for (const auto enabledPid : enabledSpecies) {
      Probability[kMerged][enabledPid] = 1.f;
      for (int det = 0; det < kNDet; det++) {
        probSum[det] += Probability[det][enabledPid];
        Probability[kMerged][enabledPid] *= Probability[det][enabledPid];
      }
      probSum[kNDet] += Probability[kMerged][enabledPid];
      LOG(DEBUG) << "For " << PID::getName(enabledPid);
      for (int det = 0; det < kNDet; det++) {
        if (enabledDet[det]) {
          LOG(DEBUG) << "\t" << detectorName[det] << " Probability " << Probability[det][enabledPid];
        }
      }
      LOG(DEBUG) << "\tCombined: " << Probability[kMerged][enabledPid];
    }
    for (int det = 0; det < kNDet; det++) {
      if (enabledDet[det]) {
        LOG(DEBUG) << "\tSum " << detectorName[det] << " Probability " << probSum[det];
      }
    }
    LOG(DEBUG) << "\tSum combined: " << probSum[kNDet];
  }

  /// Calculate Bayesian probabilities
  void ComputeBayesProbabilities()
  {
    LOG(DEBUG) << "Computing bayes probabilities";

    float sum = 0.;
    for (const auto enabledPid : enabledSpecies) {
      LOG(DEBUG) << "Adding " << Probability[kMerged][enabledPid] << " with prior " << Probability[kPrior][enabledPid];
      sum += Probability[kMerged][enabledPid] * Probability[kPrior][enabledPid];
    }
    if (sum <= 0) {
      LOG(WARNING) << "Invalid probability densities or prior probabilities";
      for (int i = 0; i < Probability[kBayesian].size(); i++) {
        Probability[kBayesian][i] = 1.f / Probability[kBayesian].size();
      }
      return;
    }
    for (const auto enabledPid : enabledSpecies) {
      Probability[kBayesian][enabledPid] = Probability[kMerged][enabledPid] * Probability[kPrior][enabledPid] / sum;
      // if (probDensityMism) {
      //   probDensityMism[enabledPid] *= Probability[kPrior][enabledPid] / sum;
      // }
      LOG(DEBUG) << "For " << PID::getName(enabledPid) << " merged prob. " << Probability[kMerged][enabledPid] << " prior " << Probability[kPrior][enabledPid] << " sum " << sum << " bayesian Probability: " << Probability[kBayesian][enabledPid];
    }
  }

  void process(Coll const& collisions, Trks const& tracks)
  {

    // Check and fill enabled tables
    auto makeTable = [&tracks](const Configurable<int>& flag, auto& table) {
      if (flag.value == 1) {
        // Prepare memory for enabled tables
        table.reserve(tracks.size());
      }
    };

    tableBayes.reserve(tracks.size());
    makeTable(pidEl, tablePIDEl);
    makeTable(pidMu, tablePIDMu);
    makeTable(pidPi, tablePIDPi);
    makeTable(pidKa, tablePIDKa);
    makeTable(pidPr, tablePIDPr);
    makeTable(pidDe, tablePIDDe);
    makeTable(pidTr, tablePIDTr);
    makeTable(pidHe, tablePIDHe);
    makeTable(pidAl, tablePIDAl);

    for (auto const& trk : tracks) { // Loop on Tracks

      ComputeTPCProbability<PID::Electron>(trk);
      ComputeTPCProbability<PID::Muon>(trk);
      ComputeTPCProbability<PID::Pion>(trk);
      ComputeTPCProbability<PID::Kaon>(trk);
      ComputeTPCProbability<PID::Proton>(trk);
      ComputeTPCProbability<PID::Deuteron>(trk);
      ComputeTPCProbability<PID::Triton>(trk);
      ComputeTPCProbability<PID::Helium3>(trk);
      ComputeTPCProbability<PID::Alpha>(trk);

      ComputeTOFProbability<PID::Electron>(trk);
      ComputeTOFProbability<PID::Muon>(trk);
      ComputeTOFProbability<PID::Pion>(trk);
      ComputeTOFProbability<PID::Kaon>(trk);
      ComputeTOFProbability<PID::Proton>(trk);
      ComputeTOFProbability<PID::Deuteron>(trk);
      ComputeTOFProbability<PID::Triton>(trk);
      ComputeTOFProbability<PID::Helium3>(trk);
      ComputeTOFProbability<PID::Alpha>(trk);

      MergeProbabilities();

      ComputeBayesProbabilities();

      if (pidEl == 1) {
        tablePIDEl(Probability[kBayesian][PID::Electron] * 100.f);
      }
      if (pidMu == 1) {
        tablePIDMu(Probability[kBayesian][PID::Muon] * 100.f);
      }
      if (pidPi == 1) {
        tablePIDPi(Probability[kBayesian][PID::Pion] * 100.f);
      }
      if (pidKa == 1) {
        tablePIDKa(Probability[kBayesian][PID::Kaon] * 100.f);
      }
      if (pidPr == 1) {
        tablePIDPr(Probability[kBayesian][PID::Proton] * 100.f);
      }
      if (pidDe == 1) {
        tablePIDDe(Probability[kBayesian][PID::Deuteron] * 100.f);
      }
      if (pidTr == 1) {
        tablePIDTr(Probability[kBayesian][PID::Triton] * 100.f);
      }
      if (pidHe == 1) {
        tablePIDHe(Probability[kBayesian][PID::Helium3] * 100.f);
      }
      if (pidAl == 1) {
        tablePIDAl(Probability[kBayesian][PID::Alpha] * 100.f);
      }
      const auto mostProbable = std::max_element(Probability[kBayesian].begin(), Probability[kBayesian].end());
      tableBayes((*mostProbable) * 100.f, std::distance(Probability[kBayesian].begin(), mostProbable));
    }
  }
};

struct bayesPidQa {

  static constexpr int Np = 9;
  static constexpr const char* pT[Np] = {"e", "#mu", "#pi", "K", "p", "d", "t", "^{3}He", "#alpha"};
  static constexpr std::string_view hprob[Np] = {"probability/El", "probability/Mu", "probability/Pi",
                                                 "probability/Ka", "probability/Pr", "probability/De",
                                                 "probability/Tr", "probability/He", "probability/Al"};
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::QAObject};

  Configurable<int> logAxis{"logAxis", 1, "Flag to use a log momentum axis"};
  Configurable<int> nBinsP{"nBinsP", 400, "Number of bins for the momentum"};
  Configurable<float> MinP{"MinP", 0.1f, "Minimum momentum in range"};
  Configurable<float> MaxP{"MaxP", 5.f, "Maximum momentum in range"};
  Configurable<int> nBinsProb{"nBinsProb", 100, "Number of bins for the Probability"};
  Configurable<float> MinProb{"MinProb", 0.f, "Minimum Probability in range"};
  Configurable<float> MaxProb{"MaxProb", 100.f, "Maximum Probability in range"};

  template <typename T>
  void makelogaxis(T h)
  {
    if (logAxis == 0) {
      return;
    }
    const int nbins = h->GetNbinsX();
    double binp[nbins + 1];
    double max = h->GetXaxis()->GetBinUpEdge(nbins);
    double min = h->GetXaxis()->GetBinLowEdge(1);
    if (min <= 0) {
      min = 0.00001;
    }
    double lmin = TMath::Log10(min);
    double ldelta = (TMath::Log10(max) - lmin) / ((double)nbins);
    for (int i = 0; i < nbins; i++) {
      binp[i] = exp(TMath::Log(10) * (lmin + i * ldelta));
    }
    binp[nbins] = max + 1;
    h->GetXaxis()->Set(nbins, binp);
  }

  template <uint8_t i>
  void addParticleHistos()
  {
    // Probability
    histos.add(hprob[i].data(), Form(";#it{p} (GeV/#it{c});N_{#sigma}^{TOF}(%s)", pT[i]), HistType::kTH2F, {{nBinsP, MinP, MaxP}, {nBinsProb, MinProb, MaxProb}});
    makelogaxis(histos.get<TH2>(HIST(hprob[i])));
  }

  void init(o2::framework::InitContext&)
  {
    // Event properties
    histos.add("event/vertexz", ";Vtx_{z} (cm);Entries", HistType::kTH1F, {{100, -20, 20}});
    histos.add("event/colltime", ";Collision time (ps);Entries", HistType::kTH1F, {{100, -2000, 2000}});
    histos.add("event/tofsignal", ";#it{p} (GeV/#it{c});TOF Signal", HistType::kTH2F, {{nBinsP, MinP, MaxP}, {10000, 0, 2e6}});
    makelogaxis(histos.get<TH2>(HIST("event/tofsignal")));
    histos.add("event/eta", ";#it{#eta};Entries", HistType::kTH1F, {{100, -2, 2}});
    histos.add("event/length", ";Track length (cm);Entries", HistType::kTH1F, {{100, 0, 500}});
    histos.add("event/pt", ";#it{p}_{T} (GeV/#it{c});Entries", HistType::kTH1F, {{nBinsP, MinP, MaxP}});
    histos.add("event/p", ";#it{p} (GeV/#it{c});Entries", HistType::kTH1F, {{nBinsP, MinP, MaxP}});
    histos.add("event/ptreso", ";#it{p} (GeV/#it{c});Entries", HistType::kTH2F, {{nBinsP, MinP, MaxP}, {100, 0, 0.1}});
    histos.add("mostProbable", ";#it{p} (GeV/#it{c});Entries", HistType::kTH2F, {{nBinsP, MinP, MaxP}, {nBinsProb, MinProb, MaxProb}});

    addParticleHistos<0>();
    addParticleHistos<1>();
    addParticleHistos<2>();
    addParticleHistos<3>();
    addParticleHistos<4>();
    addParticleHistos<5>();
    addParticleHistos<6>();
    addParticleHistos<7>();
    addParticleHistos<8>();
  }

  template <uint8_t i, typename T>
  void fillParticleHistos(const T& t, const float prob)
  {
    histos.fill(HIST(hprob[i]), t.p(), prob);
  }

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov,
                                                          aod::pidBayesEl, aod::pidBayesMu, aod::pidBayesPi,
                                                          aod::pidBayesKa, aod::pidBayesPr, aod::pidBayesDe,
                                                          aod::pidBayesTr, aod::pidBayesHe, aod::pidBayesAl,
                                                          aod::pidBayes, aod::TrackSelection> const& tracks)
  {
    const float collisionTime_ps = collision.collisionTime() * 1000.f;
    histos.fill(HIST("event/vertexz"), collision.posZ());
    histos.fill(HIST("event/colltime"), collisionTime_ps);

    for (auto t : tracks) {
      //
      if (!t.hasTOF()) { // Skipping tracks without TOF
        continue;
      }
      if (!t.isGlobalTrack()) {
        continue;
      }

      const float tof = t.tofSignal() - collisionTime_ps;

      //
      histos.fill(HIST("event/tofsignal"), t.p(), t.tofSignal());
      histos.fill(HIST("event/eta"), t.eta());
      histos.fill(HIST("event/length"), t.length());
      histos.fill(HIST("event/pt"), t.pt());
      histos.fill(HIST("event/ptreso"), t.p(), t.sigma1Pt() * t.pt() * t.pt());
      histos.fill(HIST("mostProbable"), t.p(), t.bayesProb());
      //
      fillParticleHistos<0>(t, t.bayesEl());
      fillParticleHistos<1>(t, t.bayesMu());
      fillParticleHistos<2>(t, t.bayesPi());
      fillParticleHistos<3>(t, t.bayesKa());
      fillParticleHistos<4>(t, t.bayesPr());
      fillParticleHistos<5>(t, t.bayesDe());
      fillParticleHistos<6>(t, t.bayesTr());
      fillParticleHistos<7>(t, t.bayesHe());
      fillParticleHistos<8>(t, t.bayesAl());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto workflow = WorkflowSpec{adaptAnalysisTask<bayesPid>(cfgc)};
  if (cfgc.options().get<int>("add-qa")) {
    workflow.push_back(adaptAnalysisTask<bayesPidQa>(cfgc));
  }
  return workflow;
}
