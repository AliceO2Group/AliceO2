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

/// \file ToyCluster.C
/// \brief This macro implements a simple Toy-MC for creating clusters for ideal toy-tracks. The output trees containing the cluster properties can be used to create the track topology correction.
///
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Feb 17, 2022

/*
  Overview:
    This macro can be used to create simple toy tracks, perform the digitisation for these tracks and to perform the clustering.
    The clusters can be used to study the dependencies of qMax and qTot on varoius parameters like the track angles theta and phi, or the the drift lenth.
    The output tree can be used for performing ML parametrizations of the dependencies and used as an input for the track topology correction of the dE/dx.
    Steps:
    1. create Toy tracks: straight tracks for given dE/dx. One track per event is simulated to keep track of the track properties
       call simulateTracks()
    2. perform the digitization of the Toy tracks
       call createDigitsFromSim()
    3. create the final output tree containing the cluster properties (zero supression can be added)
       call createCluster()
*/

#if !defined(__CLING__) || defined(__ROOTCLING__)
#define FMT_HEADER_ONLY // to avoid 'You are probably missing the definition of...'
#include <fmt/format.h>
#include <vector>

// root includes
#include "TTree.h"
#include "TFile.h"
#include "TF1.h"
#include "TRandom.h"

// O2 includes
#include "TPCBase/Mapper.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCSimulation/ElectronTransport.h"
#include "TPCSimulation/SAMPAProcessing.h"
#include "TPCSimulation/Point.h"
#include "DataFormatsTPC/Defs.h"
#include "TPCSimulation/DigitContainer.h"
#include "TPCSimulation/CommonMode.h"
#include <SimulationDataFormat/IOMCTruthContainerView.h>
#include "CommonUtils/TreeStreamRedirector.h"
#include "CommonUtils/ConfigurableParam.h"
#include "TPCBase/ParameterGas.h"
#include "TPCReconstruction/HwClusterer.h"
#include "TPCSimulation/GEMAmplification.h"
#endif

using namespace o2::tpc;
void fillTPCHits(const float theta, const float phi, const float dedx, std::vector<HitGroup>& hitGroupSector, std::pair<GlobalPosition3D, GlobalPosition3D>& trackInfo);
GlobalPosition3D getPointFromPhi(float phi, float theta, GlobalPosition3D globalReferencePoint);
GlobalPosition3D getPosBTrack(const GlobalPosition3D& posA, const GlobalPosition3D& posB);
GlobalPosition3D getGlobalPositionTrk(float lambda, const GlobalPosition3D& refA, const GlobalPosition3D& refB);

const int mSector = 4;       ///< consider only this mSector
const float mMaxDrift = 270; ///< maximum drift length in cm

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// Creating the Toy-MC tracks /////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// create Toy-MC tracks (o2sim.root with custom TPCHits)
/// \param maxEvents number of events (tracks) to simualate (one track per event)
/// \param outputFolder output folder where the hits will be stored
/// \param dedxMin mininmum dE/dx of the tracks (primary ionization: number of electrons per cm)
/// \param dedxMax maximum dE/dx of the tracks (primary ionization: number of electrons per cm)
/// \param maxSinPhi maximum sin(phi) of the tracks
/// \param maxTanTheta maximum tan(theta) of the tracks
void simulateTracks(const int maxEvents = 1, const char* outputFolder = "./", const int dedxMin = 14, const int dedxMax = 50, const float maxSinPhi = 1.f, const float maxTanTheta = 1.6f)
{
  // output file / tree
  TFile fOut(fmt::format("{}/o2sim_HitsTPC.root", outputFolder).data(), "RECREATE");
  TTree treeO2Sim("o2sim", "o2sim");
  std::vector<HitGroup> hitGroupSector = {};
  treeO2Sim.Branch(fmt::format("TPCHitsShiftedSector{}", mSector).data(), &hitGroupSector);

  std::pair<GlobalPosition3D, GlobalPosition3D> trackInfo = {};
  float phi = 0;   // phi of the track
  float theta = 0; // theta of the track
  float dedx = 0;  // dE/dx (primary ionization of the track)
  treeO2Sim.Branch("trackInfo", &trackInfo);
  treeO2Sim.Branch("phi", &phi);
  treeO2Sim.Branch("theta", &theta);
  treeO2Sim.Branch("dedx", &dedx);

  // set random seed
  gRandom->SetSeed(0);

  // bias dedx to lower values
  TF1 fdEdx("fdEdx", "-log(x) + 10", 1, dedxMax);

  // loop over events (tracks)
  for (int iEvent = 0; iEvent < maxEvents; ++iEvent) {
    hitGroupSector.clear();

    // draw random track parameters
    phi = std::asin(gRandom->Uniform(0, maxSinPhi));
    theta = std::atan(gRandom->Uniform(0, maxTanTheta));
    dedx = (dedxMin == dedxMax) ? dedxMin : fdEdx.GetRandom();

    // fill the tree (simulate the primary electrons along the track)
    fillTPCHits(theta, phi, dedx, hitGroupSector, trackInfo);

    // save the data on new Event
    treeO2Sim.Fill();
  }
  fOut.cd();
  fOut.Write();
}

/// fill the hitGroupSector object which will be written to the tree (create the track)
/// \param theta theta of the track
/// \param phi phi of the track
/// \parma dedx dedx of the track
/// \param hitGroupSector output vector containing the hits of the track
void fillTPCHits(const float theta, const float phi, const float dedx, std::vector<HitGroup>& hitGroupSector, std::pair<GlobalPosition3D, GlobalPosition3D>& trackInfo)
{
  const static Mapper& mapper = Mapper::instance();
  const int chargeHit = 1;
  const float xPos = gRandom->Uniform(-2, 2);          // x starting position of track
  const int ireg = gRandom->Integer(Mapper::NREGIONS); // region where the track starts
  static auto& detParam = ParameterDetector::Instance();
  const float minZ = detParam.TPClength - mMaxDrift;
  static TF1 fz("fz", "tanh(x/60 - 3.5) * 0.6  + 1.6", minZ, detParam.TPClength);
  const float zCoordinate = fz.GetRandom(); // draw z position

  const float flightTime = 0;                                                  // flight time of the particle. This value is not needed and set therefor to 0.
  const float radiusStart = mapper.getPadRegionInfo(ireg).getRadiusFirstRow(); // region start
  const auto padLength = mapper.getPadRegionInfo(ireg).getPadHeight();
  const float radius = radiusStart + Mapper::ROWSPERREGION[ireg] / 2 * padLength; // start in the center of the region
  HitGroup hitGroup{};                                                            // create HitGroup and push_back to TPCHitsShiftedSector
  const float radiusEnd = radiusStart + Mapper::ROWSPERREGION[ireg] * padLength;

  // first point of the track
  const GlobalPosition3D posATrack(xPos, radius, zCoordinate);

  // with the angle phi and theta we can a the second point of our track
  GlobalPosition3D globalPosSecondPoint = getPointFromPhi(phi, theta, posATrack);

  // storing the track parameters in global variable for easy access
  const auto posBTrack = getPosBTrack(posATrack, globalPosSecondPoint);

  // loop over the track (Ensure to draw the track along the whole region)
  // calculate stepsize! the stepsize has to be chosen such that the points are distributed equally along the track (for ALL tracks)
  // electrons are distributed just equally along the track
  float nRef = 1 / dedx;
  for (float trkPos = -200; trkPos < 400; trkPos += nRef) {
    GlobalPosition3D posTrk = getGlobalPositionTrk(trkPos, posATrack, posBTrack);
    const float xTmp = posTrk.X();
    const float yTmp = posTrk.Y();
    const float zTmp = posTrk.Z();
    const float radiusCurr = std::sqrt(xTmp * xTmp + yTmp * yTmp);

    // check if the track is in the TPC
    const float rMinTPC = radiusStart;
    const float rMaxTPC = radiusEnd;

    if (std::abs(zTmp) > detParam.TPClength || radiusCurr > rMaxTPC || yTmp < rMinTPC || zTmp < minZ || mapper.isOutOfSector(posTrk, Sector(mSector)) || radiusCurr < rMinTPC) {
      continue;
    }

    // add hit
    hitGroup.addHit(xTmp, yTmp, zTmp, flightTime, chargeHit);
  }

  // store values
  hitGroupSector.emplace_back(hitGroup);
  trackInfo = std::pair<GlobalPosition3D, GlobalPosition3D>(posATrack, posBTrack);
}

/// \return returns point along straight track from given input point and given track parameters
/// \param phi phi of the track
/// \param theta theta of the track
/// \param globalReferencePoint point on the track (starting point)
GlobalPosition3D getPointFromPhi(float phi, float theta, GlobalPosition3D globalReferencePoint)
{
  const float yVal = (phi > M_PI_2) ? std::sin(M_PI - phi) : std::sin(phi);
  const float xVal = (phi > M_PI_2) ? -std::sqrt(1 - yVal * yVal) : std::sqrt(1 - yVal * yVal);
  const float zVal = (theta > M_PI_2) ? std::tan(M_PI - theta) : std::tan(theta);
  GlobalPosition3D pos(yVal + globalReferencePoint.X(), xVal + globalReferencePoint.Y(), zVal + globalReferencePoint.Z());
  return pos;
}

/// get second point along the track
GlobalPosition3D getPosBTrack(const GlobalPosition3D& posA, const GlobalPosition3D& posB)
{
  const float xAB = posB.X() - posA.X();
  const float yAB = posB.Y() - posA.Y();
  const float zAB = posB.Z() - posA.Z();

  // normalize vector to length "1"
  const float lengthVec = std::sqrt(xAB * xAB + yAB * yAB + zAB * zAB);
  GlobalPosition3D refB(xAB / lengthVec, yAB / lengthVec, zAB / lengthVec);
  return refB;
}

/// \return returns global position of the track
/// \param lambda distance to starting point of the track
GlobalPosition3D getGlobalPositionTrk(float lambda, const GlobalPosition3D& refA, const GlobalPosition3D& refB)
{
  const float trkX = refA.X() + lambda * refB.X();
  const float trkY = refA.Y() + lambda * refB.Y();
  const float trkZ = refA.Z() + lambda * refB.Z();
  GlobalPosition3D trackPoint(trkX, trkY, trkZ);
  return trackPoint;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// Creating the digits ////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// dump digits to tree
void dumpDigits(DigitContainer& digitContainer, const float eventtime, std::vector<o2::tpc::Digit>& digits, o2::dataformats::IOMCTruthContainerView& labels, std::vector<o2::tpc::CommonMode>& commonMode, TBranch* brdigits, TBranch* brlabel, TBranch* brmCommon);

/// Secondary ionization generation with parton distribution function (PDF) ∼ 1/E 2.2 (as in GEANT3 gas implementation).
/// return number of secondary electrons for given primary electron
/// see: https://github.com/alisw/AliRoot/blob/master/TPC/fastSimul/AliTPCclusterFast.cxx: Double_t AliTPCclusterFast::GetNsec()
int getNsec()
{
  // Generate number of secondary electrons
  // copy of procedure implemented in geant
  static const auto& parGas = ParameterGas::Instance();
  const double fpot = parGas.Ipot;
  const double eend = parGas.Eend;
  const double eexpo = parGas.Exp;
  const double xexpo = -eexpo + 1;
  const double yexpo = 1 / xexpo;
  const double w = parGas.Wion;
  const double ran = gRandom->Rndm();
  return TMath::Nint(TMath::Power((TMath::Power(fpot, xexpo) * (1 - ran) + TMath::Power(eend, xexpo) * ran), yexpo) / w);
}

/// \param eleAttachmentFac electron attachement is scaled by this factor: AttCoeff = AttCoeff_Nominal * eleAttachmentFac
float setElectronAttachement(const float eleAttachmentFac)
{
  static auto& gasParam = ParameterGas::Instance();
  float attachement = gasParam.AttCoeff * eleAttachmentFac;
  o2::conf::ConfigurableParam::updateFromString(fmt::format("TPCGasParam.AttCoeff={}", attachement).data());
  LOG(info) << "electron attachement is: " << gasParam.AttCoeff;
  return attachement;
}

void setZBinWidth(const int facZWidth = 1)
{
  auto& eleParam = ParameterElectronics::Instance();
  o2::conf::ConfigurableParam::setValue<float>("TPCEleParam", "ZbinWidth", eleParam.ZbinWidth * facZWidth);
  o2::conf::ConfigurableParam::setValue<int>("TPCEleParam", "NShapedPoints", eleParam.NShapedPoints * facZWidth);
  const float zbinWidth = eleParam.ZbinWidth;
  LOG(info) << "zbinWidth: " << zbinWidth;
}

GlobalPosition3D getElectronDrift(const GlobalPosition3D& posEle)
{
  static o2::math_utils::RandomRing<> randomGaus;

  const auto& detParam = ParameterDetector::Instance();
  const auto& gasParam = ParameterGas::Instance();
  float driftl = detParam.TPClength - posEle.Z();
  if (driftl < 0.01) {
    driftl = 0.01;
  }
  driftl = std::sqrt(driftl);
  const float sigT = driftl * gasParam.DiffT;
  const float sigL = driftl * gasParam.DiffL;

  /// The position is smeared by a Gaussian with mean around the actual position and a width according to the diffusion
  /// coefficient times sqrt(drift length)
  GlobalPosition3D posEleDiffusion((randomGaus.getNextValue() * sigT) + posEle.X(), (randomGaus.getNextValue() * sigT) + posEle.Y(), (randomGaus.getNextValue() * sigL) + posEle.Z());
  return posEleDiffusion;
}

/// creating the digits from the simulated hits (similar to the digitizer in O2)
/// \param inpFileSim input sim file
/// \param outName output path
/// \param eleAttachmentFac electron attachement is scaled by this factor: AttCoeff = AttCoeff_Nominal * eleAttachmentFac
/// \param facZWidth scale factor for the the sampling time
/// \param maxSecondaries restrict maximum number of secondaries (-1 for no restriction)
/// \param nEleGEM number of electrons after GEM amplification (if nEleGEM=-1 use realistic GEM amplification)
/// \param disableNoise do not simulate any noise, commonMode and pedestal
void createDigitsFromSim(const char* inpFileSim = "o2sim_HitsTPC.root", const std::string outName = "digits.root", const float eleAttachmentFac = 1, const int facZWidth = 1, const int maxSecondaries = 3, const int nEleGEM = -1, const bool disableNoise = false)
{
  // set random seed
  gRandom->SetSeed(0);

  auto& cdb = CDBInterface::instance();
  cdb.setUseDefaults();

  // disable noise
  if (disableNoise) {
    const int mode = static_cast<int>(o2::tpc::DigitzationMode::PropagateADC);
    o2::conf::ConfigurableParam::updateFromString(Form("TPCEleParam.DigiMode=%i", mode));
  }

  // set electron attachement
  float attachement = setElectronAttachement(eleAttachmentFac);

  // set z bin width
  setZBinWidth(facZWidth);

  const Mapper& mapper = Mapper::instance();
  auto& detParam = ParameterDetector::Instance();
  auto& eleParam = ParameterElectronics::Instance();
  const float zbinWidth = eleParam.ZbinWidth;
  static GEMAmplification& gemAmplification = GEMAmplification::instance();
  const auto& gasParam = ParameterGas::Instance();

  ElectronTransport& electronTransport = ElectronTransport::instance();
  electronTransport.updateParameters();

  SAMPAProcessing& sampaProcessing = SAMPAProcessing::instance();
  sampaProcessing.updateParameters();

  // memory for signal from SAMPA
  std::vector<float> signalArray;
  const int nShapedPoints = eleParam.NShapedPoints;
  signalArray.resize(nShapedPoints);

  // open input file and tree
  std::unique_ptr<TFile> hitFile = std::unique_ptr<TFile>(TFile::Open(inpFileSim));
  std::unique_ptr<TTree> hitTree = std::unique_ptr<TTree>((TTree*)hitFile->Get("o2sim"));

  // vector of HitGroups per mSector
  std::vector<::HitGroup>* arrSectors = nullptr;
  std::stringstream sectornamestr;
  sectornamestr << "TPCHitsShiftedSector" << mSector;
  hitTree->SetBranchAddress(sectornamestr.str().c_str(), &arrSectors);

  // set up output file and tree
  TFile fOut(outName.data(), "RECREATE");
  TTree tree("o2sim", "o2sim");

  std::vector<o2::tpc::Digit> digits;
  o2::dataformats::IOMCTruthContainerView labels;
  std::vector<o2::tpc::CommonMode> commonMode;
  TBranch* brdigits = tree.Branch(fmt::format("TPCDigit_{}", mSector).data(), &digits);
  TBranch* brlabel = tree.Branch(fmt::format("TPCDigitMCTruth_{}", mSector).data(), &labels);
  TBranch* brmCommon = tree.Branch(fmt::format("TPCCommonMode_{}", mSector).data(), &commonMode);

  // loop over hits
  const int nEvents = hitTree->GetEntries(); // number of simulated events per hit file
  for (int iev = 0; iev < nEvents; ++iev) {
    const double eventTime = 0;
    if (iev % 50 == false) {
      LOG(info) << "event: " << iev + 1 << " from " << nEvents << " events";
    }
    hitTree->GetEntry(iev);

    DigitContainer digitContainer;
    const auto firstTimeBinForCurrentEvent = sampaProcessing.getTimeBinFromTime(eventTime);
    digitContainer.setStartTime(firstTimeBinForCurrentEvent);
    digitContainer.reserve(firstTimeBinForCurrentEvent);

    const float maxEleTime = (int(digitContainer.size()) - nShapedPoints) * zbinWidth;
    auto vecTracks = arrSectors;
    for (auto& hitGroup : *vecTracks) {
      const int MCTrackID = hitGroup.GetTrackID();
      for (size_t ihit = 0; ihit < hitGroup.getSize(); ++ihit) {
        const auto& eh = hitGroup.getHit(ihit);
        GlobalPosition3D posEle(eh.GetX(), eh.GetY(), eh.GetZ());

        const int nPrimaryElectrons = static_cast<int>(eh.GetEnergyLoss());
        const float hitTime = eh.GetTime() * 0.001f;
        if (nPrimaryElectrons <= 0) {
          continue;
        }

        for (int iele = 0; iele < nPrimaryElectrons; iele++) {
          const GlobalPosition3D posEleDiff = getElectronDrift(posEle);

          // add secondaries
          int nSecondaries = (maxSecondaries == 0) ? 0 : getNsec();

          // restrict secondaries to avoid clusters with high charge (tail in the qMax qTot distributions)
          if (nSecondaries > maxSecondaries) {
            nSecondaries = maxSecondaries;
          }

          // storage for all electrons (primary + secondaries)
          std::vector<GlobalPosition3D> posTotElectrons;
          posTotElectrons.reserve(nSecondaries + 1);
          posTotElectrons.emplace_back(posEleDiff);

          // apply some diffusion to secondaries
          for (int isec = 0; isec < nSecondaries; ++isec) {
            // electrons are created randomly
            const double x = gRandom->Gaus(0, std::abs(posEle.X() - posEleDiff.X())) + posEle.X();
            const double y = gRandom->Gaus(0, std::abs(posEle.Y() - posEleDiff.Y())) + posEle.Y();
            const double z = gRandom->Gaus(0, std::abs(posEle.Z() - posEleDiff.Z())) + posEle.Z();
            posTotElectrons.emplace_back(GlobalPosition3D(x, y, z));
          }

          // loop over all electrons
          for (unsigned int j = 0; j < posTotElectrons.size(); ++j) {
            auto posEleTmp = posTotElectrons[j];
            const float driftTime = (detParam.TPClength - posEleTmp.Z()) / gasParam.DriftV;

            const float eleTime = driftTime + hitTime; /// in us
            if (eleTime > maxEleTime) {
              LOG(warning) << "Skipping electron with driftTime " << driftTime << " from hit at time " << hitTime;
              continue;
            }

            // Attachment
            if (electronTransport.isElectronAttachment(driftTime)) {
              continue;
            }

            // Remove electrons that end up outside the active volume
            if (std::abs(posEleTmp.Z()) > detParam.TPClength) {
              continue;
            }

            // When the electron is not in the mSector we're processing, abandon
            // create dummy pos at A-Side
            auto posEleTmpTmp = posEleTmp;
            posEleTmpTmp.SetZ(1);
            if (mapper.isOutOfSector(posEleTmpTmp, mSector)) {
              continue;
            }

            const DigitPos digiPadPos = mapper.findDigitPosFromGlobalPosition(posEleTmpTmp, mSector);
            if (!digiPadPos.isValid()) {
              continue;
            }

            const CRU cru = digiPadPos.getCRU();
            int sectorTmp = cru.sector();
            if (sectorTmp != mSector) {
              continue;
            }

            // fixed GEM amplification for gaussian response (could be changed)
            const float nElectronsGEM = (nEleGEM == -1) ? static_cast<int>(gemAmplification.getEffectiveStackAmplification()) : nEleGEM;

            // convert electrons to ADC signal
            const GlobalPadNumber globalPad = mapper.globalPadNumber(digiPadPos.getGlobalPadPos());
            const float adcsignal = sampaProcessing.getADCvalue(static_cast<float>(nElectronsGEM));
            sampaProcessing.getShapedSignal(adcsignal, driftTime, signalArray);

            // set up MC label
            const int eventID = iev;
            const int sourceID = 0; // TPC
            const o2::MCCompLabel label(MCTrackID, eventID, sourceID, false);

            for (int i = 0; i < nShapedPoints; ++i) {
              const float timebin = driftTime / eleParam.ZbinWidth + i;
              digitContainer.addDigit(label, cru, timebin, globalPad, signalArray[i]);
            }
          }
        } // electron loop
      }   // hit loop
    }     // track loop

    // dump digits for each event to a file
    dumpDigits(digitContainer, eventTime, digits, labels, commonMode, brdigits, brlabel, brmCommon);
  } // event loop

  fOut.cd();
  tree.SetEntries(nEvents);
  fOut.WriteObject(&tree, "o2sim");

  std::vector<float> attachementTmp{attachement};
  fOut.WriteObject(&attachementTmp, "EleAttFac");
  fOut.Close();

  delete arrSectors;
}

/// dump digits to file
void dumpDigits(DigitContainer& digitContainer, const float eventtime, std::vector<o2::tpc::Digit>& digits, o2::dataformats::IOMCTruthContainerView& labels, std::vector<o2::tpc::CommonMode>& commonMode, TBranch* brdigits, TBranch* brlabel, TBranch* brmCommon)
{
  static SAMPAProcessing& sampaProcessing = SAMPAProcessing::instance();
  sampaProcessing.updateParameters();

  Sector sec(mSector);
  const bool isContinuous = false;
  const bool finalFlush = false;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> labelsTmp;
  digitContainer.fillOutputContainer(digits, labelsTmp, commonMode, sec, sampaProcessing.getTimeBinFromTime(eventtime), isContinuous, finalFlush);

  // flatten labels
  std::vector<char> buffer;
  labelsTmp.flatten_to(buffer);
  labelsTmp.clear_andfreememory();
  o2::dataformats::IOMCTruthContainerView view(buffer);
  labels = view;

  // fill output branches
  brdigits->Fill();
  brlabel->Fill();
  brmCommon->Fill();
  digits.clear();
  commonMode.clear();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// Creating the clusters //////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// \param inpHits input file containing the hits which was created using simulateTracks
/// \param outFile output file
/// \param digitFile file from the digitisation which was created using createDigitsFromSim
/// \param zeroSuppressionMin minimum zero supression threshold
/// \param zeroSuppressionMax maximum zero supression threshold
/// \param rejectSinglePadTimeCl reject single time and pad clusters during cluster finder
void createCluster(const char* inpHits = "o2sim_HitsTPC.root", const char* outFile = "clusters.root", const char* digitFile = "digits.root", const float zeroSuppressionMin = 0, const float zeroSuppressionMax = 1, const bool rejectSinglePadTimeCl = true)
{
  gRandom->SetSeed(0);

  auto& eleParam = ParameterElectronics::Instance();
  auto& gasParam = ParameterGas::Instance();
  auto& cdb = CDBInterface::instance();
  cdb.setUseDefaults();

  const static Mapper& mapper = Mapper::instance();
  SAMPAProcessing& sampaProcessing = SAMPAProcessing::instance();

  // load the theta, phi and dE/dx angles from the simulation
  TFile fTrackInf(inpHits, "READ");
  TTree* tTrkInf = (TTree*)fTrackInf.Get("o2sim");
  float phi = 0;
  float theta = 0;
  float dedx = 0;
  std::pair<GlobalPosition3D, GlobalPosition3D>* trackInfo = new std::pair<GlobalPosition3D, GlobalPosition3D>;
  tTrkInf->SetBranchAddress("phi", &phi);
  tTrkInf->SetBranchAddress("theta", &theta);
  tTrkInf->SetBranchAddress("dedx", &dedx);
  tTrkInf->SetBranchAddress("trackInfo", &trackInfo);
  const int nEventsTrk = tTrkInf->GetEntries();

  // setup clusterer
  std::vector<ClusterHardwareContainer8kb> clusterOutput;
  HwClusterer clusterer(&clusterOutput, mSector);
  clusterer.setContinuousReadout(false);
  clusterer.setRejectSinglePadClusters(rejectSinglePadTimeCl);
  clusterer.setRejectSingleTimeClusters(rejectSinglePadTimeCl);

  // setup digits tree
  TFile fDigits(digitFile, "READ");
  TTree* tDigi = (TTree*)fDigits.Get("o2sim");
  float eleAttFac = 0;
  std::vector<float>* eleAtt = nullptr;
  fDigits.GetObject("EleAttFac", eleAtt);
  eleAttFac = eleAtt != nullptr ? eleAtt->front() : 1;
  if (eleAtt) {
    delete eleAtt;
  }
  std::vector<o2::tpc::Digit>* digits = new std::vector<o2::tpc::Digit>;
  tDigi->SetBranchAddress(fmt::format("TPCDigit_{}", mSector).data(), &digits);

  std::vector<float> vdedx;
  std::vector<float> vrelTime;
  std::vector<float> vpad;
  std::vector<float> vrelPad;
  std::vector<float> vsigmaTime;
  std::vector<float> vsigmaPad;
  std::vector<int> vpadrow;
  std::vector<int> vregion;
  std::vector<int> vqMax;
  std::vector<int> vqTot;
  std::vector<float> vphi;
  std::vector<float> vtheta;
  std::vector<float> vzPos;
  std::vector<bool> visEdge;
  std::vector<bool> vsinglePadOrTime;
  std::vector<bool> vSigmaTimeCut;
  std::vector<bool> vLargestqTotinRow;
  std::vector<int> vLocalRow;
  std::vector<float> zeroSuppOut;
  std::vector<o2::tpc::Digit> digitsZS;
  std::vector<float> vdistToTrack;
  std::vector<float> vdistToTrackXY;
  std::vector<float> vdistToTrackZ;

  // output file streamer
  o2::utils::TreeStreamRedirector pcstream(outFile, "RECREATE");
  for (int ev = 0; ev < nEventsTrk; ++ev) {
    tTrkInf->GetEntry(ev);
    tDigi->GetEntry(ev);

    vdistToTrack.clear();
    vdistToTrackXY.clear();
    vdistToTrackZ.clear();
    vdedx.clear();
    vrelTime.clear();
    vpad.clear();
    vrelPad.clear();
    vsigmaTime.clear();
    vsigmaPad.clear();
    vpadrow.clear();
    vregion.clear();
    vqMax.clear();
    vqTot.clear();
    vphi.clear();
    vtheta.clear();
    vzPos.clear();
    visEdge.clear();
    vsinglePadOrTime.clear();
    vSigmaTimeCut.clear();
    vLargestqTotinRow.clear();
    vLocalRow.clear();
    zeroSuppOut.clear();

    // perform zero supression
    const float zeroSuppression = gRandom->Uniform(zeroSuppressionMin, zeroSuppressionMax);
    const float ndigits = digits->size();
    digitsZS.clear();
    digitsZS.reserve(ndigits);
    for (int j = 0; j < ndigits; ++j) {
      const auto digiVal = (*digits)[j];
      if (digiVal.getChargeFloat() > zeroSuppression) {
        digitsZS.emplace_back(digiVal);
      }
    }

    // perform clustering for event
    const o2::dataformats::ConstMCLabelContainerView emptyLabels;
    clusterer.process(digitsZS, emptyLabels);

    // loop over clusters
    for (auto cont : clusterOutput) {
      auto container = cont.getContainer();
      const CRU cru(container->CRU);
      const int rowOffset = mapper.getPadRegionInfo(cru.region()).getGlobalRowOffset();

      for (int clusterCount = 0; clusterCount < container->numberOfClusters; ++clusterCount) {
        const auto timeBinOffset = container->timeBinOffset;
        auto& cluster = container->clusters[clusterCount];
        const int qTot = cluster.getQTot();
        const int qMax = cluster.getQMax();
        const float pad = cluster.getPad() + 0.5f;
        const float time = cluster.getTimeLocal() + timeBinOffset;
        const int padrow = rowOffset + cluster.getRow();
        const int region = cru.region();
        const float sigmaPad = std::sqrt(cluster.getSigmaPad2());
        const float sigmaTime = std::sqrt(cluster.getSigmaTime2());
        const float zPos = sampaProcessing.getZfromTimeBin(time, Side::A) + eleParam.ZbinWidth * gasParam.DriftV;
        const float relPad = cluster.getPad() - static_cast<int>(pad);
        const float relTime = time - static_cast<int>(time + 0.5f);

        // check for mSector edge pad
        const int off = 2;
        const int offPad = 2;
        const int localPadRow = Mapper::getLocalRowFromGlobalRow(padrow);
        bool isEdge = false;
        if (pad < offPad || pad >= (Mapper::PADSPERROW[region][localPadRow] - offPad - 1) || localPadRow < off || localPadRow >= Mapper::ROWSPERREGION[region] - off) {
          isEdge = true;
        }

        bool singlePadOrTime = false;
        if (sigmaPad == 0 || sigmaTime == 0) {
          singlePadOrTime = true;
        }

        // perform check on clusters in same row
        bool largestqTot = true;
        std::vector<int>::iterator iter = vpadrow.begin();
        while ((iter = std::find(iter, vpadrow.end(), padrow)) != vpadrow.end()) {
          const int index = std::distance(vpadrow.begin(), iter);
          if (qTot < vqTot[index]) {
            largestqTot = false;
            break;
          } else {
            vLargestqTotinRow[index] = false;
          }
          iter++;
        }

        const int glPadNumber = Mapper::getGlobalPadNumber(localPadRow, pad, region);
        const auto& padPosLocal = mapper.padPos(glPadNumber);

        PadSecPos pos(Sector(mSector), padPosLocal);
        GlobalPosition2D globalPosA = mapper.getPadCentre(pos);

        const auto posB = getGlobalPositionTrk(1, trackInfo->first, trackInfo->second);
        const float xTrkF = trackInfo->first.X() - posB.X();
        const float yTrkF = trackInfo->first.Y() - posB.Y();
        const float zTrkF = trackInfo->first.Z() - posB.Z();

        const float globX = globalPosA.X() - posB.X();
        const float globY = globalPosA.Y() - posB.Y();
        const float globZ = zPos - posB.Z();

        const float dist = (globX * xTrkF + globY * yTrkF + globZ * zTrkF) / (xTrkF * xTrkF + yTrkF * yTrkF + zTrkF * zTrkF);
        const float distanceToTrack = std::sqrt(std::pow(globX - dist * xTrkF, 2) + std::pow(globY - dist * yTrkF, 2) + std::pow(globZ - dist * zTrkF, 2));

        const float distXT = (globX * xTrkF + globY * yTrkF) / (xTrkF * xTrkF + yTrkF * yTrkF);
        const float distanceToTrackXY = std::sqrt(std::pow(globX - distXT * xTrkF, 2) + std::pow(globY - distXT * yTrkF, 2));

        const float distanceToTrackZ = globZ - dist * zTrkF;

        vLargestqTotinRow.emplace_back(largestqTot);
        vdedx.emplace_back(dedx);
        vdistToTrack.emplace_back(distanceToTrack);
        vdistToTrackXY.emplace_back(distanceToTrackXY);
        vdistToTrackZ.emplace_back(distanceToTrackZ);
        vrelTime.emplace_back(relTime);
        vpad.emplace_back(pad);
        vrelPad.emplace_back(relPad);
        vsigmaTime.emplace_back(sigmaTime);
        vsigmaPad.emplace_back(sigmaPad);
        vpadrow.emplace_back(padrow);
        vregion.emplace_back(region);
        vqMax.emplace_back(qMax);
        vqTot.emplace_back(qTot);
        vphi.emplace_back(phi);
        vtheta.emplace_back(theta);
        vzPos.emplace_back(zPos);
        visEdge.emplace_back(isEdge);
        vsinglePadOrTime.emplace_back(singlePadOrTime);
        vLocalRow.emplace_back(localPadRow);
        zeroSuppOut.emplace_back(zeroSuppression);
      }
    }

    // writer
    pcstream.GetFile()->cd();
    pcstream << "cl"
             << "dedx=" << vdedx                       // dE/dx (primary ionization) of the track
             << "pad=" << vpad                         // pad position of the cluster
             << "relPad=" << vrelPad                   // relative pad position of the cluster
             << "relTime=" << vrelTime                 // relative time position of the cluster
             << "sigmaTime=" << vsigmaTime             // sigma time of the cluster
             << "sigmaPad=" << vsigmaPad               // sigma pad of the cluster
             << "padrow=" << vpadrow                   // pad row of the cluster
             << "lpadrow=" << vLocalRow                // local pad row in the region of the cluster
             << "region=" << vregion                   // region of the cluster
             << "qMax=" << vqMax                       // qMax of the cluster
             << "qTot=" << vqTot                       // qTot of the cluster
             << "phi=" << vphi                         // phi of the track
             << "theta=" << vtheta                     // theta of the track
             << "z=" << vzPos                          // z of the cluster
             << "isEdge=" << visEdge                   // true if the cluster is edge pad
             << "singlePadOrTime=" << vsinglePadOrTime // true if the cluster is single pad or single time cluster
             << "eleAttFac=" << eleAttFac              // electron attachement factor
             << "zeroSupp=" << zeroSuppOut             // absolute zero supression value in ADC counts
             << "isLargestqTot=" << vLargestqTotinRow  // is true cluster has the highest qTot from all clusters in the same pad row (cluster with lower charge are noise)
             << "distToTrack=" << vdistToTrack         // distance of cluster to track
             << "distToTrackXY=" << vdistToTrackXY     // distance of cluster to track in xy direction
             << "distToTrackZ=" << vdistToTrackZ       // distance of cluster to track in z direction
             << "\n";
  }

  TTree* tree = (TTree*)(pcstream.GetFile()->Get("cl"));
  tree->SetAlias("cut", "isLargestqTot==1 && singlePadOrTime==0 && isEdge==0"); // cut to filter clusters
  pcstream.Close();
  fTrackInf.Close();
}
