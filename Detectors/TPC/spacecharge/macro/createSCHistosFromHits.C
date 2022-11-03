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

/// \file createSCHistosFromHits
/// \brief This macro implements the creation of space charge density maps from hit files.

/*
  Macro to create random space-charge density distributions from simulated TPC hits and calculated the corresponding integrated digital currents. The density distributions are used to calculated the space-charge distortions.

  Parallization (omp) can only be used if compiled with g++ and not with ROOT! Uncommented for now
  Example:
g++ -o createSCHistosFromHits createSCHistosFromHits.C -I ~/alice/sw/osx_x86-64/FairLogger/latest/include -L ~/alice/sw/osx_x86-64/FairLogger/latest/lib -I /Users/matthias/alice/sw/osx_x86-64/FairRoot/latest/include/  -I$ROOTSYS/include -L$ROOTSYS/lib -lRIO -lCore -std=c++17 -I$O2_ROOT/include -L$O2_ROOT/lib -I$O2_ROOT/include/GPU -I /Users/matthias/alice/sw/osx_x86-64/Vc/latest/include/ -lO2TPCBase -lO2TPCSimulation -lO2CommonUtils -lHist -lMathCore -O3 -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib -lomp -lfairlogger -lboost_system -lboost_filesystem

  Input:
    - One o2sim_HitsTPC.root file for first studies
    - List of o2sim_HitsTPC.root files
    - Number of ion pile-up events to be used
    - Ion drift velocity

  Outputs:
    - CalDet Objects with 3D-IDCs, std::vector<float> with 1D-IDCs, std::vector<float> with 0D-IDCs (the vector has size 1),
      - Granularity?
        - Rows, pads at first
    - Histograms with space-charge density
      - Granularity
        - Different for IDCs
        - Same as distortions
    - Objects of type SpaceCharge<> containing the density, global distortions, global corrections

  Algorithm:
  1) Choose n events randomly from list of o2sim_HitsTPC.root files
    - n = number of ion pile-up events
    - Assign global indices to single files

  2) Place r-phi projection of hits randomly in z
    - Take into account A, C side
    - Electron transport
      - Added: diffusion, average space-charge distortions (if provided)
      - Could be added: static distortions, electron drift-time
    - Ion transport
      - Along straight lines
      - Could be added: average space-charge distortions, static distortions
    - Space charge:
      - Apply epsilon
      - Convert to charge (C)
      - epsilon variations from IBF-map (has to be provided as input)
    - IDC:
      - Apply effective gain and digitize
      - Convert z position to time of IDC measurement
      - Gain variations from gain-map (has to be provided as input)
    - Grouping of IDCs has not yet be determined
      - Same or different for IDC and SC density?
      - Integration over 1 ms steps in time
      - r, phi: rows, pads
        - Average over several rows, pads

  3) Primary ionization
    - Transport of ions
      - Along straight lines
        - Could be added: average space-charge distortions, static distortions
      - 3D hit positions as starting points
      - Calculate end points using z position of IBF slice
        - Could be added: electron drift-time

  4) Calculate distortions and corrections
    - Corrections to be used for calibration studies
      - Active area / volume only?
        - Dead regions due to radial distortions
        - Do we care about |eta|>0.9 ?

  TODO:
    - replace hard coded constants (named constXXX and global constants) by O2 constants
    - CalDet objects for IDCs?
      - Array of nZ objects, one for each 1 ms
        - Variation of ion drift time in IDCs and SC density?
    - Make many smaller (~1000 event) CalDets / histos / maps and combine later? DONE
    - For the final data for ML, first use only A side values for IDCs, density, corrections
      - Later provide data for both sides? Fixed boundary at z = 0 required.

  Usage:
    .L $SC/createSCHistosFromHits.C+

    root -b -l -q $SC/createSCHistosFromHits.C+
*/

#if !defined(__CLING__) || defined(__ROOTCLING__)
// root includes
#include "TFile.h"
#include "TH3F.h"
#include "TMath.h"
#include "TObjArray.h"
#include "TRandom.h"
#include "TString.h"
#include "TSystem.h"
#include "TTree.h"

// O2 includes
#include "CommonUtils/TreeStreamRedirector.h"
#include "MathUtils/Utils.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/CRU.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/PadPos.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterGEM.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/CDBInterface.h"
#include "TPCSimulation/ElectronTransport.h"
#include "TPCSimulation/GEMAmplification.h"
#include "TPCSimulation/SAMPAProcessing.h"
#include "TPCSimulation/Point.h"
#include "TPCSpaceCharge/SpaceCharge.h"
#include "TPCSpaceCharge/PoissonSolverHelpers.h"
#include "DataFormatsTPC/Defs.h"
#include "TPCSpaceCharge/SpaceChargeHelpers.h"

#include <chrono>
// #include <omp.h> // comment in if using multithreading
#endif

using namespace o2::tpc;
using CalPad = CalDet<float>;
using CalPadArr = CalArray<float>;

static constexpr unsigned short NR = 129;   // grid in r
static constexpr unsigned short NZ = 129;   // grid in z
static constexpr unsigned short NPHI = 180; // grid in phi

// Gas parameters
// const float mK0 = 2.92f;            // in cm^2 / (Vs), reduced ion mobility K0 for Ne-CO2-N2 (90-10-5) with H2O content 130 ppm. Deisting thesis page 88
// const float mTNull = 273.15f;       // in K, zero temperature
// const float mTGas = 294.15f;        // in K, TPC gas temperature corresponding to 21 degrees Celsius
// const float mVDriftIon = 1.2577968; // in cm / ms, ion drift velocity = K0 * TGas / TNull * 1 atm / Pmeasured * Ez0 * 1e-3. For Ne-CO2-N2 (90-10-5) with H2O content 130 ppm at Pmeasured = 1 atm.

// TPC parameters
// const float mEz0 = 400.f;                                  // in V / cm, nominal drift field in z direction
const float mZROC = o2::tpc::TPCParameters<double>::TPCZ0;     // absolute - position of G1T
const float mRIFC = o2::tpc::TPCParameters<double>::IFCRADIUS; // inner field cage radius in cm
const float mROFC = o2::tpc::TPCParameters<double>::OFCRADIUS; // outer field cage radius in cm
const float mOmegatau = 0.32f;

const char* outfnameHists = "spaceChargeDensityHist"; // name of the output file for the histograms
const char* outfnameIDC = "spaceChargeDensityIDC";    // name of the output file for the IDCs
const char* hisSCRandomName = "hisSCRandom";          // name of the histogram of the combined space charge density of IBF and PI
const char* hisSCIBFRandomName = "hisIBF";            // name of the histogram of the space charge density of IBF
const char* hisSCPIRandomName = "hisPI";              // name of the histogram of the space charge density of PI

CalPad loadMap(std::string mapfile, std::string mapName);

float get0DIDCs(const std::vector<float>& oneDIDC);
float get1DIDCs(const CalPad& calPad, const o2::tpc::Side side);

void scale(TH3& hist, const float fac);
const std::string getNameSide(const o2::tpc::Side side, const char* name);
Side getSide(const float z) { return z < 0 ? Side::C : Side::A; }
int getSideStart(const int sides);
int getSideEnd(const int sides);

/// Create SC density histograms and IDC containers from simulated TPC hits
/// An interaction rate of 50 kHz is assumed. Therefore, the ion drift time determines the number of ion pile-up events.
/// \param ionDriftTime ion drift time in ms. The value determines the number of bins in z/time direction of the histograms and IDCs ( 1 bin / ms / side).
/// \param nEvIon number of ion pile-up events
/// \param debug debug info streaming level
/// \param sides set which sides will be simulated. sides=0: A- and C-Side, sides=1: A-Side only, sides=2: C-Side only
/// \param inputfolder folder to the directory where the gain and epsilon map is stored. If an average distortion map is used, the distortion.root file should also be located there
/// \param distortionType sets the type of the electron distortions: 0->no distortions of electrons are applied, 1->average distortion of electrons. Distortions can be created by the makeDistortionsCorrections() function.
/// \nPhiBins number of phi bins the sc density histograms
/// \nRBins number of phi bins the sc density histograms
/// \nZBins number of phi bins the sc density histograms
void createSCHistosFromHits(const int ionDriftTime = 200, const int nEvIon = 1, const int sides = 0, const char* inputfolder = "", const int distortionType = 0, const int nPhiBins = 720, const int nRBins = 257, const int nZBins = 514, const std::array<float, GEMSTACKSPERSECTOR> gainStackScaling = std::array<float, GEMSTACKSPERSECTOR>{1, 1, 1, 1} /*, const int nThreads = 1*/)
{
  o2::tpc::SpaceCharge<double>::setGrid(NZ, NR, NPHI);

  // load average distortions of electrons
  SpaceCharge<double> spacecharge;
  if (distortionType == 1) {
    const std::string inpFileDistortions = Form("%sdistortions.root", inputfolder);
    TFile fInp(inpFileDistortions.data(), "READ");
    for (int iside = getSideStart(sides); iside < getSideEnd(sides); ++iside) {
      o2::tpc::Side side = (iside == 0) ? o2::tpc::Side::A : o2::tpc::Side::C;
      spacecharge.setGlobalDistortionsFromFile(fInp, side);
    }
  }

  auto startTotal = std::chrono::high_resolution_clock::now();
  gRandom->SetSeed(0);
  std::cout << "Seed is: " << gRandom->GetSeed() << std::endl;

  auto& cdb = CDBInterface::instance();
  cdb.setUseDefaults();

  const Mapper& mapper = Mapper::instance();

  GEMAmplification& gemAmplification = GEMAmplification::instance();
  gemAmplification.updateParameters();

  ElectronTransport& electronTransport = ElectronTransport::instance();
  electronTransport.updateParameters();

  auto& eleParam = ParameterElectronics::Instance();
  SAMPAProcessing& sampaProcessing = SAMPAProcessing::instance();
  sampaProcessing.updateParameters();

  const int nShapedPoints = eleParam.NShapedPoints;
  std::vector<float> signalArray;
  signalArray.resize(nShapedPoints);

  // load gain map
  const std::string gainMapFile = Form("%sGainMap.root", inputfolder);
  const std::string gainMapName = "GainMap";
  const CalPad gainMap = loadMap(gainMapFile, gainMapName);

  // load ibf map
  const std::string ibfMapFile = Form("%sIBFMap.root", inputfolder);
  const std::string ibfMapName = "Gain";
  const CalPad ibfMap = loadMap(ibfMapFile, ibfMapName);

  const std::string hitFileList = Form("%so2sim_HitsTPC.list", inputfolder);
  TObjArray* arrHitFiles = (TObjArray*)gSystem->GetFromPipe(TString::Format("cat %s", hitFileList.data()).Data()).Tokenize("\n");
  const int nHitFiles = arrHitFiles->GetEntries(); // number of files with TPC hits
  std::cout << "Number of Hit Files: " << nHitFiles << std::endl;

  std::array<TH3F, SIDES> hisSCRandom{TH3F(getNameSide(Side::A, hisSCRandomName).data(), getNameSide(Side::A, hisSCRandomName).data(), nPhiBins, 0, ::TWOPI, nRBins, mRIFC, mROFC, nZBins, 0, mZROC),
                                      TH3F(getNameSide(Side::C, hisSCRandomName).data(), getNameSide(Side::C, hisSCRandomName).data(), nPhiBins, 0, ::TWOPI, nRBins, mRIFC, mROFC, nZBins, -mZROC, 0)};

  std::array<TH3F, SIDES> hisIBF{TH3F(getNameSide(Side::A, hisSCIBFRandomName).data(), getNameSide(Side::A, hisSCIBFRandomName).data(), nPhiBins, 0, ::TWOPI, nRBins, mRIFC, mROFC, nZBins, 0, mZROC),
                                 TH3F(getNameSide(Side::C, hisSCIBFRandomName).data(), getNameSide(Side::C, hisSCIBFRandomName).data(), nPhiBins, 0, ::TWOPI, nRBins, mRIFC, mROFC, nZBins, -mZROC, 0)};

  std::array<TH3F, SIDES> hisPI{TH3F(getNameSide(Side::A, hisSCPIRandomName).data(), getNameSide(Side::A, hisSCPIRandomName).data(), nPhiBins, 0, ::TWOPI, nRBins, mRIFC, mROFC, nZBins, 0, mZROC),
                                TH3F(getNameSide(Side::C, hisSCPIRandomName).data(), getNameSide(Side::C, hisSCPIRandomName).data(), nPhiBins, 0, ::TWOPI, nRBins, mRIFC, mROFC, nZBins, -mZROC, 0)};

  const int nZBinsSide = ionDriftTime;
  // vector with CalDet objects for IDCs (1 / ms)
  std::vector<CalPad> vecIDC(nZBinsSide);
  for (auto& calpadIDC : vecIDC) {
    calpadIDC = CalPad("IDC", PadSubset::ROC);
  }

  std::vector<CalPad> vecCalCharge(nZBinsSide);
  for (auto& calpadIDC : vecCalCharge) {
    calpadIDC = CalPad("charge", PadSubset::ROC);
  }

  for (int iev = 0; iev < nEvIon; ++iev) {
    std::cout << " " << std::endl;
    std::cout << "event: " << iev + 1 << " from " << nEvIon << " events" << std::endl;
    const int indexHitFile = gRandom->Uniform(0, nHitFiles - 1);
    std::cout << "indexHitFile: " << indexHitFile << std::endl;
    const std::string hitFileName = arrHitFiles->At(indexHitFile)->GetName();
    std::cout << "hitFileName: " << hitFileName.data() << std::endl;

    std::unique_ptr<TFile> hitFile = std::unique_ptr<TFile>(TFile::Open(hitFileName.data()));
    std::unique_ptr<TTree> hitTree = std::unique_ptr<TTree>((TTree*)hitFile->Get("o2sim"));
    const int nEvents = hitTree->GetEntries(); // number of simulated events per hit file

    // vector of HitGroups per sector
    std::vector<::HitGroup>* arrSectors[::Sector::MAXSECTOR];
    for (int isec = 0; isec < ::Sector::MAXSECTOR; ++isec) {
      arrSectors[isec] = nullptr;
      std::stringstream sectornamestr;
      sectornamestr << "TPCHitsShiftedSector" << isec;
      hitTree->SetBranchAddress(sectornamestr.str().c_str(), &arrSectors[isec]);
    }

    // 1) Choose n events randomly from list of o2sim_HitsTPC.root files
    const int indexEv = gRandom->Uniform(0, nEvents - 1);
    std::cout << "index of event which is chosen: " << indexEv << std::endl;
    hitTree->GetEntry(indexEv);

    // randomize ion drift length for position of ibf disk and dirft of primary ionization
    const float driftLIons = gRandom->Uniform(0., mZROC);

    // rotate event by random phi angle
    const float phiRot = gRandom->Uniform(0, ::TWOPI);

    // z position of ions
    const float zIonsIBF = mZROC - driftLIons;
    const int zbinIDC = static_cast<int>(zIonsIBF / mZROC * ionDriftTime);

    // in case driftLIons = 0 avoid seg fault
    if (zbinIDC == nZBinsSide) {
      continue;
    }

    int startSec = 0;
    int endSec = ::Sector::MAXSECTOR;

    // set sector loop depending on the side which was set
    if (sides == 1) { // A-side
      endSec /= 2;
    } else if (sides == 2) { // C-side
      startSec = endSec / 2;
    }

    // #pragma omp parallel for num_threads(nThreads) // comment in for using multi threading
    for (int isec = startSec; isec < endSec; ++isec) { // loop over sectors
      auto vecTracks = arrSectors[isec];
      for (auto& hitsTrack : *vecTracks) {
        for (size_t ihit = 0; ihit < hitsTrack.getSize(); ++ihit) {
          const auto& elHit = hitsTrack.getHit(ihit);
          GlobalPosition3D posHit(elHit.GetX(), elHit.GetY(), elHit.GetZ());
          const int nEle = static_cast<int>(elHit.GetEnergyLoss());
          if (nEle <= 0) {
            continue;
          }

          // phi rotation of hit
          const float rHit = posHit.rho();
          float phiHit = posHit.phi() + phiRot;
          o2::math_utils::bringTo02PiGen(phiHit);
          posHit.SetX(rHit * std::cos(phiHit));
          posHit.SetY(rHit * std::sin(phiHit));

          // z position of ions
          float zIonsPI = std::abs(posHit.Z()) - driftLIons;
          float zIonsIBFTmp = zIonsIBF;
          if (posHit.Z() < 0) {
            zIonsIBFTmp *= -1;
            zIonsPI *= -1;
          }

          // Primary ionization
          const Side side = getSide(posHit.Z());
          if (std::signbit(zIonsPI) == std::signbit(posHit.Z())) {
            const auto binPhi = hisSCRandom[side].GetXaxis()->FindBin(phiHit);
            const auto binR = hisSCRandom[side].GetYaxis()->FindBin(rHit);
            const auto binZ = hisSCRandom[side].GetZaxis()->FindBin(zIonsPI);
            const auto globBin = hisSCRandom[side].GetBin(binPhi, binR, binZ);
            hisSCRandom[side].AddBinContent(globBin, nEle);
            hisPI[side].AddBinContent(globBin, nEle);
          }

          // apply distortion of electron if specified
          if (distortionType == 1) {
            spacecharge.distortElectron(posHit);
            if (side != getSide(posHit.Z())) {
              posHit.SetZ(side == Side::A ? 0.1f : -0.1f);
            }
          }

          // IBF: Place r-phi projection of hits randomly in z
          float driftTimeEle = 0.f;
          for (int iele = 0; iele < nEle; iele++) {
            const GlobalPosition3D posHitDiff = electronTransport.getElectronDrift(posHit, driftTimeEle);
            float phiHitDiff = posHitDiff.phi();
            o2::math_utils::bringTo02PiGen(phiHitDiff);

            const DigitPos digiPadPos = mapper.findDigitPosFromGlobalPosition(posHitDiff);
            if (!digiPadPos.isValid()) {
              continue;
            }

            // Attachment
            if (electronTransport.isElectronAttachment(driftTimeEle)) {
              continue;
            }

            const auto padPos = digiPadPos.getPadPos();
            const auto row = static_cast<size_t>(padPos.getRow());
            const auto pad = static_cast<size_t>(padPos.getPad());
            const CRU cru = digiPadPos.getCRU();

            const int gain = static_cast<int>(gemAmplification.getEffectiveStackAmplification() * gainMap.getValue(cru, row, pad));
            if (gain == 0) {
              continue;
            }
            const int epsilon = static_cast<int>(gainStackScaling[cru.gemStack()] * gain * ibfMap.getValue(cru, row, pad) * 0.01); // IBF value is in % -> convert to absolute value

            const Side sideIBF = getSide(zIonsIBFTmp);
            const auto binPhi = hisSCRandom[sideIBF].GetXaxis()->FindBin(phiHitDiff);
            const auto binR = hisSCRandom[sideIBF].GetYaxis()->FindBin(posHitDiff.rho());
            const auto binZ = hisSCRandom[sideIBF].GetZaxis()->FindBin(zIonsIBFTmp);

            const auto globBin = hisSCRandom[sideIBF].GetBin(binPhi, binR, binZ);
            hisSCRandom[sideIBF].AddBinContent(globBin, epsilon);
            hisIBF[sideIBF].AddBinContent(globBin, epsilon);

            // convert electrons to ADC signal
            const float adcsignal = sampaProcessing.getADCvalue(static_cast<float>(gain));
            sampaProcessing.getShapedSignal(adcsignal, driftTimeEle, signalArray);
            const float signaladc = std::accumulate(signalArray.begin(), signalArray.end(), 0.f);

            // fill pads with adc value
            auto padPosGlobal = digiPadPos.getGlobalPadPos();
            auto rowRoc = static_cast<size_t>(padPosGlobal.getRow());
            if (cru.roc().isOROC()) {
              rowRoc -= mapper.getNumberOfRowsROC(ROC(0));
            }

            const float charge = vecIDC[zbinIDC].getValue(cru, row, pad) + signaladc;
            ((CalPadArr&)(vecIDC[zbinIDC].getCalArray(static_cast<size_t>(cru.roc().getRoc())))).setValue(rowRoc, pad, charge);

            const float chargeEpsilon = vecCalCharge[zbinIDC].getValue(cru, row, pad) + epsilon;
            ((CalPadArr&)(vecCalCharge[zbinIDC].getCalArray(static_cast<size_t>(cru.roc().getRoc())))).setValue(rowRoc, pad, chargeEpsilon);
          } // electron loop
        }   // hit loop
      }     // track loop
    }       // sector loop

    for (int isec = 0; isec < ::Sector::MAXSECTOR; ++isec) {
      delete arrSectors[isec];
    }
  } // event loop

  // normalize histograms to Q / cm^3 / epsilon0
  for (int iside = getSideStart(sides); iside < getSideEnd(sides); ++iside) {
    o2::tpc::SpaceCharge<float>::normalizeHistoQVEps0(hisSCRandom[iside]);
    o2::tpc::SpaceCharge<float>::normalizeHistoQVEps0(hisIBF[iside]);
    o2::tpc::SpaceCharge<float>::normalizeHistoQVEps0(hisPI[iside]);
  }

  auto stopTotal = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsedTotal = stopTotal - startTotal;
  printf("Total time: %f sec for %d events\n", elapsedTotal.count(), nEvIon);

  TFile fOut(Form("%s.root", outfnameHists), "RECREATE");
  // always store both sides!
  for (int iside = 0; iside < 2; ++iside) {
    hisSCRandom[iside].Write();
    hisIBF[iside].Write();
    hisPI[iside].Write();
  }

  // write idcs in different files than the histogram to be able to use "hadd" for merging
  // make 1D-IDCs and 0D-IDCs
  std::vector<float> idc1DASide(nZBinsSide); // 1D-merged idc
  std::vector<float> idc1DCSide(nZBinsSide); // 1D-merged idc

  std::vector<float> idc0DASide(1); // 0D-merged idc (A single float value cannot be written to file) TODO find better solution
  std::vector<float> idc0DCSide(1); // 0D-merged idc (A single float value cannot be written to file) TODO find better solution

  // calculate 1D IDC
  for (unsigned long iSlice = 0; iSlice < vecIDC.size(); ++iSlice) {
    const auto vecCalArr = vecIDC[iSlice];
    if (sides == 1) { // A-side
      idc1DASide[iSlice] = get1DIDCs(vecCalArr, o2::tpc::Side::A);
    } else if (sides == 2) { // C-side
      idc1DCSide[iSlice] = get1DIDCs(vecCalArr, o2::tpc::Side::C);
    } else {
      idc1DASide[iSlice] = get1DIDCs(vecCalArr, o2::tpc::Side::A);
      idc1DCSide[iSlice] = get1DIDCs(vecCalArr, o2::tpc::Side::C);
    }
  }

  if (sides == 1) { // A-side
    idc0DASide[0] = get0DIDCs(idc1DASide);
  } else if (sides == 2) { // C-side
    idc0DCSide[0] = get0DIDCs(idc1DCSide);
  } else {
    idc0DASide[0] = get0DIDCs(idc1DASide);
    idc0DCSide[0] = get0DIDCs(idc1DCSide);
  }

  std::cout << "output path is: " << outfnameIDC << std::endl;
  TFile fIDC(Form("%s.root", outfnameIDC), "RECREATE");
  fIDC.WriteObject(&vecIDC, "IDC");
  fIDC.WriteObject(&vecCalCharge, "charge");
  fIDC.WriteObject(&idc1DASide, "IDC_1D_A_Side");
  fIDC.WriteObject(&idc1DCSide, "IDC_1D_C_Side");
  fIDC.WriteObject(&idc0DASide, "IDC_0D_A_Side");
  fIDC.WriteObject(&idc0DCSide, "IDC_0D_C_Side");
}

/// load gain or ibf map
/// \param mapfile file to the map
/// \param mapName name of the object
CalPad loadMap(std::string mapfile, std::string mapName)
{
  TFile f(mapfile.data(), "READ");
  CalPad* map = nullptr;
  f.GetObject(mapName.data(), map);

  if (!map) {
    std::cout << mapfile.data() << " NOT FOUND! RETURNING! " << std::endl;
  }

  f.Close();
  return *map;
}

/// create average IDCs from random maps
/// \param files vetor of paths to files containing the IDCs which will be averaged
/// \param outFile output filename
void makeAverageIDCs(const std::vector<std::string>& files, const char* outFile = outfnameIDC)
{
  // vector containing the path of the relevant files which will be averaged
  std::cout << "merge IDCs for average map" << std::endl;

  // average idc CalPads
  std::vector<CalPad> idc3D;     // 3D-average idc
  std::vector<float> idc1DASide; // 1D-average idc
  std::vector<float> idc1DCSide; // 1D-average idc

  std::vector<float> idc0DASide(1); // 0D-average idc (A single float value cannot be written to file) TODO find better solution
  std::vector<float> idc0DCSide(1); // 0D-average idc (A single float value cannot be written to file) TODO find better solution

  // merge the 3D IDC values
  const int nMaps = files.size();
  for (int iFile = 0; iFile < nMaps; ++iFile) {
    const auto str = files[iFile];
    std::cout << "merging file: " << str.data() << std::endl;

    TFile finp(str.data(), "READ");
    std::vector<CalPad>* idcTmp = nullptr;
    finp.GetObject("IDC", idcTmp);

    // number of z-bins
    const auto nSlices = idcTmp->size();

    // init vector for first file
    if (iFile == 0) {
      idc3D.resize(nSlices);
      for (auto& calpadIDC : idc3D) {
        calpadIDC = CalPad("IDC", PadSubset::ROC);
      }
      idc1DASide.resize(nSlices);
      idc1DCSide.resize(nSlices);
    }

    // merge the idc values
    for (unsigned long iSlice = 0; iSlice < nSlices; ++iSlice) {
      idc3D[iSlice] += (*idcTmp)[iSlice];
    }
    delete idcTmp;
  }

  // IDCs have to be normalized to the number of maps
  const int nSlices = idc3D.size();

  // sum up all z-slices for A- and C-side
  CalPad idcZSlicesSummedTmp("IDC", PadSubset::ROC);
  for (int iSlice = 0; iSlice < nSlices; ++iSlice) {
    idc3D[iSlice] /= nMaps;
    idcZSlicesSummedTmp += idc3D[iSlice];
  }
  // normalize to number of z slices
  idcZSlicesSummedTmp /= nSlices;

  // setting each z-slice to average IDC
  for (int iSlice = 0; iSlice < nSlices; ++iSlice) {
    idc3D[iSlice] = idcZSlicesSummedTmp;
  }

  // calculate 1D IDC
  for (unsigned long iSlice = 0; iSlice < idc3D.size(); ++iSlice) {
    const auto vecCalArr = idc3D[iSlice];
    idc1DASide[iSlice] = get1DIDCs(vecCalArr, o2::tpc::Side::A);
    idc1DCSide[iSlice] = get1DIDCs(vecCalArr, o2::tpc::Side::C);
  }

  // calculate 0D IDC
  idc0DASide[0] = get0DIDCs(idc1DASide);
  idc0DCSide[0] = get0DIDCs(idc1DCSide);

  std::cout << "output path is: " << outFile << std::endl;
  TFile fMergedIDC(outFile, "RECREATE");
  fMergedIDC.WriteObject(&idc3D, "IDC");
  fMergedIDC.WriteObject(&idc1DASide, "IDC_1D_A_Side");
  fMergedIDC.WriteObject(&idc1DCSide, "IDC_1D_C_Side");
  fMergedIDC.WriteObject(&idc0DASide, "IDC_0D_A_Side");
  fMergedIDC.WriteObject(&idc0DCSide, "IDC_0D_C_Side");
}

/// \param histSC input histogram containing the space charge density
/// \param nZ number of z granularity for calculating the distortions
/// \param nR number of r granularity for calculating the distortions
/// \param nPhi number of phi granularity for calculating the distortions
/// \param outFileDistortions path to the file for output distortions (which can be read in from a SpaceCharge object)
/// \param sides set for which sides the distortions/corrections will be calculated. sides=0: A- and C-Side, sides=1: A-Side only, sides=2: C-Side only
template <typename DataT = double>
void makeDistortionsCorrections(const TH3& histSC, const int nZ, const int nR, const int nPhi, const char* outFileDistortions = "distortions.root", const int sides = 0)
{
  o2::tpc::SpaceCharge<double>::setGrid(nZ, nR, nPhi);
  std::cout << "output file: " << outFileDistortions << std::endl;

  o2::tpc::SpaceCharge<DataT> spacecharge(mOmegatau, 1, 1);
  spacecharge.fillChargeDensityFromHisto(histSC);

  // dump distortion object to file if output file is specified
  TFile fOut(outFileDistortions, "RECREATE");
  for (int iSide = getSideStart(sides); iSide < getSideEnd(sides); ++iSide) {
    const Side side = iSide == 0 ? Side::A : Side::C;
    spacecharge.calculateDistortionsCorrections(side, true);
    spacecharge.dumpGlobalCorrections(fOut, side);
    spacecharge.dumpGlobalDistortions(fOut, side);
    spacecharge.dumpLocalCorrections(fOut, side);
    spacecharge.dumpLocalDistCorrVectors(fOut, side);
    spacecharge.dumpDensity(fOut, side);
    spacecharge.dumpPotential(fOut, side);
    spacecharge.dumpElectricFields(fOut, side);
  }
}

/// make average distortion map from random maps for histograms for A or C side
/// \param files vector with files which contain the random space charge maps
/// \param histNameNoZDep name of the space charge histogram in the root files which dont have a z dependence
/// \param histNameZDep name of the space charge histogram in the root files which have a z dependence (can also be empty)
/// \param outFileName name of the output file
/// \param outFile output file where the histograms will be written to
void makeAverageDensityMap(const std::vector<std::string> files, TFile& outFile, const char* histNameNoZDep, const char* histNameZDep)
{
  const std::string tmphistNameZDep = histNameZDep;

  // 1. loop over the maps and create the average map (still z dependent)
  TH3F averageMapNoZDep; // average sc map which doesnt depend on z (like the IBF)
  TH3F averageMapZDep;   // average sc map which depends on z (like the PI)
  const int nMaps = files.size();
  for (int iFile = 0; iFile < nMaps; ++iFile) {
    const auto str = files[iFile];
    std::cout << "using density map: " << str.data() << std::endl;
    TFile fInp(str.data(), "READ");
    TH3F* densMapTmpNoZDep = (TH3F*)fInp.Get(histNameNoZDep);

    TH3F* densMapTmpZDep = nullptr;
    if (!tmphistNameZDep.empty()) {
      densMapTmpZDep = (TH3F*)fInp.Get(histNameZDep);
    }

    if (iFile == 0) {
      averageMapNoZDep = *densMapTmpNoZDep;
      if (densMapTmpZDep) {
        averageMapZDep = *densMapTmpZDep;
      }
    } else {
      averageMapNoZDep.Add(densMapTmpNoZDep);
      if (densMapTmpZDep) {
        averageMapZDep.Add(densMapTmpZDep);
      }
    }
    delete densMapTmpNoZDep;
    delete densMapTmpZDep;
  }
  const float scaleFac = 1.f / nMaps;
  scale(averageMapNoZDep, scaleFac);
  scale(averageMapZDep, scaleFac);

  // 2.a sum up all z-slices to remove z dependence
  const int nBinsPhi = averageMapNoZDep.GetNbinsX();
  const int nBinsR = averageMapNoZDep.GetNbinsY();
  const int nBinsZ = averageMapNoZDep.GetNbinsZ();

  for (int iPhi = 1; iPhi <= nBinsPhi; ++iPhi) {
    for (int iR = 1; iR <= nBinsR; ++iR) {
      // either A or C side dependent on the input
      const float meanDens = averageMapNoZDep.Integral(iPhi, iPhi, iR, iR, 1, nBinsZ) / nBinsZ; // integral over all z bins for each r and phi bin normalized to number of z slices
      for (int iZ = 1; iZ <= nBinsZ; ++iZ) {
        averageMapNoZDep.SetBinContent(iPhi, iR, iZ, meanDens);
      }
    }
  }

  outFile.cd();
  averageMapNoZDep.Write();
  if (!tmphistNameZDep.empty()) {
    averageMapZDep.Write();

    // calculate final sc density
    averageMapNoZDep.Add(&averageMapZDep);
    const auto side = getSide(averageMapNoZDep.GetZaxis()->GetBinCenter(nBinsZ / 2));
    const std::string nameOut = getNameSide(side, hisSCRandomName);
    averageMapNoZDep.SetTitle(nameOut.data());
    averageMapNoZDep.SetName(nameOut.data());
    averageMapNoZDep.Write();
  }
}

/// \param scaleFactorConst constant scaling factor
/// \param scaleFactorLinear linear scaling factor - z dependent
/// \param scaleFactorParabolic parabolic scaling factor  - z dependent
float getScaleValueZDep(const float scaleFactorConst, const float scaleFactorLinear, const float scaleFactorParabolic, const float driftLength)
{
  const float scaleVal = 1 + scaleFactorConst + scaleFactorLinear * (driftLength - 0.5f) + scaleFactorParabolic * (driftLength - 0.5f) * (driftLength - 0.5f); // scale factor for data augment (constant, linear(drift), quadratic(drift))
  return scaleVal;
}

/// use the average space charge density map, scale it and calculat the corrections
/// \param inpFile input density file
/// \param outFile output scaled density file
/// \param sides set for which sides will be processed. sides=0: A- and C-Side, sides=1: A-Side only, sides=2: C-Side only
/// \param scaleFactorConst constant scaling factor
/// \param scaleFactorLinear linear scaling factor - z dependent
/// \param scaleFactorParabolic parabolic scaling factor  - z dependent
template <typename DataT = double>
void createScaledMeanMap(const std::string inpFile, const std::string outFile, const int sides, const float scaleFactorConst, const float scaleFactorLinear, const float scaleFactorParabolic, const int nZ, const int nR, const int nPhi)
{
  o2::tpc::SpaceCharge<DataT>::setGrid(nZ, nR, nPhi);

  // load the mean histo
  using SC = o2::tpc::SpaceCharge<DataT>;
  SC scScaled(mOmegatau, 1, 1);

  TFile fInp(inpFile.data(), "READ");
  if (sides != 2) {
    scScaled.setDensityFromFile(fInp, Side::A);
  }
  if (sides != 1) {
    scScaled.setDensityFromFile(fInp, Side::C);
  }

  for (int iSide = getSideStart(sides); iSide < getSideEnd(sides); ++iSide) {
    const Side side = iSide == 0 ? Side::A : Side::C;
    for (size_t iZ = 0; iZ < scScaled.getNZVertices(); ++iZ) {
      const float zPos = std::abs(scScaled.getZVertex(iZ, side));
      for (size_t iR = 0; iR < scScaled.getNRVertices(); ++iR) {
        for (size_t iPhi = 0; iPhi < scScaled.getNPhiVertices(); ++iPhi) {
          const DataT density = scScaled.getDensity(iZ, iR, iPhi, side);
          const float driftLength = (mZROC - zPos) / mZROC; // drift relative to full drift
          const float scaleVal = getScaleValueZDep(scaleFactorConst, scaleFactorLinear, scaleFactorParabolic, driftLength);
          scScaled.fillDensity(density * scaleVal, iZ, iR, iPhi, side);
        }
      }
    }
    scScaled.setDensityFilled(side);
  }
  const bool calcLocalVectors = true;

  if (sides != 2) {
    scScaled.calculateDistortionsCorrections(Side::A, calcLocalVectors);
  }
  if (sides != 1) {
    scScaled.calculateDistortionsCorrections(Side::C, calcLocalVectors);
  }

  // dump distortion object to file if output file is specified
  TFile fOut(outFile.data(), "RECREATE");
  for (int iSide = getSideStart(sides); iSide < getSideEnd(sides); ++iSide) {
    const Side side = iSide == 0 ? Side::A : Side::C;
    scScaled.dumpGlobalCorrections(fOut, side);
    scScaled.dumpGlobalDistortions(fOut, side);
    scScaled.dumpLocalCorrections(fOut, side);
    scScaled.dumpLocalDistCorrVectors(fOut, side);
    scScaled.dumpDensity(fOut, side);
    scScaled.dumpPotential(fOut, side);
    scScaled.dumpElectricFields(fOut, side);
  }
}

/// scale the IDCs from the average (input) map
/// \param inpIDCs input IDC File
/// \param outFile output file name
/// \param scaleFac multiply sigma by this value. The resulting scaling is "1 + scaleFac * sigmaScale"
/// \param sigmaScale sigma of the scaling
void scaleIDCs(const char* inpIDCs, const char* outFile, const float scaleFactorConst, const float scaleFactorLinear, const float scaleFactorParabolic)
{
  // const float scaleVal = 1 + scaleFac * sigmaScale;
  std::cout << "scaling IDC map: " << inpIDCs << std::endl;

  TFile finp(inpIDCs, "READ");
  std::vector<CalPad>* idc3D = nullptr;
  finp.GetObject("IDC", idc3D);

  std::vector<float>* idc1DASide = nullptr;
  std::vector<float>* idc1DCSide = nullptr;
  finp.GetObject("IDC_1D_A_Side", idc1DASide);
  finp.GetObject("IDC_1D_C_Side", idc1DCSide);

  // scale the 3d idcs
  const int nZBins = idc3D->size();
  const float zHalfBin = 0.5 * mZROC / nZBins;

  for (int iSlice = 0; iSlice < nZBins; ++iSlice) {
    const float driftLength = (mZROC - mZROC * iSlice / nZBins - zHalfBin) / mZROC; // index 0 is close to CE. Set z coordinate to middle of z-bin
    const float scaleVal = getScaleValueZDep(scaleFactorConst, scaleFactorLinear, scaleFactorParabolic, driftLength);
    (*idc3D)[iSlice] *= scaleVal;
  }

  // scale the 1d idcs
  for (int iSlice = 0; iSlice < nZBins; ++iSlice) {
    const float driftLength = (mZROC - mZROC * iSlice / nZBins - zHalfBin) / mZROC; // index 0 is close to CE
    const float scaleVal = getScaleValueZDep(scaleFactorConst, scaleFactorLinear, scaleFactorParabolic, driftLength);
    (*idc1DASide)[iSlice] *= scaleVal;
    (*idc1DCSide)[iSlice] *= scaleVal;
  }

  // calculate the 0d idcs
  std::vector<float> idc0DASide{get0DIDCs((*idc1DASide))};
  std::vector<float> idc0DCSide{get0DIDCs((*idc1DCSide))};

  std::cout << "output path is: " << outFile << std::endl;
  TFile fMergedIDC(outFile, "RECREATE");
  fMergedIDC.WriteObject(idc3D, "IDC");
  fMergedIDC.WriteObject(idc1DASide, "IDC_1D_A_Side");
  fMergedIDC.WriteObject(idc1DCSide, "IDC_1D_C_Side");
  fMergedIDC.WriteObject(&idc0DASide, "IDC_0D_A_Side");
  fMergedIDC.WriteObject(&idc0DCSide, "IDC_0D_C_Side");

  delete idc3D;
  delete idc1DASide;
  delete idc1DCSide;
}

/// \param calPad create 1D-IDCs from calpad object
/// \side side of the calpad
float get1DIDCs(const CalPad& calPad, const o2::tpc::Side side)
{
  const auto& mapper = Mapper::instance();
  const int nRowsIROC = mapper.getNumberOfRowsROC(ROC(0));

  // values for weighted mean
  float mean = 0;
  float ww = 0;

  // create average IDCs from CalPad. weighted with pad size
  for (ROC roc; !roc.looped(); ++roc) {
    if (roc.side() != side) {
      continue;
    }

    const int nrows = mapper.getNumberOfRowsROC(roc);
    for (int irow = 0; irow < nrows; ++irow) {
      // get pad width and length
      const int irowGlobal = roc.rocType() == o2::tpc::RocType::IROC ? irow : irow + nRowsIROC; // set global pad row
      const int region = o2::tpc::Mapper::REGION[irowGlobal];
      const int npads = mapper.getNumberOfPadsInRowROC(roc, irow);
      for (int ipad = 0; ipad < npads; ++ipad) {
        const auto idc = calPad.getValue(roc, irow, ipad);
        mean += idc * o2::tpc::Mapper::INVPADAREA[region]; //PADAREA[NREGIONS] = inverse pad area
        ++ww;
      }
    }
  }

  mean /= ww;
  return mean;
}

/// \param oneDIDC vector containg the 1D-IDC values for one side
/// \return returns the average of the input vector
float get0DIDCs(const std::vector<float>& oneDIDC)
{
  const float zeroDIDC = std::accumulate(oneDIDC.begin(), oneDIDC.end(), (float)0) / oneDIDC.size();
  return zeroDIDC;
}

// scale an histogram same as TH3::Scale(), but avoiding an error when a lots of bbins are used and the histogram is written to a file
void scale(TH3& hist, const float fac)
{
  for (int iphi = 1; iphi <= hist.GetNbinsX(); ++iphi) {
    for (int ir = 1; ir <= hist.GetNbinsY(); ++ir) {
      for (int iz = 1; iz <= hist.GetNbinsZ(); ++iz) {
        const auto content = hist.GetBinContent(iphi, ir, iz);
        hist.SetBinContent(iphi, ir, iz, content * fac);
      }
    }
  }
}

const std::string getNameSide(const o2::tpc::Side side, const char* name)
{
  const std::string nameTmp = (side == Side::A) ? Form("%s_A", name) : Form("%s_C", name);
  return nameTmp;
}

/// helper function to set the loop over the sides for the tpc
/// \param sides set for which sides the distortions/corrections will be calculated. sides=0: A- and C-Side, sides=1: A-Side only, sides=2: C-Side only
int getSideStart(const int sides)
{
  if (sides == 2) {
    return 1;
  }
  return 0;
}

/// helper function to set the loop over the sides for the tpc
/// \param sides set for which sides the distortions/corrections will be calculated. sides=0: A- and C-Side, sides=1: A-Side only, sides=2: C-Side only
int getSideEnd(const int sides)
{
  if (sides == 1) {
    return 1;
  }
  return 2;
}

/// merge two high granularity space charge density histograms which are separated into the A and the C side (size would be larger than 1GB-> writing to file not possible)
/// \param inputFile input file containing the two histograms
/// \param nameA of the histogram for the A-Side
/// \param nameC of the histogram for the C-Side
TH3F mergeHistos(const char* inputFile = ".", const char* nameA = "hisIBF_A", const char* nameC = "hisIBF_C")
{
  TFile fInp(inputFile, "READ");
  TH3F* hSC = (TH3F*)fInp.Get(nameA);
  if (hSC == nullptr) {
    std::cout << "histogram " << nameA << " not found " << std::endl;
  }
  const int nPhiBinsTmp = hSC->GetXaxis()->GetNbins();
  const int nRBinsTmp = hSC->GetYaxis()->GetNbins();
  const int nZBins = hSC->GetZaxis()->GetNbins();
  const auto phiLow = hSC->GetXaxis()->GetBinLowEdge(1);
  const auto phiUp = hSC->GetXaxis()->GetBinUpEdge(nPhiBinsTmp);
  const auto rLow = hSC->GetYaxis()->GetBinLowEdge(1);
  const auto rUp = hSC->GetYaxis()->GetBinUpEdge(nRBinsTmp);
  const auto zUp = hSC->GetZaxis()->GetBinUpEdge(nZBins);

  // merged histogram
  TH3F hisSCMerged("hisMerged", "hisMerged", nPhiBinsTmp, phiLow, phiUp, nRBinsTmp, rLow, rUp, 2 * nZBins, -zUp, zUp);

  std::cout << "merging histograms" << std::endl;
  for (int iside = 0; iside < 2; ++iside) {
    if (iside == 1) {
      delete hSC;
      hSC = (TH3F*)fInp.Get(nameC);
      if (hSC == nullptr) {
        std::cout << "histogram " << nameC << " not found " << std::endl;
      }
    }
    for (int iz = 1; iz <= nZBins; ++iz) {
      for (int ir = 1; ir <= nRBinsTmp; ++ir) {
        for (int iphi = 1; iphi <= nPhiBinsTmp; ++iphi) {
          const int izTmp = iside == 0 ? nZBins + iz : iz;
          hisSCMerged.SetBinContent(iphi, ir, izTmp, hSC->GetBinContent(iphi, ir, iz));
        }
      }
    }
  }

  delete hSC;
  fInp.Close();
  return hisSCMerged;
}
