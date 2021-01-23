// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

// Physics parameters
const float mEpsilon0 = o2::tpc::TPCParameters<double>::E0 * 0.01; //8.854187817e-14; // vacuum permittivity [A·s/(V·cm)]

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

const char* outfnameHists = "spaceChargeDensityHist"; // name of the output file for the histograms
const char* outfnameIDC = "spaceChargeDensityIDC";    // name of the output file for the IDCs
const char* hisSCRandomName = "hisSCRandom";

CalPad loadMap(std::string mapfile, std::string mapName);
void normalizeHistoQVEps0(TH3& histoIonsPhiRZ);

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void setOmegaTauT1T2(o2::tpc::SpaceCharge<DataT, Nz, Nr, Nphi>& sc);

/// Create SC density histograms and IDC containers from simulated TPC hits
/// An interaction rate of 50 kHz is assumed. Therefore, the ion drift time determines the number of ion pile-up events.
/// \param ionDriftTime ion drift time in ms. The value determines the number of bins in z/time direction of the histograms and IDCs ( 1 bin / ms / side).
/// \param nEvIon number of ion pile-up events
/// \param debug debug info streaming level
/// \param sides set which sides will be simulated. sides=0: A- and C-Side, sides=1: A-Side only, sides=2: C-Side only
/// \param inputfolder folder to the directory where the gain and epsilon map is stored. If an average distortion map is used, the distortion.root file should also be located there
/// \param distortionType sets the type of the electron distortions: 0->no distortions of electrons are applied, 1->average distortion of electrons. Distortions can be created by the makeDistortionsCorrections() function.
void createSCHistosFromHits(const int ionDriftTime = 200, const int nEvIon = 1, const int debug = 1, const int sides = 0, const char* inputfolder = "", const int distortionType = 0 /*, const int nThreads = 1*/)
{
  // load average distortions of electrons
  SpaceCharge<double, 129, 129, 180> spacecharge;
  if (distortionType == 1) {
    const std::string inpFileDistortions = Form("%sdistortions.root", inputfolder);
    TFile fInp(inpFileDistortions.data(), "READ");
    if (sides == 1) { // A-side
      spacecharge.setGlobalDistortionsFromFile(fInp, Side::A);
    } else if (sides == 2) { // C-side
      spacecharge.setGlobalDistortionsFromFile(fInp, Side::C);
    } else {
      spacecharge.setGlobalDistortionsFromFile(fInp, Side::A);
      spacecharge.setGlobalDistortionsFromFile(fInp, Side::C);
    }
  }

  auto startTotal = std::chrono::high_resolution_clock::now();
  gRandom->SetSeed(0);
  std::cout << "Seed is: " << gRandom->GetSeed() << std::endl;

  auto& cdb = CDBInterface::instance();
  cdb.setUseDefaults();

  const static Mapper& mapper = Mapper::instance();

  static GEMAmplification& gemAmplification = GEMAmplification::instance();
  gemAmplification.updateParameters();

  static ElectronTransport& electronTransport = ElectronTransport::instance();
  electronTransport.updateParameters();

  static SAMPAProcessing& sampaProcessing = SAMPAProcessing::instance();
  sampaProcessing.updateParameters();

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

  // histograms for space-charge density
  const int nPhiBins = 360;
  const int nRBins = 257;
  const int nZBins = 257 * 2;

  TH3F hisSCRandom(hisSCRandomName, hisSCRandomName, nPhiBins, 0, ::TWOPI, nRBins, mRIFC, mROFC, nZBins, -mZROC, mZROC);
  TH3F hisIBF("hisIBF", "hisIBF", nPhiBins, 0, ::TWOPI, nRBins, mRIFC, mROFC, nZBins, -mZROC, mZROC);
  TH3F hisPI("hisPI", "hisPI", nPhiBins, 0, ::TWOPI, nRBins, mRIFC, mROFC, nZBins, -mZROC, mZROC);
  TH1F hisEpsilon("hisEpsilon", "hisEpsilon", 50, 0, 50);

  const int nZBinsSide = ionDriftTime;
  // vector with CalDet objects for IDCs (1 / ms)
  std::vector<CalPad> vecIDC(nZBinsSide);
  for (auto& calpadIDC : vecIDC) {
    calpadIDC = CalPad("IDC", PadSubset::ROC);
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
    float zIonsIBF = mZROC - driftLIons;
    const int zbinIDC = static_cast<int>(zIonsIBF / mZROC * ionDriftTime);

    // in case driftLIons = 0 avoid seg fault
    if (zbinIDC == nZBinsSide) {
      continue;
    }

    int startSec = 0;
    int endSec = ::Sector::MAXSECTOR;

    // set sector loop depending on the side which was set
    if (sides == 1) { // A-side
      endSec *= 0.5;
    } else if (sides == 2) { // C-side
      startSec = endSec * 0.5;
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
          if (posHit.Z() < 0) {
            zIonsIBF *= -1;
            zIonsPI *= -1;
          }

          // Primary ionization
          if (std::signbit(zIonsPI) == std::signbit(posHit.Z())) {
            hisSCRandom.Fill(phiHit, rHit, zIonsPI, nEle);
            hisPI.Fill(phiHit, rHit, zIonsPI, nEle);
          }

          // apply distortion of electron if specified
          if (distortionType == 1) {
            spacecharge.distortElectron(posHit);
          }

          // IBF: Place r-phi projection of hits randomly in z
          float driftTimeEle = 0.f;
          int epsilonSum = 0;
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
            const int epsilon = static_cast<int>(gain * ibfMap.getValue(cru, row, pad) * 0.01); // IBF value is in % -> convert to absolute value
            epsilonSum += epsilon;

            hisSCRandom.Fill(phiHitDiff, posHitDiff.rho(), zIonsIBF, epsilon);
            hisIBF.Fill(phiHitDiff, posHitDiff.rho(), zIonsIBF, epsilon);

            // fill pads with adc value
            auto padPosGlobal = digiPadPos.getGlobalPadPos();
            auto rowRoc = static_cast<size_t>(padPosGlobal.getRow());
            if (cru.roc().isOROC()) {
              rowRoc -= mapper.getNumberOfRowsROC(ROC(0));
            }

            const float charge = vecIDC[zbinIDC].getValue(cru, row, pad) + gain;
            ((CalPadArr&)(vecIDC[zbinIDC].getCalArray(static_cast<size_t>(cru.roc().getRoc())))).setValue(rowRoc, pad, charge);
          } // electron loop
          if (nEle > 0) {
            hisEpsilon.Fill(epsilonSum / nEle);
          }
        } // hit loop
      }   // track loop
    }     // sector loop

    for (int isec = 0; isec < ::Sector::MAXSECTOR; ++isec) {
      delete arrSectors[isec];
    }
  } // event loop

  // normalize histograms to Q / cm^3 / epsilon0 and IDC to Q
  normalizeHistoQVEps0(hisSCRandom);
  normalizeHistoQVEps0(hisIBF);
  normalizeHistoQVEps0(hisPI);
  for (auto& calpadIDC : vecIDC) {
    calpadIDC *= TMath::Qe();
  }

  auto stopTotal = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsedTotal = stopTotal - startTotal;
  printf("Total time: %f sec for %d events\n", elapsedTotal.count(), nEvIon);

  o2::utils::TreeStreamRedirector pcstream(Form("%s.root", outfnameHists), "recreate");
  pcstream.GetFile()->cd();
  if (debug > 0) {
    printf("Dumping space-charge density to debug tree...\n");
    for (int iphi = 1; iphi <= hisSCRandom.GetNbinsX(); ++iphi) {
      float phi = hisSCRandom.GetXaxis()->GetBinCenter(iphi);

      for (int ir = 1; ir <= hisSCRandom.GetNbinsY(); ++ir) {
        float r = hisSCRandom.GetYaxis()->GetBinCenter(ir);

        for (int iz = 1; iz <= hisSCRandom.GetNbinsZ(); ++iz) {
          float z = hisSCRandom.GetZaxis()->GetBinCenter(iz);

          float density = hisSCRandom.GetBinContent(iphi, ir, iz);
          float densityIBF = hisIBF.GetBinContent(iphi, ir, iz);
          float densityPI = hisPI.GetBinContent(iphi, ir, iz);

          pcstream << "density"
                   << "iphi=" << iphi
                   << "ir=" << ir
                   << "iz=" << iz
                   << "phi=" << phi
                   << "r=" << r
                   << "z=" << z
                   << "scdensity=" << density
                   << "scibf=" << densityIBF
                   << "scpi=" << densityPI
                   << "\n";
        }
      }
    }
  }
  hisSCRandom.Write();
  hisIBF.Write();
  hisPI.Write();
  hisEpsilon.Write();
  pcstream.Close();

  // write idcs in different files than the histogram to be able to use "hadd" for merging
  // make 1D-IDCs and 0D-IDCs
  std::vector<float> idc1DASide(nZBinsSide); // 1D-merged idc
  std::vector<float> idc1DCSide(nZBinsSide); // 1D-merged idc

  std::vector<float> idc0DASide(1); // 0D-merged idc (A single float value cannot be written to file) TODO find better solution
  std::vector<float> idc0DCSide(1); // 0D-merged idc (A single float value cannot be written to file) TODO find better solution

  // calculate 1D IDC
  for (unsigned long iSlice = 0; iSlice < vecIDC.size(); ++iSlice) {
    const auto vecCalArr = vecIDC[iSlice].getData();
    const int maxrocs = ROC::MaxROC;
    for (int iROC = 0; iROC < maxrocs; ++iROC) {
      ROC roc(iROC);
      const Side side = roc.side();
      if (side == Side::A) {
        // 1D IDC for A side
        idc1DASide[iSlice] += vecCalArr[iROC].getSum();
      } else {
        // 1D IDC for C side
        idc1DCSide[iSlice] += vecCalArr[iROC].getSum();
      }
    }
  }

  // calculate 0D IDC
  idc0DASide[0] = std::accumulate(idc1DASide.begin(), idc1DASide.end(), (float)0);
  idc0DCSide[0] = std::accumulate(idc1DCSide.begin(), idc1DCSide.end(), (float)0);

  std::cout << "output path is: " << outfnameIDC << std::endl;
  TFile fIDC(Form("%s.root", outfnameIDC), "RECREATE");
  fIDC.WriteObject(&vecIDC, "IDC");
  fIDC.WriteObject(&idc1DASide, "IDC_1D_A_Side");
  fIDC.WriteObject(&idc1DCSide, "IDC_1D_C_Side");
  fIDC.WriteObject(&idc0DASide, "IDC_0D_A_Side");
  fIDC.WriteObject(&idc0DCSide, "IDC_0D_C_Side");
}

/// Normalize histogram with number of ions per bin to charge per bin volume
/// \param histoIonsPhiRZ TH3 histogram. Content: number of ions. Axes: phi, r, z
void normalizeHistoQVEps0(TH3& histoIonsPhiRZ)
{
  for (int iphi = 1; iphi <= histoIonsPhiRZ.GetNbinsX(); ++iphi) {
    const float deltaPhi = histoIonsPhiRZ.GetXaxis()->GetBinWidth(iphi);

    for (int ir = 1; ir <= histoIonsPhiRZ.GetNbinsY(); ++ir) {
      const float r0 = histoIonsPhiRZ.GetYaxis()->GetBinLowEdge(ir);
      const float r1 = histoIonsPhiRZ.GetYaxis()->GetBinUpEdge(ir);

      for (int iz = 1; iz <= histoIonsPhiRZ.GetNbinsZ(); ++iz) {
        const float deltaZ = histoIonsPhiRZ.GetZaxis()->GetBinWidth(iz);
        const float volume = deltaPhi * 0.5 * (r1 * r1 - r0 * r0) * deltaZ;

        const float charge = histoIonsPhiRZ.GetBinContent(iphi, ir, iz) * TMath::Qe();
        histoIonsPhiRZ.SetBinContent(iphi, ir, iz, charge / (volume * mEpsilon0));
      }
    }
  }
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

/// merge the idc values from different root files
/// \param files vetor of paths to files containing the IDCs which will be averaged
/// \param outFile output filename
void makeAverageIDCs(const std::vector<std::string>& files, const char* outFile = outfnameIDC)
{
  // vector containing the path of the relevant files which will be merged
  std::cout << "merge IDCs for average map" << std::endl;

  // merged idc CalPads
  std::vector<CalPad> idc3D;     // 3D-merged idc
  std::vector<float> idc1DASide; // 1D-merged idc
  std::vector<float> idc1DCSide; // 1D-merged idc

  std::vector<float> idc0DASide(1); // 0D-merged idc (A single float value cannot be written to file) TODO find better solution
  std::vector<float> idc0DCSide(1); // 0D-merged idc (A single float value cannot be written to file) TODO find better solution

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

  // if the average IDCs are calculated the IDCs have to be normalized to the number of maps
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
    const auto vecCalArr = idc3D[iSlice].getData();
    const int maxrocs = ROC::MaxROC;
    for (int iROC = 0; iROC < maxrocs; ++iROC) {
      ROC roc(iROC);
      const Side side = roc.side();
      if (side == Side::A) {
        // 1D IDC for A side
        idc1DASide[iSlice] += vecCalArr[iROC].getSum();
      } else {
        // 1D IDC for C side
        idc1DCSide[iSlice] += vecCalArr[iROC].getSum();
      }
    }
  }

  // calculate 0D IDC
  idc0DASide[0] = std::accumulate(idc1DASide.begin(), idc1DASide.end(), (float)0);
  idc0DCSide[0] = std::accumulate(idc1DCSide.begin(), idc1DCSide.end(), (float)0);

  std::cout << "output path is: " << outFile << std::endl;
  TFile fMergedIDC(outFile, "RECREATE");
  fMergedIDC.WriteObject(&idc3D, "IDC");
  fMergedIDC.WriteObject(&idc1DASide, "IDC_1D_A_Side");
  fMergedIDC.WriteObject(&idc1DCSide, "IDC_1D_C_Side");
  fMergedIDC.WriteObject(&idc0DASide, "IDC_0D_A_Side");
  fMergedIDC.WriteObject(&idc0DCSide, "IDC_0D_C_Side");
}

/// \param outFileDistortions path to the file for output distortions (which can be read in from a SpaceCharge object)
/// \param sides set for which sides the distortions/corrections will be calculated. sides=0: A- and C-Side, sides=1: A-Side only, sides=2: C-Side only
/// \param inpFile name of the root file containing the space charge density histogram
/// \param histName name of the space charge density histogram in the root file
template <typename DataT = double, size_t nZ = 17, size_t nR = 17, size_t nPhi = 90>
void makeDistortionsCorrections(const char* outFileDistortions = "distortions.root", const int sides = 0, const char* inpFile = "", const char* histName = hisSCRandomName)
{
  TFile fSCDensity(inpFile, "READ");
  std::cout << "input file: " << inpFile << std::endl;
  std::cout << "output file: " << outFileDistortions << std::endl;

  using SC = o2::tpc::SpaceCharge<DataT, nZ, nR, nPhi>;
  SC spacecharge;
  setOmegaTauT1T2<DataT, nZ, nR, nPhi>(spacecharge);
  spacecharge.fillChargeDensityFromFile(fSCDensity, histName);

  if (sides != 2) {
    spacecharge.calculateDistortionsCorrections(Side::A);
  }
  if (sides != 1) {
    spacecharge.calculateDistortionsCorrections(Side::C);
  }

  int sideStart = 0;
  int sideEnd = 2;
  if (sides == 1) {
    sideEnd = 1;
  } else if (sides == 2) {
    sideStart = 1;
  }

  // dump distortion object to file if output file is specified
  TFile fOut(outFileDistortions, "RECREATE");
  for (int iSide = sideStart; iSide < sideEnd; ++iSide) {
    const Side side = iSide == 0 ? Side::A : Side::C;
    spacecharge.dumpGlobalCorrections(fOut, side);
    spacecharge.dumpGlobalDistortions(fOut, side);
    spacecharge.dumpLocalCorrections(fOut, side);
    spacecharge.dumpDensity(fOut, side);
  }
}

/// make average distortion map from random maps
/// \param files vector with files which contain the random space charge maps
/// \param histName name of the space charge histogram in the root files
void makeAverageDensityMap(const std::vector<std::string> files, const char* histName = hisSCRandomName)
{
  // 1. loop over the maps and create the average map (still z dependent)
  TH3F averageMap;
  const int nMaps = files.size();
  for (int iFile = 0; iFile < nMaps; ++iFile) {
    const auto str = files[iFile];
    std::cout << "using density map: " << str.data() << std::endl;
    TFile fInp(str.data(), "READ");
    TH3F* densMapTmp = (TH3F*)fInp.Get(histName);
    if (iFile == 0) {
      averageMap = *densMapTmp;
    } else {
      averageMap.Add(densMapTmp);
    }
    delete densMapTmp;
  }
  averageMap.Scale(1. / nMaps);

  // 2.a sum up all z-slices to remove z dependence
  const int nBinsPhi = averageMap.GetNbinsX();
  const int nBinsR = averageMap.GetNbinsY();
  const int nBinsZ = averageMap.GetNbinsZ();

  for (int iPhi = 1; iPhi <= nBinsPhi; ++iPhi) {
    for (int iR = 1; iR <= nBinsR; ++iR) {
      const int nBinsHalfZ = 0.5 * nBinsZ;

      // C-Side (?)
      const int startBinCSide = 1;
      const int endBinCSide = nBinsHalfZ;
      const float meanDensCSide = averageMap.Integral(iPhi, iPhi, iR, iR, startBinCSide, endBinCSide) / nBinsHalfZ; // integral over all z bins for each r and phi bin normalized to number of z slices
      for (int iZ = startBinCSide; iZ <= endBinCSide; ++iZ) {
        averageMap.SetBinContent(iPhi, iR, iZ, meanDensCSide);
      }

      // A-Side (?)
      const int startBinASide = 0.5 * nBinsZ + 1;
      const int endBinASide = nBinsZ;
      const float meanDensASide = averageMap.Integral(iPhi, iPhi, iR, iR, startBinASide, endBinASide) / nBinsHalfZ; // integral over all z bins for each r and phi bin normalized to number of z slices
      for (int iZ = startBinASide; iZ <= endBinASide; ++iZ) {
        averageMap.SetBinContent(iPhi, iR, iZ, meanDensASide);
      }
    }
  }

  TFile fOut("spaceChargeDensityHist_average.root", "RECREATE");
  averageMap.Write();
  fOut.Close();
}

/// use the average space charge density map, scale it and calculat the corrections
/// \param inpFile input density file
/// \param outFile output scaled density file
/// \param sides set for which sides will be processed. sides=0: A- and C-Side, sides=1: A-Side only, sides=2: C-Side only
/// \param scaleFac multiply sigma by this value. The resulting scaling is "1 + scaleFac * sigmaScale"
/// \param sigmaScale sigma of the scaling
template <typename DataT = double, size_t nZ = 17, size_t nR = 17, size_t nPhi = 90>
void createScaledMeanMap(const std::string inpFile, const std::string outFile, const int sides, const int scaleFac = 1, const float sigmaScale = 0.03f)
{
  // load the mean histo
  using SC = o2::tpc::SpaceCharge<DataT, nZ, nR, nPhi>;
  SC scOriginal;
  SC scScaled;

  TFile fInp(inpFile.data(), "READ");
  if (sides != 2) {
    scOriginal.setDensityFromFile(fInp, Side::A);
  }
  if (sides != 1) {
    scOriginal.setDensityFromFile(fInp, Side::C);
  }

  setOmegaTauT1T2<DataT, nZ, nR, nPhi>(scScaled);
  int sideStart = 0;
  int sideEnd = 2;
  if (sides == 1) {
    sideEnd = 1;
  } else if (sides == 2) {
    sideStart = 1;
  }

  for (int iSide = sideStart; iSide < sideEnd; ++iSide) {
    const Side side = iSide == 0 ? Side::A : Side::C;
    for (size_t iZ = 0; iZ < nZ; ++iZ) {
      for (size_t iR = 0; iR < nR; ++iR) {
        for (size_t iPhi = 0; iPhi < nPhi; ++iPhi) {
          const DataT density = scOriginal.getDensity(iZ, iR, iPhi, side);
          const float scaleVal = 1 + scaleFac * sigmaScale;
          scScaled.fillDensity(density * scaleVal, iZ, iR, iPhi, side);
        }
      }
    }
    scScaled.setDensityFilled(side);
  }

  if (sides != 2) {
    scScaled.calculateDistortionsCorrections(Side::A);
  }
  if (sides != 1) {
    scScaled.calculateDistortionsCorrections(Side::C);
  }

  // dump distortion object to file if output file is specified
  for (int iSide = sideStart; iSide < sideEnd; ++iSide) {
    const Side side = iSide == 0 ? Side::A : Side::C;
    TFile fOut(outFile.data(), "RECREATE");
    scScaled.dumpGlobalCorrections(fOut, side);
    scScaled.dumpGlobalDistortions(fOut, side);
    scScaled.dumpLocalCorrections(fOut, side);
    scScaled.dumpDensity(fOut, side);
  }
}

/// scale the IDCs from the average (input) map
/// \param inpIDCs input IDC File
/// \param outFile output file name
/// \param scaleFac multiply sigma by this value. The resulting scaling is "1 + scaleFac * sigmaScale"
/// \param sigmaScale sigma of the scaling
void scaleIDCs(const char* inpIDCs, const char* outFile, const int scaleFac = 1, const float sigmaScale = 0.03f)
{
  const float scaleVal = 1 + scaleFac * sigmaScale;
  std::cout << "scaling IDC map: " << inpIDCs << std::endl;

  TFile finp(inpIDCs, "READ");
  std::vector<CalPad>* idc3D = nullptr;
  finp.GetObject("IDC", idc3D);

  std::vector<float>* idc1DASide = nullptr;
  std::vector<float>* idc1DCSide = nullptr;
  finp.GetObject("IDC_1D_A_Side", idc1DASide);
  finp.GetObject("IDC_1D_C_Side", idc1DCSide);

  std::vector<float>* idc0DASide = nullptr;
  std::vector<float>* idc0DCSide = nullptr;
  finp.GetObject("IDC_0D_A_Side", idc0DASide);
  finp.GetObject("IDC_0D_C_Side", idc0DCSide);

  // scale the 3d idcs
  for (unsigned long iSlice = 0; iSlice < idc3D->size(); ++iSlice) {
    (*idc3D)[iSlice] *= scaleVal;
  }

  // scale the 1d idcs
  for (unsigned long iSlice = 0; iSlice < idc1DASide->size(); ++iSlice) {
    (*idc1DASide)[iSlice] *= scaleVal;
    (*idc1DCSide)[iSlice] *= scaleVal;
  }

  // scale the 0d idcs
  for (unsigned long iSlice = 0; iSlice < idc0DASide->size(); ++iSlice) {
    (*idc0DASide)[iSlice] *= scaleVal;
    (*idc0DCSide)[iSlice] *= scaleVal;
  }

  std::cout << "output path is: " << outFile << std::endl;
  TFile fMergedIDC(outFile, "RECREATE");
  fMergedIDC.WriteObject(idc3D, "IDC");
  fMergedIDC.WriteObject(idc1DASide, "IDC_1D_A_Side");
  fMergedIDC.WriteObject(idc1DCSide, "IDC_1D_C_Side");
  fMergedIDC.WriteObject(idc0DASide, "IDC_0D_A_Side");
  fMergedIDC.WriteObject(idc0DCSide, "IDC_0D_C_Side");

  delete idc3D;
  delete idc1DASide;
  delete idc1DCSide;
  delete idc0DASide;
  delete idc0DCSide;
}

/// helper function to set omegatau for the space charge class
template <typename DataT = double, size_t nZ = 17, size_t nR = 17, size_t nPhi = 90>
void setOmegaTauT1T2(o2::tpc::SpaceCharge<DataT, nZ, nR, nPhi>& sc)
{
  sc.setOmegaTauT1T2(0.32f, 1, 1);
}
