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

// g++ -o spacecharge ~/alice/O2/Detectors/TPC/spacecharge/macro/calculateDistortionsCorrections.C -I ~/alice/sw/osx_x86-64/FairLogger/latest/include -L ~/alice/sw/osx_x86-64/FairLogger/latest/lib -I$O2_ROOT/include -L$O2_ROOT/lib -lO2TPCSpacecharge -lO2CommonUtils -std=c++17 -I$ROOTSYS/include -L$ROOTSYS/lib -lCore  -L$VC_ROOT/lib -lVc -I$VC_ROOT/include -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib -lomp -O3 -ffast-math -lFairLogger -lRIO
#include "TPCSpaceCharge/SpaceCharge.h"
#include "TPCBase/Mapper.h"
#include <iostream>
#include <chrono>
#include <array>
#include "CommonUtils/TreeStreamRedirector.h"
#include "TFile.h"

template <typename DataT>
void calculateDistortionsAnalytical(const int sides = 0, const bool staticDistGEMFrame = false, const int globalEFieldTypeAna = 1, const int globalDistTypeAna = 1, const int eFieldTypeAna = 1, const int usePoissonSolverAna = 1, const int nSteps = 1, const int simpsonIterations = 3, const int nThreads = -1);

template <typename DataT>
void calculateDistortionsFromHist(const char* path, const char* histoName, const int sides, const int globalEFieldType = 1, const int globalDistType = 1, const int nSteps = 1, const int simpsonIterations = 3, const int nThreads = -1);

int getSideStart(const int sides);
int getSideEnd(const int sides);

int mNz = 129;
int mNr = 129;
int mNphi = 180;
int mBField = 5;

/// \param sides set which sides will be simulated. sides=0: A- and C-Side, sides=1: A-Side only, sides=2: C-Side only
/// \param staticDistGEMFrame enable simulation of static distortions at GEM frame
/// \param nZVertices number of vertices of the grid in z direction
/// \param nRVertices number of vertices of the grid in z direction
/// \param nPhiVertices number of vertices of the grid in z direction
/// \param globalEFieldTypeAna setting for global distortions/corrections:                          0: using electric field, 1: using local dis/corr interpolator
/// \param globalDistTypeAna setting for global distortions:                                        0: standard method,      1: interpolation of global corrections
/// \param eFieldTypeAna setting for electrc field:                                                 0: analytical formula,   1: tricubic interpolator
/// \param usePoissonSolverAna setting for use poisson solver or analytical formula for potential:  0: analytical formula,   1: poisson solver
/// \param nSteps number of which are used for calculation of distortions/corrections per z-bin
/// \param simpsonIterations number of iterations used in the simpson intergration
/// \param nThreads number of threads which are used (if the value is -1 all threads should be used)
template <typename DataT>
void calcDistAna(const int sides = 0, const bool staticDistGEMFrame = false, const unsigned short nZVertices = 129, const unsigned short nRVertices = 129, const unsigned short nPhiVertices = 180, const int globalEFieldTypeAna = 1, const int globalDistTypeAna = 1, const int eFieldTypeAna = 1, const int usePoissonSolverAna = 1, const int nSteps = 1, const int simpsonIterations = 3, const int nThreads = -1)
{
  mNz = nZVertices;
  mNr = nRVertices;
  mNphi = nPhiVertices;
  calculateDistortionsAnalytical<DataT>(sides, staticDistGEMFrame, globalEFieldTypeAna, globalDistTypeAna, eFieldTypeAna, usePoissonSolverAna, nSteps, simpsonIterations, nThreads);
}

/// \param path path to the root file containing the 3D density histogram
/// \param histoName name of the histogram in the root file
/// \param nZVertices number of vertices of the grid in z direction
/// \param nRVertices number of vertices of the grid in z direction
/// \param nPhiVertices number of vertices of the grid in z direction/// \param sides setting which sides will be processed: 0: A- and C-Side, 1: A-Side, 2: C-Side
/// \param globalEFieldType setting for  global distortions/corrections:   0: using electric field, 1: using local dis/corr interpolator
/// \param globalDistType setting for global distortions:                  0: standard method,      1: interpolation of global corrections
/// \param nSteps number of which are used for calculation of distortions/corrections per z-bin
/// \param simpsonIterations number of iterations used in the simpson intergration
/// \param nThreads number of threads which are used (if the value is -1 all threads should be used)
void calcDistFromHist(const char* path, const char* histoName, const unsigned short nZVertices = 129, const unsigned short nRVertices = 129, const unsigned short nPhiVertices = 180, const int sides = 0, const int globalEFieldType = 1, const int globalDistType = 1, const int nSteps = 1, const int simpsonIterations = 3, const int nThreads = -1)
{
  mNz = nZVertices;
  mNr = nRVertices;
  mNphi = nPhiVertices;
  calculateDistortionsFromHist<double>(path, histoName, sides, globalEFieldType, globalDistType, nSteps, simpsonIterations, nThreads);
}

///  \param spaceChargeCalc SpaceCharge object in which the calculations are performed
///  \param anaFields struct containing the analytical electric fields potential, space charge
///  \param fileOut output file where the calculated values are stored
///  \param side side of the TPC
///  \param globalEFieldType settings for global distortions/corrections: 0: using electric field for calculation of global distortions/corrections, 1: using local dis/corr interpolator for calculation of global distortions/corrections
///  \param globalDistType settings for global distortions: 0: standard method (start calculation of global distortion at each voxel in the tpc and follow electron drift to readout -slow-), 1: interpolation of global corrections (use the global corrections to apply an iterative approach to obtain the global distortions -fast-)
///  \param eFieldType setting for the electric field: 0: use analytical formula for the eletrical field for all calculations, 1: use the tricubic interpolator for the electric field
///  \param usePoissonSolver use poisson solver to calculate the potential or get the potential from the analytical formula 0: use analytical formula, 1: use poisson solver to calculate the potential (also calculates Efields using the obtained potential)
template <typename DataT>
void calculateDistortionsCorrectionsAnalytical(o2::tpc::SpaceCharge<DataT>& spaceChargeCalc, o2::tpc::AnalyticalFields<DataT> anaFields, const int globalEFieldType, const int eFieldType, const int globalDistType, const int usePoissonSolver, const bool staticDistGEMFrame)
{
  using timer = std::chrono::high_resolution_clock;

  std::cout << "====== STARTING CALCULATIION OF DISTORTIONS AND CORRECTIONS BY USING A ANALYTICAL FORMULA AS INPUT ======" << std::endl;
  std::cout << "bins in z: " << spaceChargeCalc.getNZVertices() << std::endl;
  std::cout << "bins in r: " << spaceChargeCalc.getNRVertices() << std::endl;
  std::cout << "bins in phi: " << spaceChargeCalc.getNPhiVertices() << std::endl;

  const std::array<std::string, 2> sGlobalType{"Electric fields", "local distortion/correction interpolator"};
  std::cout << "calculation of global distortions and corrections are performed by using: " << sGlobalType[globalEFieldType] << std::endl;

  const std::array<std::string, 2> sGlobalDistType{"Standard method", "interpolation of global corrections"};
  std::cout << "calculation of global distortions performed by following method: " << sGlobalDistType[globalDistType] << std::endl;

  const std::array<std::string, 2> sEfieldType{"analytical formula", "tricubic interpolator"};
  std::cout << "Using the E-fields from: " << sEfieldType[eFieldType] << std::endl;

  const std::array<std::string, 2> sUsePoissonSolver{"analytical formula", "poisson solver"};
  std::cout << "Using the Potential from: " << sUsePoissonSolver[usePoissonSolver] << std::endl;
  std::cout << std::endl;

  const o2::tpc::Side side = anaFields.getSide();
  spaceChargeCalc.setChargeDensityFromFormula(anaFields);

  auto startTotal = timer::now();

  if (usePoissonSolver == 1) {
    spaceChargeCalc.setPotentialBoundaryFromFormula(anaFields);

    if (staticDistGEMFrame) {
      std::cout << "setting static distortions at GEM frame" << std::endl;
      spaceChargeCalc.setDefaultStaticDistortionsGEMFrameChargeUp(side);
    }

    auto start = timer::now();
    spaceChargeCalc.poissonSolver(side);
    auto stop = timer::now();
    std::chrono::duration<float> time = stop - start;
    std::cout << "poissonSolver: " << time.count() << std::endl;
  } else {
    spaceChargeCalc.setPotentialFromFormula(anaFields);
  }

  if (usePoissonSolver == 1) {
    auto start = timer::now();
    spaceChargeCalc.calcEField(side);
    auto stop = timer::now();
    std::chrono::duration<float> time = stop - start;
    std::cout << "electric field calculation: " << time.count() << std::endl;
  } else {
    spaceChargeCalc.setEFieldFromFormula(anaFields);
  }

  const auto numEFields = spaceChargeCalc.getElectricFieldsInterpolator(side);
  auto start = timer::now();
  const auto dist = o2::tpc::SpaceCharge<DataT>::Type::Distortions;
  (eFieldType == 1) ? spaceChargeCalc.calcLocalDistortionsCorrections(dist, numEFields) : spaceChargeCalc.calcLocalDistortionsCorrections(dist, anaFields); // local distortion calculation
  auto stop = timer::now();
  std::chrono::duration<float> time = stop - start;
  std::cout << "local distortions analytical: " << time.count() << std::endl;

  start = timer::now();
  const auto corr = o2::tpc::SpaceCharge<DataT>::Type::Corrections;
  (eFieldType == 1) ? spaceChargeCalc.calcLocalDistortionsCorrections(corr, numEFields) : spaceChargeCalc.calcLocalDistortionsCorrections(corr, anaFields); // local correction calculation
  stop = timer::now();
  time = stop - start;
  std::cout << "local corrections analytical: " << time.count() << std::endl;

  start = timer::now();
  const auto lCorrInterpolator = spaceChargeCalc.getLocalCorrInterpolator(side);
  if (globalEFieldType == 1) {
    spaceChargeCalc.calcGlobalCorrections(lCorrInterpolator);
  } else if (eFieldType == 1) {
    spaceChargeCalc.calcGlobalCorrections(numEFields);
  } else {
    spaceChargeCalc.calcGlobalCorrections(anaFields);
  }

  stop = timer::now();
  time = stop - start;
  std::cout << "global corrections analytical: " << time.count() << std::endl;

  start = timer::now();
  const auto lDistInterpolator = spaceChargeCalc.getLocalDistInterpolator(side);
  if (globalDistType == 0) {
    if (globalEFieldType == 1) {
      spaceChargeCalc.calcGlobalDistortions(lDistInterpolator);
    } else if (eFieldType == 1) {
      spaceChargeCalc.calcGlobalDistortions(numEFields);
    } else {
      spaceChargeCalc.calcGlobalDistortions(anaFields);
    }
  } else {
    const auto globalCorrInterpolator = spaceChargeCalc.getGlobalCorrInterpolator(side);
    spaceChargeCalc.calcGlobalDistWithGlobalCorrIterative(globalCorrInterpolator);
  }
  stop = timer::now();
  time = stop - start;
  std::cout << "global distortions analytical: " << time.count() << std::endl;

  auto stopTotal = timer::now();
  time = stopTotal - startTotal;
  std::cout << "====== everything is done. Total Time: " << time.count() << std::endl;
  std::cout << std::endl;
}

template <typename DataT>
void writeToTree(o2::tpc::SpaceCharge<DataT>& spaceCharge3D, o2::utils::TreeStreamRedirector& pcstream, const o2::tpc::Side side)
{
  for (size_t iPhi = 0; iPhi < spaceCharge3D.getNPhiVertices(); ++iPhi) {
    for (size_t iR = 0; iR < spaceCharge3D.getNRVertices(); ++iR) {
      for (size_t iZ = 0; iZ < spaceCharge3D.getNZVertices(); ++iZ) {
        auto ldistR = spaceCharge3D.getLocalDistR(iZ, iR, iPhi, side);
        auto ldistZ = spaceCharge3D.getLocalDistZ(iZ, iR, iPhi, side);
        auto ldistRPhi = spaceCharge3D.getLocalDistRPhi(iZ, iR, iPhi, side);
        auto lcorrR = spaceCharge3D.getLocalCorrR(iZ, iR, iPhi, side);
        auto lcorrZ = spaceCharge3D.getLocalCorrZ(iZ, iR, iPhi, side);
        auto lcorrRPhi = spaceCharge3D.getLocalCorrRPhi(iZ, iR, iPhi, side);

        auto lvecdistR = spaceCharge3D.getLocalVecDistR(iZ, iR, iPhi, side);
        auto lvecdistZ = spaceCharge3D.getLocalVecDistZ(iZ, iR, iPhi, side);
        auto lvecdistRPhi = spaceCharge3D.getLocalVecDistRPhi(iZ, iR, iPhi, side);
        auto lveccorrR = spaceCharge3D.getLocalVecCorrR(iZ, iR, iPhi, side);
        auto lveccorrZ = spaceCharge3D.getLocalVecCorrZ(iZ, iR, iPhi, side);
        auto lveccorrRPhi = spaceCharge3D.getLocalVecCorrRPhi(iZ, iR, iPhi, side);

        auto distR = spaceCharge3D.getGlobalDistR(iZ, iR, iPhi, side);
        auto distZ = spaceCharge3D.getGlobalDistZ(iZ, iR, iPhi, side);
        auto distRPhi = spaceCharge3D.getGlobalDistRPhi(iZ, iR, iPhi, side);
        auto corrR = spaceCharge3D.getGlobalCorrR(iZ, iR, iPhi, side);
        auto corrZ = spaceCharge3D.getGlobalCorrZ(iZ, iR, iPhi, side);
        auto corrRPhi = spaceCharge3D.getGlobalCorrRPhi(iZ, iR, iPhi, side);

        // distort then correct
        auto radius = spaceCharge3D.getRVertex(iR, side);
        auto z = spaceCharge3D.getZVertex(iZ, side);
        auto phi = spaceCharge3D.getPhiVertex(iPhi, side);
        DataT corrRDistPoint{};
        DataT corrZDistPoint{};
        DataT corrRPhiDistPoint{};

        const DataT zDistorted = z + distZ;
        const DataT radiusDistorted = radius + distR;
        const DataT phiDistorted = spaceCharge3D.regulatePhi(phi + distRPhi / radius, side);
        spaceCharge3D.getCorrectionsCyl(zDistorted, radiusDistorted, phiDistorted, side, corrZDistPoint, corrRDistPoint, corrRPhiDistPoint);
        corrRPhiDistPoint *= radius / radiusDistorted;

        auto eZ = spaceCharge3D.getEz(iZ, iR, iPhi, side);
        auto eR = spaceCharge3D.getEr(iZ, iR, iPhi, side);
        auto ePhi = spaceCharge3D.getEphi(iZ, iR, iPhi, side);
        auto pot = spaceCharge3D.getPotential(iZ, iR, iPhi, side);
        auto charge = spaceCharge3D.getDensity(iZ, iR, iPhi, side);

        auto xPos = spaceCharge3D.getXFromPolar(radius, phi);
        auto yPos = spaceCharge3D.getYFromPolar(radius, phi);

        int nr = spaceCharge3D.getNRVertices();
        int nz = spaceCharge3D.getNZVertices();
        int nphi = spaceCharge3D.getNPhiVertices();
        int iSide = side;

        bool inVolume = true;
        if (zDistorted < spaceCharge3D.getZMin(side) || zDistorted > spaceCharge3D.getZMax(side) || radiusDistorted < spaceCharge3D.getRMin(side) || radiusDistorted > spaceCharge3D.getRMax(side)) {
          inVolume = false;
        }

        const auto& mapper = o2::tpc::Mapper::instance();
        bool onPad = mapper.findDigitPosFromGlobalPosition(o2::tpc::GlobalPosition3D{static_cast<float>(spaceCharge3D.getXFromPolar(radius, phi)), static_cast<float>(spaceCharge3D.getYFromPolar(radius, phi)), static_cast<float>(z)}).isValid();

        pcstream << "distortions"
                 /// numer of bins
                 << "nR=" << nr
                 << "nPhi=" << nphi
                 << "nZ=" << nz
                 // bin indices
                 << "ir=" << iR
                 << "iz=" << iZ
                 << "iphi=" << iPhi
                 // coordinates
                 << "r=" << radius
                 << "z=" << z
                 << "x=" << xPos
                 << "y=" << yPos
                 << "phi=" << phi
                 // local distortions
                 << "ldistR=" << ldistR
                 << "ldistZ=" << ldistZ
                 << "ldistRPhi=" << ldistRPhi
                 // local corrections
                 << "lcorrR=" << lcorrR
                 << "lcorrZ=" << lcorrZ
                 << "lcorrRPhi=" << lcorrRPhi
                 // local distortion vector
                 << "lvecdistR=" << lvecdistR
                 << "lvecdistZ=" << lvecdistZ
                 << "lvecdistRPhi=" << lvecdistRPhi
                 // local correction  vector
                 << "lveccorrR=" << lveccorrR
                 << "lveccorrZ=" << lveccorrZ
                 << "lveccorrRPhi=" << lveccorrRPhi
                 // global distortions
                 << "distR=" << distR
                 << "distZ=" << distZ
                 << "distRPhi=" << distRPhi
                 // global corrections
                 << "corrR=" << corrR
                 << "corrZ=" << corrZ
                 << "corrRPhi=" << corrRPhi
                 // correction after distortion applied (test for consistency)
                 << "corrRDistortedPoint=" << corrRDistPoint
                 << "corrRPhiDistortedPoint=" << corrRPhiDistPoint
                 << "corrZDistortedPoint=" << corrZDistPoint
                 << "inVolume=" << inVolume
                 << "onPad=" << onPad
                 // electric fields etc.
                 << "Er=" << eR
                 << "Ez=" << eZ
                 << "Ephi=" << ePhi
                 << "potential=" << pot
                 << "charge=" << charge
                 << "side=" << iSide
                 << "\n";
      }
    }
  }
}

/// \param globalEFieldTypeAna setting for global distortions/corrections:                          0: using electric field, 1: using local dis/corr interpolator
/// \param globalDistTypeAna setting for global distortions:                                        0: standard method,      1: interpolation of global corrections
/// \param eFieldTypeAna setting for electrc field:                                                 0: analytical formula,   1: tricubic interpolator
/// \param usePoissonSolverAna setting for use poisson solver or analytical formula for potential:  0: analytical formula,   1: poisson solver
/// \param nSteps number of which are used for calculation of distortions/corrections per z-bin
/// \param simpsonIterations number of iterations used in the simpson intergration
template <typename DataT = double>
void calculateDistortionsAnalytical(const int sides, const bool staticDistGEMFrame, const int globalEFieldTypeAna, const int globalDistTypeAna, const int eFieldTypeAna, const int usePoissonSolverAna, const int nSteps, const int simpsonIterations, const int nThreads)
{
  const auto integrationStrategy = o2::tpc::SpaceCharge<DataT>::IntegrationStrategy::Root;
  o2::tpc::SpaceCharge<DataT> spaceCharge3D(mBField, mNz, mNr, mNphi);
  spaceCharge3D.setNStep(nSteps);
  spaceCharge3D.setSimpsonNIteratives(simpsonIterations);
  spaceCharge3D.setNumericalIntegrationStrategy(integrationStrategy);
  if (nThreads != -1) {
    spaceCharge3D.setNThreads(nThreads);
  }

  // write to root file
  o2::utils::TreeStreamRedirector pcstream(TString::Format("distortions_ana_nR%hu_nZ%hu_nPhi%hu_SimpsonsIter%i.root", spaceCharge3D.getNRVertices(), spaceCharge3D.getNZVertices(), spaceCharge3D.getNPhiVertices(), simpsonIterations).Data(), "RECREATE");

  for (int iside = getSideStart(sides); iside < getSideEnd(sides); ++iside) {
    std::cout << "side: " << iside << std::endl;
    o2::tpc::Side side = (iside == 0) ? o2::tpc::Side::A : o2::tpc::Side::C;
    o2::tpc::AnalyticalFields<DataT> anaFields(side);
    calculateDistortionsCorrectionsAnalytical(spaceCharge3D, anaFields, globalEFieldTypeAna, eFieldTypeAna, globalDistTypeAna, usePoissonSolverAna, staticDistGEMFrame);
    pcstream.GetFile()->cd();
    writeToTree(spaceCharge3D, pcstream, side);
    spaceCharge3D.dumpToFile(*pcstream.GetFile(), side);
  }

  pcstream.GetFile()->cd();
  pcstream.Close();
}

/// \param path path to the root file containing the 3D density histogram
/// \param histoName name of the histogram in the root file
/// \param sides setting which sides will be processed: 0: A- and C-Side, 1: A-Side, 2: C-Side
/// \param globalEFieldType setting for  global distortions/corrections: 0: using electric field, 1: using local dis/corr interpolator
/// \param globalDistType setting for global distortions: 0: standard method,      1: interpolation of global corrections
/// \param nSteps number of which are used for calculation of distortions/corrections per z-bin
/// \param simpsonIterations number of iterations used in the simpson intergration
template <typename DataT = double>
void calculateDistortionsFromHist(const char* path, const char* histoName, const int sides, const int globalEFieldType, const int globalDistType, const int nSteps, const int simpsonIterations, const int nThreads)
{
  using SC = o2::tpc::SpaceCharge<DataT>;
  const auto integrationStrategy = o2::tpc::SpaceCharge<DataT>::IntegrationStrategy::SimpsonIterative;
  SC spaceCharge3D(mBField, mNz, mNr, mNphi);
  spaceCharge3D.setNStep(nSteps);
  spaceCharge3D.setSimpsonNIteratives(simpsonIterations);
  spaceCharge3D.setNumericalIntegrationStrategy(integrationStrategy);
  if (nThreads != -1) {
    spaceCharge3D.setNThreads(nThreads);
  }

  // set density from input file
  TFile fileOutSCDensity(path, "READ");
  spaceCharge3D.fillChargeDensityFromFile(fileOutSCDensity, histoName);
  fileOutSCDensity.Close();

  o2::utils::TreeStreamRedirector pcstream(TString::Format("distortions_real_nR%hu_nZ%hu_nPhi%hu_SimpsonsIter%i.root", spaceCharge3D.getNRVertices(), spaceCharge3D.getNZVertices(), spaceCharge3D.getNPhiVertices(), simpsonIterations).Data(), "RECREATE");
  for (int iside = getSideStart(sides); iside < getSideEnd(sides); ++iside) {
    std::cout << "side: " << iside << std::endl;
    o2::tpc::Side side = (iside == 0) ? o2::tpc::Side::A : o2::tpc::Side::C;
    const auto distType = globalDistType == 0 ? SC::GlobalDistType::Standard : SC ::GlobalDistType::Fast;
    spaceCharge3D.setGlobalDistType(distType);
    const auto eType = globalEFieldType == 1 ? SC::GlobalDistCorrMethod::LocalDistCorr : SC::GlobalDistCorrMethod::ElectricalField;
    spaceCharge3D.setGlobalDistCorrMethod(eType);
    const bool calcLocalVectors = true;
    spaceCharge3D.calculateDistortionsCorrections(side, calcLocalVectors);
    // write to root file
    pcstream.GetFile()->cd();
    writeToTree(spaceCharge3D, pcstream, side);
  }
  pcstream.GetFile()->cd();
  pcstream.Close();

  // write global corrections and distortions to file
  std::string_view file = "spacecharge.root";
  if (sides != 2) {
    spaceCharge3D.dumpGlobalDistortions(file, o2::tpc::Side::A, "UPDATE");
    spaceCharge3D.dumpGlobalCorrections(file, o2::tpc::Side::A, "UPDATE");
  }
  if (sides != 1) {
    spaceCharge3D.dumpGlobalDistortions(file, o2::tpc::Side::C, "UPDATE");
    spaceCharge3D.dumpGlobalCorrections(file, o2::tpc::Side::C, "UPDATE");
  }
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
