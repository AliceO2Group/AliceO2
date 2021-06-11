// g++ -o spacecharge ~/alice/O2/Detectors/TPC/spacecharge/macro/calculateDistortionsCorrections.C -I ~/alice/sw/osx_x86-64/FairLogger/latest/include -L ~/alice/sw/osx_x86-64/FairLogger/latest/lib -I$O2_ROOT/include -L$O2_ROOT/lib -lO2TPCSpacecharge -lO2CommonUtils -std=c++17 -I$ROOTSYS/include -L$ROOTSYS/lib -lCore  -L$VC_ROOT/lib -lVc -I$VC_ROOT/include -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib -lomp -O3 -ffast-math -lFairLogger -lRIO
#include "TPCSpaceCharge/SpaceCharge.h"
#include <iostream>
#include <chrono>
#include <array>
#include "CommonUtils/TreeStreamRedirector.h"

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void calculateDistortionsAnalytical(const int globalEFieldTypeAna = 1, const int globalDistTypeAna = 1, const int eFieldTypeAna = 1, const int usePoissonSolverAna = 1, const int nSteps = 1, const int simpsonIterations = 3, const int nThreads = -1);

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void calculateDistortionsFromHist(const char* path, const char* histoName, const int sides, const int globalEFieldType = 1, const int globalDistType = 1, const int nSteps = 1, const int simpsonIterations = 3, const int nThreads = -1);

/// \param gridType granularity of the grid
/// \param globalEFieldTypeAna setting for global distortions/corrections:                          0: using electric field, 1: using local dis/corr interpolator
/// \param globalDistTypeAna setting for global distortions:                                        0: standard method,      1: interpolation of global corrections
/// \param eFieldTypeAna setting for electrc field:                                                 0: analytical formula,   1: tricubic interpolator
/// \param usePoissonSolverAna setting for use poisson solver or analytical formula for potential:  0: analytical formula,   1: poisson solver
/// \param nSteps number of which are used for calculation of distortions/corrections per z-bin
/// \param simpsonIterations number of iterations used in the simpson intergration
/// \param nThreads number of threads which are used (if the value is -1 all threads should be used)
void calcDistAna(const int gridType = 0, const int globalEFieldTypeAna = 1, const int globalDistTypeAna = 1, const int eFieldTypeAna = 1, const int usePoissonSolverAna = 1, const int nSteps = 1, const int simpsonIterations = 3, const int nThreads = -1)
{
  if (gridType == 0) {
    calculateDistortionsAnalytical<double, 129, 129, 180>(globalEFieldTypeAna, globalDistTypeAna, eFieldTypeAna, usePoissonSolverAna, nSteps, simpsonIterations, nThreads);
  } else if (gridType == 1) {
    calculateDistortionsAnalytical<double, 65, 65, 180>(globalEFieldTypeAna, globalDistTypeAna, eFieldTypeAna, usePoissonSolverAna, nSteps, simpsonIterations, nThreads);
  } else if (gridType == 2) {
    calculateDistortionsAnalytical<double, 33, 33, 180>(globalEFieldTypeAna, globalDistTypeAna, eFieldTypeAna, usePoissonSolverAna, nSteps, simpsonIterations, nThreads);
  } else if (gridType == 3) {
    calculateDistortionsAnalytical<double, 17, 17, 90>(globalEFieldTypeAna, globalDistTypeAna, eFieldTypeAna, usePoissonSolverAna, nSteps, simpsonIterations, nThreads);
  } else if (gridType == 4) {
    calculateDistortionsAnalytical<double, 257, 257, 180>(globalEFieldTypeAna, globalDistTypeAna, eFieldTypeAna, usePoissonSolverAna, nSteps, simpsonIterations, nThreads);
  }
}

/// \param path path to the root file containing the 3D density histogram
/// \param histoName name of the histogram in the root file
/// \param gridType granularity of the grid
/// \param sides setting which sides will be processed: 0: A- and C-Side, 1: A-Side, 2: C-Side
/// \param globalEFieldType setting for  global distortions/corrections:   0: using electric field, 1: using local dis/corr interpolator
/// \param globalDistType setting for global distortions:                  0: standard method,      1: interpolation of global corrections
/// \param nSteps number of which are used for calculation of distortions/corrections per z-bin
/// \param simpsonIterations number of iterations used in the simpson intergration
/// \param nThreads number of threads which are used (if the value is -1 all threads should be used)
void calcDistFromHist(const char* path, const char* histoName, const int gridType = 0, const int sides = 0, const int globalEFieldType = 1, const int globalDistType = 1, const int nSteps = 1, const int simpsonIterations = 3, const int nThreads = -1)
{
  if (gridType == 0) {
    calculateDistortionsFromHist<double, 129, 129, 180>(path, histoName, sides, globalEFieldType, globalDistType, nSteps, simpsonIterations, nThreads);
  } else if (gridType == 1) {
    calculateDistortionsFromHist<double, 65, 65, 180>(path, histoName, sides, globalEFieldType, globalDistType, nSteps, simpsonIterations, nThreads);
  } else if (gridType == 2) {
    calculateDistortionsFromHist<double, 33, 33, 180>(path, histoName, sides, globalEFieldType, globalDistType, nSteps, simpsonIterations, nThreads);
  } else if (gridType == 3) {
    calculateDistortionsFromHist<double, 17, 17, 90>(path, histoName, sides, globalEFieldType, globalDistType, nSteps, simpsonIterations, nThreads);
  } else if (gridType == 4) {
    calculateDistortionsFromHist<double, 257, 257, 180>(path, histoName, sides, globalEFieldType, globalDistType, nSteps, simpsonIterations, nThreads);
  } else if (gridType == 5) {
    calculateDistortionsFromHist<double, 257, 257, 360>(path, histoName, sides, globalEFieldType, globalDistType, nSteps, simpsonIterations, nThreads);
  }
}

///  \param spaceChargeCalc SpaceCharge object in which the calculations are performed
///  \param anaFields struct containing the analytical electric fields potential, space charge
///  \param fileOut output file where the calculated values are stored
///  \param side side of the TPC
///  \param globalEFieldType settings for global distortions/corrections: 0: using electric field for calculation of global distortions/corrections, 1: using local dis/corr interpolator for calculation of global distortions/corrections
///  \param globalDistType settings for global distortions: 0: standard method (start calculation of global distortion at each voxel in the tpc and follow electron drift to readout -slow-), 1: interpolation of global corrections (use the global corrections to apply an iterative approach to obtain the global distortions -fast-)
///  \param eFieldType setting for the electric field: 0: use analytical formula for the eletrical field for all calculations, 1: use the tricubic interpolator for the electric field
///  \param usePoissonSolver use poisson solver to calculate the potential or get the potential from the analytical formula 0: use analytical formula, 1: use poisson solver to calculate the potential (also calculates Efields using the obtained potential)
template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void calculateDistortionsCorrectionsAnalytical(o2::tpc::SpaceCharge<DataT, Nz, Nr, Nphi>& spaceChargeCalc, o2::tpc::AnalyticalFields<DataT> anaFields, const int globalEFieldType, const int eFieldType, const int globalDistType, const int usePoissonSolver)
{
  using timer = std::chrono::high_resolution_clock;

  std::cout << "====== STARTING CALCULATIION OF DISTORTIONS AND CORRECTIONS BY USING A ANALYTICAL FORMULA AS INPUT ======" << std::endl;
  std::cout << "bins in z: " << Nz << std::endl;
  std::cout << "bins in r: " << Nr << std::endl;
  std::cout << "bins in phi: " << Nphi << std::endl;

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
  const auto dist = o2::tpc::SpaceCharge<DataT, Nz, Nr, Nphi>::Type::Distortions;
  (eFieldType == 1) ? spaceChargeCalc.calcLocalDistortionsCorrections(dist, numEFields) : spaceChargeCalc.calcLocalDistortionsCorrections(dist, anaFields); // local distortion calculation
  auto stop = timer::now();
  std::chrono::duration<float> time = stop - start;
  std::cout << "local distortions analytical: " << time.count() << std::endl;

  start = timer::now();
  const auto corr = o2::tpc::SpaceCharge<DataT, Nz, Nr, Nphi>::Type::Corrections;
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

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void writeToTree(o2::tpc::SpaceCharge<DataT, Nz, Nr, Nphi>& spaceCharge3D, o2::utils::TreeStreamRedirector& pcstream, const o2::tpc::Side side)
{
  for (size_t iPhi = 0; iPhi < Nphi; ++iPhi) {
    for (size_t iR = 0; iR < Nr; ++iR) {
      for (size_t iZ = 0; iZ < Nz; ++iZ) {
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

        int nr = Nr;
        int nz = Nz;
        int nphi = Nphi;
        int iSide = side;

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
template <typename DataT = double, size_t Nz = 17, size_t Nr = 17, size_t Nphi = 90>
void calculateDistortionsAnalytical(const int globalEFieldTypeAna, const int globalDistTypeAna, const int eFieldTypeAna, const int usePoissonSolverAna, const int nSteps, const int simpsonIterations, const int nThreads)
{
  const auto integrationStrategy = o2::tpc::SpaceCharge<DataT, Nz, Nr, Nphi>::IntegrationStrategy::SimpsonIterative;
  o2::tpc::SpaceCharge<DataT, Nz, Nr, Nphi> spaceCharge3D;
  spaceCharge3D.setOmegaTauT1T2(0.32f, 1, 1);
  spaceCharge3D.setNStep(nSteps);
  spaceCharge3D.setSimpsonNIteratives(simpsonIterations);
  spaceCharge3D.setNumericalIntegrationStrategy(integrationStrategy);
  if (nThreads != -1) {
    spaceCharge3D.setNThreads(nThreads);
  }

  // write to root file
  o2::utils::TreeStreamRedirector pcstream(TString::Format("distortions_ana_nR%lu_nZ%lu_nPhi%lu_SimpsonsIter%i.root", Nr, Nz, Nphi, simpsonIterations).Data(), "RECREATE");
  for (int iside = 0; iside < 2; ++iside) {
    std::cout << "side: " << iside << std::endl;
    o2::tpc::Side side = (iside == 0) ? o2::tpc::Side::A : o2::tpc::Side::C;
    o2::tpc::AnalyticalFields<DataT> anaFields(side);
    calculateDistortionsCorrectionsAnalytical(spaceCharge3D, anaFields, globalEFieldTypeAna, eFieldTypeAna, globalDistTypeAna, usePoissonSolverAna);
    pcstream.GetFile()->cd();
    writeToTree(spaceCharge3D, pcstream, side);
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
template <typename DataT = double, size_t Nz = 17, size_t Nr = 17, size_t Nphi = 90>
void calculateDistortionsFromHist(const char* path, const char* histoName, const int sides, const int globalEFieldType, const int globalDistType, const int nSteps, const int simpsonIterations, const int nThreads)
{
  using SC = o2::tpc::SpaceCharge<DataT, Nz, Nr, Nphi>;
  const auto integrationStrategy = o2::tpc::SpaceCharge<DataT, Nz, Nr, Nphi>::IntegrationStrategy::SimpsonIterative;
  SC spaceCharge3D;
  spaceCharge3D.setOmegaTauT1T2(0.32f, 1, 1);
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

  o2::utils::TreeStreamRedirector pcstream(TString::Format("distortions_real_nR%lu_nZ%lu_nPhi%lu_SimpsonsIter%i.root", Nr, Nz, Nphi, simpsonIterations).Data(), "RECREATE");
  int iSideStart = 0;
  int iSideEnd = 2;
  if (sides == 1) {
    // a side only
    iSideEnd = 1;
  } else if (sides == 2) {
    // c side only
    iSideStart = 1;
  }
  for (int iside = iSideStart; iside < iSideEnd; ++iside) {
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
  TFile fOut("spacecharge.root", "RECREATE");
  if (sides != 2) {
    spaceCharge3D.dumpGlobalDistortions(fOut, o2::tpc::Side::A);
    spaceCharge3D.dumpGlobalCorrections(fOut, o2::tpc::Side::A);
  }
  if (sides != 1) {
    spaceCharge3D.dumpGlobalDistortions(fOut, o2::tpc::Side::C);
    spaceCharge3D.dumpGlobalCorrections(fOut, o2::tpc::Side::C);
  }
  fOut.Close();
}
