// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file createResidualDistortionObject.C
/// \brief This macro creates a SpaceCharge object with residual distortions from a fluctuating and an average space-charge density histogram and stores it in a file.
/// \author Ernst Hellbär, Goethe-Universität Frankfurt, ernst.hellbar@cern.ch

#include <cmath>

#include "TFile.h"
#include "TH3.h"
#include "TMatrixD.h"
#include "TString.h"

#include "AliTPCSpaceCharge3DCalc.h"
#include "CommonConstants/MathConstants.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "TPCSimulation/SpaceCharge.h"

using namespace o2::tpc;

// function declarations
AliTPCSpaceCharge3DCalc* createSpaceCharge(TH3* hisSCDensity, int bSign, int nR, int nPhi, int nZ);
void fillDistortionLookupMatrices(AliTPCSpaceCharge3DCalc* spaceChargeCalc, AliTPCSpaceCharge3DCalc* spaceChargeCalcAvg, TMatrixD** matrixDistDrA, TMatrixD** matrixDistDrphiA, TMatrixD** matrixDistDzA, TMatrixD** matrixDistDrC, TMatrixD** matrixDistDrphiC, TMatrixD** matrixDistDzC);
void makeDebugTreeResiduals(AliTPCSpaceCharge3DCalc* calcFluc, AliTPCSpaceCharge3DCalc* calcAvg, SpaceCharge* spaceChargeRes);

/// Create a SpaceCharge object with residual distortions from a fluctuating and an average space-charge density histogram and write it to a file.
/// \param bSign sign of the B-field: -1 = negative, 0 = no B-field, 1 = positive
/// \param nR number of bins in r
/// \param nPhi number of bins in phi
/// \param nZ number of bins in z
/// \param pathToHistoFile path to a file with the fluctuating and average histograms
/// \param histoFlucName name of the fluctuating histogram
/// \param histoAvgName name of the average histogram
void createResidualDistortionObject(int bSign, int nR = 129, int nPhi = 144, int nZ = 129, const char* pathToHistoFile = "InputSCDensityHistograms_8000events.root", const char* histoFlucName = "inputSCDensity3D_8000_0", const char* histoAvgName = "inputSCDensity3D_8000_avg", bool debug = false)
{
  /*
     Usage:
     root -l -b -q createResidualDistortionObject.C+\(-1,129,144,129,\"InputSCDensityHistograms_8000events.root\",\"inputSCDensity3D_8000_0\",\"inputSCDensity3D_8000_avg\",true\)
   */
  TFile* fileHistos = TFile::Open(pathToHistoFile);
  auto histoFluc = fileHistos->Get<TH3>(histoFlucName);
  auto histoAvg = fileHistos->Get<TH3>(histoAvgName);

  TFile* fileOutput = TFile::Open("ResidualDistortions.root", "recreate");

  // Calculate fluctuating and average distortion and correction lookup tables
  AliTPCSpaceCharge3DCalc* scCalcFluc = createSpaceCharge(histoFluc, bSign, nR, nPhi, nZ);
  AliTPCSpaceCharge3DCalc* scCalcAvg = createSpaceCharge(histoAvg, bSign, nR, nPhi, nZ);

  // Create matrices and fill them with residual distortions. Create SpaceCharge object, assign residual distortion matrices to it and store it in the output file.
  TMatrixD** matrixResDistDrA = new TMatrixD*[nPhi];
  TMatrixD** matrixResDistDrphiA = new TMatrixD*[nPhi];
  TMatrixD** matrixResDistDzA = new TMatrixD*[nPhi];
  TMatrixD** matrixResDistDrC = new TMatrixD*[nPhi];
  TMatrixD** matrixResDistDrphiC = new TMatrixD*[nPhi];
  TMatrixD** matrixResDistDzC = new TMatrixD*[nPhi];
  for (int iphi = 0; iphi < nPhi; ++iphi) {
    matrixResDistDrA[iphi] = new TMatrixD(nR, nZ);
    matrixResDistDrphiA[iphi] = new TMatrixD(nR, nZ);
    matrixResDistDzA[iphi] = new TMatrixD(nR, nZ);
    matrixResDistDrC[iphi] = new TMatrixD(nR, nZ);
    matrixResDistDrphiC[iphi] = new TMatrixD(nR, nZ);
    matrixResDistDzC[iphi] = new TMatrixD(nR, nZ);
  }
  fillDistortionLookupMatrices(scCalcFluc, scCalcAvg, matrixResDistDrA, matrixResDistDrphiA, matrixResDistDzA, matrixResDistDrC, matrixResDistDrphiC, matrixResDistDzC);
  SpaceCharge spaceChargeRes(nR, nPhi, nZ);
  spaceChargeRes.setDistortionLookupTables(matrixResDistDrA, matrixResDistDrphiA, matrixResDistDzA, matrixResDistDrC, matrixResDistDrphiC, matrixResDistDzC);
  fileOutput->WriteObject(&spaceChargeRes, "spaceChargeRes");

  if (debug) {
    makeDebugTreeResiduals(scCalcFluc, scCalcAvg, &spaceChargeRes);
  }
}

/// Create AliTPCSpaceCharge3DCalc object from a space-charge density histogram
/// \param hisSCDensity input space-charge density histogram
/// \param bSign sign of the B-field: -1 = negative, 0 = no B-field, 1 = positive
/// \param nR number of bins in r
/// \param nPhi number of bins in phi
/// \param nZ number of bins in z
/// \return pointer to AliTPCSpaceCharge3DCalc object
AliTPCSpaceCharge3DCalc* createSpaceCharge(TH3* hisSCDensity, int bSign, int nR, int nPhi, int nZ)
{
  AliTPCSpaceCharge3DCalc* spaceCharge = new AliTPCSpaceCharge3DCalc(nR, nZ, nPhi, 2, 3, 0);
  std::unique_ptr<std::unique_ptr<TMatrixD>[]> mMatrixChargeA = std::make_unique<std::unique_ptr<TMatrixD>[]>(nPhi);
  std::unique_ptr<std::unique_ptr<TMatrixD>[]> mMatrixChargeC = std::make_unique<std::unique_ptr<TMatrixD>[]>(nPhi);
  for (int iphi = 0; iphi < nPhi; ++iphi) {
    mMatrixChargeA[iphi] = std::make_unique<TMatrixD>(nR, nZ);
    mMatrixChargeC[iphi] = std::make_unique<TMatrixD>(nR, nZ);
  }
  spaceCharge->GetChargeDensity((TMatrixD**)mMatrixChargeA.get(), (TMatrixD**)mMatrixChargeC.get(), hisSCDensity, nR, nZ, nPhi);
  spaceCharge->SetInputSpaceChargeA((TMatrixD**)mMatrixChargeA.get());
  spaceCharge->SetInputSpaceChargeC((TMatrixD**)mMatrixChargeC.get());
  float omegaTau = -0.32 * bSign;
  spaceCharge->SetOmegaTauT1T2(omegaTau, 1.f, 1.f);
  spaceCharge->SetCorrectionType(0);
  spaceCharge->SetIntegrationStrategy(0);
  spaceCharge->ForceInitSpaceCharge3DPoissonIntegralDz(nR, nZ, nPhi, 300, 1e-8);
  return spaceCharge;
}

/// Store distortions from spaceChargeCalcFluc in the matrices provided. If providing an object with average space-charge distortions spaceChargeCalcAvg, the residual distortions (dist_fluctuation(xyzTrue) + corr_average(xyzDistorted)) will be stored.
/// \param spaceChargeCalcFluc fluctuating distortions object
/// \param spaceChargeCalcAvg average distortions object
/// \param matrixDistDrA matrix to store radial distortions on the A side
/// \param matrixDistDrphiA matrix to store rphi distortions on the A side
/// \param matrixDistDzA matrix to store z distortions on the A side
/// \param matrixDistDrC matrix to store radial distortions on the C side
/// \param matrixDistDrphiC matrix to store rphi distortions on the C side
/// \param matrixDistDzC matrix to store z distortions on the C side
void fillDistortionLookupMatrices(AliTPCSpaceCharge3DCalc* spaceChargeCalcFluc, AliTPCSpaceCharge3DCalc* spaceChargeCalcAvg, TMatrixD** matrixDistDrA, TMatrixD** matrixDistDrphiA, TMatrixD** matrixDistDzA, TMatrixD** matrixDistDrC, TMatrixD** matrixDistDrphiC, TMatrixD** matrixDistDzC)
{
  const int nR = spaceChargeCalcFluc->GetNRRows();
  const int nPhi = spaceChargeCalcFluc->GetNPhiSlices();
  const int nZ = spaceChargeCalcFluc->GetNZColumns();
  const float mVoxelSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nR - 1);
  const float mVoxelSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (nZ - 1);
  const float mVoxelSizePhi = o2::constants::math::TwoPI / nPhi;

  for (int iphi = 0; iphi < nPhi; ++iphi) {
    float phi = iphi * mVoxelSizePhi;

    for (int ir = 0; ir < nR; ++ir) {
      float r = AliTPCPoissonSolver::fgkIFCRadius + ir * mVoxelSizeR;

      for (int iz = 0; iz < nZ; ++iz) {
        float absz = iz * mVoxelSizeZ;

        float xCylA[3] = {r, phi, absz};
        float xCylC[3] = {r, phi, -1 * absz};
        int roc = 0;
        float distFlucA[3] = {0.f, 0.f, 0.f};
        float distFlucC[3] = {0.f, 0.f, 0.f};

        // get fluctuating distortions
        spaceChargeCalcFluc->GetDistortionCylAC(xCylA, roc, distFlucA);
        spaceChargeCalcFluc->GetDistortionCylAC(xCylC, roc + 18, distFlucC);
        float distA[3] = {distFlucA[0], distFlucA[1], distFlucA[2]};
        float distC[3] = {distFlucC[0], distFlucC[1], distFlucC[2]};

        // get average corrections if provided and add them to the fluctuating distortions
        if (spaceChargeCalcAvg) {
          float xDistCylA[3] = {xCylA[0] + distFlucA[0], xCylA[1] + distFlucA[1] / xCylA[0], xCylA[2] + distFlucA[2]};
          float xDistCylC[3] = {xCylC[0] + distFlucC[0], xCylC[1] + distFlucC[1] / xCylC[0], xCylC[2] + distFlucC[2]};
          float corrAvgA[3] = {0.f, 0.f, 0.f};
          float corrAvgC[3] = {0.f, 0.f, 0.f};
          spaceChargeCalcAvg->GetCorrectionCylAC(xDistCylA, roc, corrAvgA);
          spaceChargeCalcAvg->GetCorrectionCylAC(xDistCylC, roc + 18, corrAvgC);
          distA[0] += corrAvgA[0];
          distA[1] += (corrAvgA[1] * xCylA[0] / xDistCylA[0]);
          distA[2] += corrAvgA[2];
          distC[0] += corrAvgC[0];
          distC[1] += (corrAvgC[1] * xCylC[0] / xDistCylC[0]);
          distC[2] += corrAvgC[2];
        }

        // store (residual) distortions in the matrices
        (*matrixDistDrA[iphi])(ir, iz) = distA[0];
        (*matrixDistDrphiA[iphi])(ir, iz) = distA[1];
        (*matrixDistDzA[iphi])(ir, iz) = distA[2];
        (*matrixDistDrC[iphi])(ir, iz) = distC[0];
        (*matrixDistDrphiC[iphi])(ir, iz) = distC[1];
        (*matrixDistDzC[iphi])(ir, iz) = -1 * distC[2];
      }
    }
  }
}

/// Calculate and stream residual distortions from spaceChargeRes and from calcFluc and calcAvg for comparison.
/// \param calcFluc AliTPCSpaceCharge3DCalc object with fluctuation distortions
/// \param calcAvg AliTPCSpaceCharge3DCalc object with average distortions
/// \param spaceChargeRes SpaceCharge object with residual distortions
void makeDebugTreeResiduals(AliTPCSpaceCharge3DCalc* calcFluc, AliTPCSpaceCharge3DCalc* calcAvg, SpaceCharge* spaceChargeRes)
{
  const int nR = calcFluc->GetNRRows();
  const int nPhi = calcFluc->GetNPhiSlices();
  const int nZ = calcFluc->GetNZColumns();
  const float mVoxelSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nR - 1);
  const float mVoxelSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (nZ - 1);
  const float mVoxelSizePhi = o2::constants::math::TwoPI / nPhi;

  o2::utils::TreeStreamRedirector pcstream("debugResidualDistortions.root", "recreate");
  for (int iside = 0; iside < 2; ++iside) {
    int roc = iside == 0 ? 0 : 18;
    for (int iphi = 0; iphi < nPhi; ++iphi) {
      float phi = iphi * mVoxelSizePhi;
      for (int ir = 0; ir < nR; ++ir) {
        float r = AliTPCPoissonSolver::fgkIFCRadius + ir * mVoxelSizeR;
        float x = r * std::cos(phi);
        float y = r * std::sin(phi);
        for (int iz = 1; iz < nZ; ++iz) {
          float z = iside == 0 ? iz * mVoxelSizeZ : -1 * iz * mVoxelSizeZ;

          GlobalPosition3D posDistRes(x, y, z);
          spaceChargeRes->distortElectron(posDistRes);
          float xyzDistRes[3] = {posDistRes.x(), posDistRes.y(), posDistRes.z()};
          float distRes[3] = {posDistRes.x() - x, posDistRes.y() - y, posDistRes.z() - z};

          float xyz[3] = {x, y, z};
          float distFluc[3] = {0.f, 0.f, 0.f};
          calcFluc->GetDistortion(xyz, roc, distFluc);
          float xyzDist[3] = {xyz[0] + distFluc[0], xyz[1] + distFluc[1], xyz[2] + distFluc[2]};
          float corrAvg[3] = {0.f, 0.f, 0.f};
          calcAvg->GetCorrection(xyzDist, roc, corrAvg);
          float xyzDistResTrue[3] = {xyzDist[0] + corrAvg[0], xyzDist[1] + corrAvg[1], xyzDist[2] + corrAvg[2]};
          float distResTrue[3] = {distFluc[0] + corrAvg[0], distFluc[1] + corrAvg[1], distFluc[2] + corrAvg[2]};

          pcstream << "debug"
                   << "iside=" << iside
                   << "iphi=" << iphi
                   << "ir=" << ir
                   << "iz=" << iz
                   // original position
                   << "phi=" << phi
                   << "r=" << r
                   << "x=" << x
                   << "y=" << y
                   << "z=" << z
                   // position of distorted points
                   << "xRes=" << xyzDistRes[0]
                   << "yRes=" << xyzDistRes[1]
                   << "zRes=" << xyzDistRes[2]
                   // true position of distorted points
                   << "xResTrue=" << xyzDistResTrue[0]
                   << "yResTrue=" << xyzDistResTrue[1]
                   << "zResTrue=" << xyzDistResTrue[2]
                   // residual distortions
                   << "distX=" << distRes[0]
                   << "distY=" << distRes[1]
                   << "distZ=" << distRes[2]
                   // true residual distortions
                   << "distXTrue=" << distResTrue[0]
                   << "distYTrue=" << distResTrue[1]
                   << "distZTrue=" << distResTrue[2]
                   //
                   << "\n";
        }
      }
    }
  }
  pcstream.Close();
}