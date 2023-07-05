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

/// \file createResidualDistortionObject.C
/// \brief This macro creates a SpaceCharge object with residual distortions from a fluctuating and an average space-charge density histogram and stores it in a file.
/// \author Ernst Hellbär, Goethe-Universität Frankfurt, ernst.hellbar@cern.ch

#include <cmath>

#include "TFile.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "TPCSpaceCharge/SpaceCharge.h"
#include "MathUtils/Cartesian.h"

using namespace o2::tpc;
using DataT = double;
using DataContainer = DataContainer3D<DataT>;

// function declarations
void createSpaceCharge(o2::tpc::SpaceCharge<DataT>& spaceCharge, const char* path, const char* histoName, const int bSign, const int nThreads = -1);
void fillDistortionLookupMatrices(o2::tpc::SpaceCharge<DataT>& spaceChargeCalcFluc, o2::tpc::SpaceCharge<DataT>* spaceChargeCalcAvg, DataContainer matrixDistDr[2], DataContainer matrixDistDrphi[2], DataContainer matrixDistDz[2]);
void makeDebugTreeResiduals(o2::tpc::SpaceCharge<DataT>& calcFluc, o2::tpc::SpaceCharge<DataT>& calcAvg, o2::tpc::SpaceCharge<DataT>& spaceChargeRes);
/// Create a SpaceCharge object with residual distortions from a fluctuating and an average space-charge density histogram and write it to a file.
/// \param bSign sign of the B-field: -1 = negative, 0 = no B-field, 1 = positive
/// \param pathToHistoFile path to a file with the fluctuating and average histograms
/// \param histoFlucName name of the fluctuating histogram
/// \param histoAvgName name of the average histogram
void createResidualDistortionObject(int bSign, const char* pathToHistoFile = "InputSCDensityHistograms_8000events.root", const char* histoFlucName = "inputSCDensity3D_8000_0", const char* histoAvgName = "inputSCDensity3D_8000_avg", const int nThreads = -1, bool debug = false)
{
  /*
     Usage:
     root -l -b -q createResidualDistortionObject.C+\(-1,\"InputSCDensityHistograms_8000events.root\",\"inputSCDensity3D_8000_0\",\"inputSCDensity3D_8000_avg\",true\)
   */

  // Calculate fluctuating and average distortion and correction lookup tables
  o2::tpc::SpaceCharge<DataT> scCalcFluc;
  o2::tpc::SpaceCharge<DataT> scCalcAvg;
  createSpaceCharge(scCalcFluc, pathToHistoFile, histoFlucName, bSign, nThreads);
  createSpaceCharge(scCalcAvg, pathToHistoFile, histoAvgName, bSign, nThreads);

  // Create matrices and fill them with residual distortions. Create SpaceCharge object, assign residual distortion matrices to it and store it in the output file.
  DataContainer matrixResDistDr[2]{DataContainer(scCalcAvg.getNZVertices(), scCalcAvg.getNRVertices(), scCalcAvg.getNPhiVertices()), DataContainer(scCalcAvg.getNZVertices(), scCalcAvg.getNRVertices(), scCalcAvg.getNPhiVertices())};
  DataContainer matrixResDistDrphi[2]{DataContainer(scCalcAvg.getNZVertices(), scCalcAvg.getNRVertices(), scCalcAvg.getNPhiVertices()), DataContainer(scCalcAvg.getNZVertices(), scCalcAvg.getNRVertices(), scCalcAvg.getNPhiVertices())};
  DataContainer matrixResDistDz[2]{DataContainer(scCalcAvg.getNZVertices(), scCalcAvg.getNRVertices(), scCalcAvg.getNPhiVertices()), DataContainer(scCalcAvg.getNZVertices(), scCalcAvg.getNRVertices(), scCalcAvg.getNPhiVertices())};

  fillDistortionLookupMatrices(scCalcFluc, &scCalcAvg, matrixResDistDr, matrixResDistDrphi, matrixResDistDz);
  o2::tpc::SpaceCharge<DataT> spaceChargeRes;
  spaceChargeRes.setDistortionLookupTables(matrixResDistDz[0], matrixResDistDr[0], matrixResDistDrphi[0], o2::tpc::Side::A);
  spaceChargeRes.setDistortionLookupTables(matrixResDistDz[1], matrixResDistDr[1], matrixResDistDrphi[1], o2::tpc::Side::C);

  std::string_view file = "ResidualDistortions.root";
  spaceChargeRes.dumpGlobalDistortions(file, o2::tpc::Side::A, "RECREATE");
  spaceChargeRes.dumpGlobalDistortions(file, o2::tpc::Side::C, "UPDATE");

  if (debug) {
    makeDebugTreeResiduals(scCalcFluc, scCalcAvg, spaceChargeRes);
  }
}

/// calculate the distortions and corrections from a space-charge density histogram
/// \param spaceCharge input space-charge density object which will be filled
/// \param path path to a file with the histograms
/// \param histoName name of the histogram
/// \param bSign sign of the B-field: -1 = negative, 0 = no B-field, 1 = positive
/// \param nThreads number of threads which are used
void createSpaceCharge(o2::tpc::SpaceCharge<DataT>& spaceCharge, const char* path, const char* histoName, const int bSign, const int nThreads)
{
  const float omegaTau = -0.32 * bSign;
  spaceCharge.setOmegaTauT1T2(omegaTau, 1, 1);
  if (nThreads != -1) {
    spaceCharge.setNThreads(nThreads);
  }
  // set density from input file
  TFile fileOutSCDensity(path, "READ");
  spaceCharge.fillChargeDensityFromFile(fileOutSCDensity, histoName);
  fileOutSCDensity.Close();

  // calculate the distortions
  spaceCharge.calculateDistortionsCorrections(o2::tpc::Side::A);
  spaceCharge.calculateDistortionsCorrections(o2::tpc::Side::C);
}

/// Store distortions from spaceChargeCalcFluc in the matrices provided. If providing an object with average space-charge distortions spaceChargeCalcAvg, the residual distortions (dist_fluctuation(xyzTrue) + corr_average(xyzDistorted)) will be stored.
/// \param spaceChargeCalcFluc fluctuating distortions object
/// \param spaceChargeCalcAvg average distortions object
/// \param matrixDistDr matrix to store radial distortions on the A and C side
/// \param matrixDistDrphi matrix to store rphi distortions on the A and C side
/// \param matrixDistDz matrix to store z distortions on the A and C side
void fillDistortionLookupMatrices(o2::tpc::SpaceCharge<DataT>& spaceChargeCalcFluc, o2::tpc::SpaceCharge<DataT>* spaceChargeCalcAvg, DataContainer matrixDistDr[2], DataContainer matrixDistDrphi[2], DataContainer matrixDistDz[2])
{
  for (int iside = 0; iside < 2; ++iside) {
    o2::tpc::Side side = (iside == 0) ? o2::tpc::Side::A : o2::tpc::Side::C;
    for (int iphi = 0; iphi < spaceChargeCalcFluc.getNPhiVertices(); ++iphi) {
      const auto phi = spaceChargeCalcFluc.getPhiVertex(iphi, side);

      for (int ir = 0; ir < spaceChargeCalcFluc.getNRVertices(); ++ir) {
        const auto r = spaceChargeCalcFluc.getRVertex(ir, side);

        for (int iz = 0; iz < spaceChargeCalcFluc.getNZVertices(); ++iz) {
          const auto z = spaceChargeCalcFluc.getZVertex(iz, side);

          // get fluctuating distortions
          DataT distFlucdZ = 0;
          DataT distFlucdR = 0;
          DataT distFlucdRPhi = 0;
          spaceChargeCalcFluc.getDistortionsCyl(z, r, phi, side, distFlucdZ, distFlucdR, distFlucdRPhi);

          // get average corrections if provided and add them to the fluctuating distortions
          if (spaceChargeCalcAvg) {
            const float zDistorted = z + distFlucdZ;
            const float rDistorted = r + distFlucdR;
            const float phiDistorted = phi + distFlucdRPhi / r;
            DataT corrZDistPoint = 0;
            DataT corrRDistPoint = 0;
            DataT corrRPhiDistPoint = 0;
            spaceChargeCalcAvg->getCorrectionsCyl(zDistorted, rDistorted, phiDistorted, side, corrZDistPoint, corrRDistPoint, corrRPhiDistPoint);
            distFlucdZ += corrZDistPoint;
            distFlucdR += corrRDistPoint;
            distFlucdRPhi += corrRPhiDistPoint * r / rDistorted;
          }

          // store (residual) distortions in the matrices
          matrixDistDr[iside](ir, iz, iphi) = distFlucdR;
          matrixDistDrphi[iside](ir, iz, iphi) = distFlucdRPhi;
          matrixDistDz[iside](ir, iz, iphi) = distFlucdZ;
        }
      }
    }
  }
}

/// Calculate and stream residual distortions from spaceChargeRes and from calcFluc and calcAvg for comparison.
/// \param calcFluc space charge object with fluctuation distortions
/// \param calcAvg space charge object with average distortions
/// \param spaceChargeRes space charge object with residual distortions
void makeDebugTreeResiduals(o2::tpc::SpaceCharge<DataT>& calcFluc, o2::tpc::SpaceCharge<DataT>& calcAvg, o2::tpc::SpaceCharge<DataT>& spaceChargeRes)
{
  o2::utils::TreeStreamRedirector pcstream("debugResidualDistortions.root", "recreate");
  for (int iside = 0; iside < 2; ++iside) {
    o2::tpc::Side side = (iside == 0) ? o2::tpc::Side::A : o2::tpc::Side::C;
    for (int iphi = 0; iphi < calcFluc.getNPhiVertices(); ++iphi) {
      auto phi = calcFluc.getPhiVertex(iphi, side);
      for (int ir = 0; ir < calcFluc.getNRVertices(); ++ir) {
        auto r = calcFluc.getRVertex(ir, side);
        DataT x = r * std::cos(phi);
        DataT y = r * std::sin(phi);
        for (int iz = 1; iz < calcFluc.getNZVertices(); ++iz) {
          DataT z = calcFluc.getZVertex(iz, side);

          GlobalPosition3D posDistRes(x, y, z);
          spaceChargeRes.distortElectron(posDistRes);
          DataT xyzDistRes[3] = {posDistRes.x(), posDistRes.y(), posDistRes.z()};
          DataT distRes[3] = {posDistRes.x() - x, posDistRes.y() - y, posDistRes.z() - z};

          DataT distFlucX = 0;
          DataT distFlucY = 0;
          DataT distFlucZ = 0;
          calcFluc.getDistortions(x, y, z, side, distFlucX, distFlucY, distFlucZ);

          DataT xDistorted = x + distFlucX;
          DataT yDistorted = y + distFlucY;
          DataT zDistorted = z + distFlucZ;

          DataT corrAvgX = 0;
          DataT corrAvgY = 0;
          DataT corrAvgZ = 0;
          calcAvg.getCorrections(xDistorted, yDistorted, zDistorted, side, corrAvgX, corrAvgY, corrAvgZ);
          DataT xyzDistResTrue[3] = {xDistorted + corrAvgX, yDistorted + corrAvgY, zDistorted + corrAvgZ};
          DataT distResTrue[3] = {distFlucX + corrAvgX, distFlucY + corrAvgY, distFlucZ + corrAvgZ};

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
