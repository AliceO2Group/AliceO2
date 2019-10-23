// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file SpaceCharge.cxx
/// \brief Implementation of the interface for the ALICE TPC space-charge distortions calculations
/// \author Ernst Hellbär, Goethe-Universität Frankfurt, ernst.hellbar@cern.ch

#include "TGeoGlobalMagField.h"
#include "TH3.h"
#include "TMath.h"
#include "TMatrixD.h"
#include "TStopwatch.h"

#include "FairLogger.h"

#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTPC/Defs.h"
#include "Field/MagneticField.h"
#include "MathUtils/Utils.h"
#include "TPCBase/ParameterGas.h"
#include "TPCSimulation/SpaceCharge.h"

using namespace o2::tpc;

const float o2::tpc::SpaceCharge::sEzField = (AliTPCPoissonSolver::fgkCathodeV - AliTPCPoissonSolver::fgkGG) / AliTPCPoissonSolver::fgkTPCZ0;

SpaceCharge::SpaceCharge()
  : mNZ(MaxNZ),
    mNPhi(MaxNPhi),
    mNR(Constants::MAXGLOBALPADROW),
    mVoxelSizeZ(AliTPCPoissonSolver::fgkTPCZ0 / (MaxNZ - 1)),
    mDriftTimeVoxel(IonDriftTime / (MaxNZ - 1)),
    mVoxelSizePhi(TWOPI / MaxNPhi),
    mVoxelSizeR((AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (Constants::MAXGLOBALPADROW - 1)),
    mCoordZ(MaxNZ),
    mCoordPhi(MaxNPhi),
    mCoordR(Constants::MAXGLOBALPADROW),
    mInterpolationOrder(2),
    mUseInitialSCDensity(false),
    mInitLookUpTables(false),
    mMemoryAllocated(false),
    mTimeInit(-1),
    mSCDistortionType(SpaceCharge::SCDistortionType::SCDistortionsRealistic),
    mLookUpTableCalculator(Constants::MAXGLOBALPADROW, MaxNZ, MaxNPhi, 2, 3, 0),
    mSpaceChargeDensityA(MaxNPhi * Constants::MAXGLOBALPADROW * MaxNZ),
    mSpaceChargeDensityC(MaxNPhi * Constants::MAXGLOBALPADROW * MaxNZ),
    mRandomFlat(RandomRing<>::RandomType::Flat)
{
  mLookUpTableCalculator.SetCorrectionType(0);
  mLookUpTableCalculator.SetIntegrationStrategy(0);
  setVoxelCoordinates();
}

SpaceCharge::SpaceCharge(int nRBins, int nPhiBins, int nZSlices)
  : mNZ(nZSlices),
    mNPhi(nPhiBins),
    mNR(nRBins),
    mVoxelSizeZ(AliTPCPoissonSolver::fgkTPCZ0 / (nZSlices - 1)),
    mDriftTimeVoxel(IonDriftTime / (nZSlices - 1)),
    mVoxelSizePhi(TWOPI / nPhiBins),
    mVoxelSizeR((AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nRBins - 1)),
    mCoordZ(nZSlices),
    mCoordPhi(nPhiBins),
    mCoordR(nRBins),
    mInterpolationOrder(2),
    mUseInitialSCDensity(false),
    mInitLookUpTables(false),
    mMemoryAllocated(false),
    mTimeInit(-1),
    mSCDistortionType(SpaceCharge::SCDistortionType::SCDistortionsRealistic),
    mLookUpTableCalculator(nRBins, nZSlices, nPhiBins, 2, 3, 0),
    mSpaceChargeDensityA(nPhiBins * nRBins * nZSlices),
    mSpaceChargeDensityC(nPhiBins * nRBins * nZSlices),
    mRandomFlat(RandomRing<>::RandomType::Flat)
{
  mLookUpTableCalculator.SetCorrectionType(0);
  mLookUpTableCalculator.SetIntegrationStrategy(0);
  setVoxelCoordinates();
}

SpaceCharge::SpaceCharge(int nRBins, int nPhiBins, int nZSlices, int interpolationOrder)
  : mNZ(nZSlices),
    mNPhi(nPhiBins),
    mNR(nRBins),
    mVoxelSizeZ(AliTPCPoissonSolver::fgkTPCZ0 / (nZSlices - 1)),
    mDriftTimeVoxel(IonDriftTime / (nZSlices - 1)),
    mVoxelSizePhi(TWOPI / nPhiBins),
    mVoxelSizeR((AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nRBins - 1)),
    mCoordZ(nZSlices),
    mCoordPhi(nPhiBins),
    mCoordR(nRBins),
    mInterpolationOrder(interpolationOrder),
    mUseInitialSCDensity(false),
    mInitLookUpTables(false),
    mMemoryAllocated(false),
    mTimeInit(-1),
    mSCDistortionType(SpaceCharge::SCDistortionType::SCDistortionsRealistic),
    mLookUpTableCalculator(nRBins, nZSlices, nPhiBins, interpolationOrder, 3, 0),
    mSpaceChargeDensityA(nPhiBins * nRBins * nZSlices),
    mSpaceChargeDensityC(nPhiBins * nRBins * nZSlices),
    mRandomFlat(RandomRing<>::RandomType::Flat)
{
  mLookUpTableCalculator.SetCorrectionType(0);
  mLookUpTableCalculator.SetIntegrationStrategy(0);
  setVoxelCoordinates();
}

void SpaceCharge::setVoxelCoordinates()
{
  for (int iz = 0; iz < mNZ; ++iz) {
    mCoordZ[iz] = iz * mVoxelSizeZ;
  }
  for (int iphi = 0; iphi < mNPhi; ++iphi) {
    mCoordPhi[iphi] = iphi * mVoxelSizePhi;
  }
  for (int ir = 0; ir < mNR; ++ir) {
    mCoordR[ir] = AliTPCPoissonSolver::fgkIFCRadius + ir * mVoxelSizeR;
  }
}

void SpaceCharge::allocateMemory()
{
  mMatrixIonDriftZA = std::make_unique<std::unique_ptr<TMatrixD>[]>(mNPhi);
  mMatrixIonDriftRPhiA = std::make_unique<std::unique_ptr<TMatrixD>[]>(mNPhi);
  mMatrixIonDriftRA = std::make_unique<std::unique_ptr<TMatrixD>[]>(mNPhi);
  mMatrixIonDriftZC = std::make_unique<std::unique_ptr<TMatrixD>[]>(mNPhi);
  mMatrixIonDriftRPhiC = std::make_unique<std::unique_ptr<TMatrixD>[]>(mNPhi);
  mMatrixIonDriftRC = std::make_unique<std::unique_ptr<TMatrixD>[]>(mNPhi);
  for (int iphi = 0; iphi < mNPhi; ++iphi) {
    mMatrixIonDriftZA[iphi] = std::make_unique<TMatrixD>(mNR, mNZ);
    mMatrixIonDriftRPhiA[iphi] = std::make_unique<TMatrixD>(mNR, mNZ);
    mMatrixIonDriftRA[iphi] = std::make_unique<TMatrixD>(mNR, mNZ);
    mMatrixIonDriftZC[iphi] = std::make_unique<TMatrixD>(mNR, mNZ);
    mMatrixIonDriftRPhiC[iphi] = std::make_unique<TMatrixD>(mNR, mNZ);
    mMatrixIonDriftRC[iphi] = std::make_unique<TMatrixD>(mNR, mNZ);
  }
  mLookUpIonDriftA = std::make_unique<AliTPCLookUpTable3DInterpolatorD>(mNR, (TMatrixD**)mMatrixIonDriftRA.get(), mCoordR.data(), mNPhi, (TMatrixD**)mMatrixIonDriftRPhiA.get(), mCoordPhi.data(), mNZ, (TMatrixD**)mMatrixIonDriftZA.get(), mCoordZ.data(), mInterpolationOrder);
  mLookUpIonDriftC = std::make_unique<AliTPCLookUpTable3DInterpolatorD>(mNR, (TMatrixD**)mMatrixIonDriftRC.get(), mCoordR.data(), mNPhi, (TMatrixD**)mMatrixIonDriftRPhiC.get(), mCoordPhi.data(), mNZ, (TMatrixD**)mMatrixIonDriftZC.get(), mCoordZ.data(), mInterpolationOrder);
  mMemoryAllocated = true;
}

void SpaceCharge::init()
{
  if (!mInitLookUpTables) {
    auto o2field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
    const float bzField = o2field->solenoidField(); // magnetic field in kGauss
    /// TODO is there a faster way to get the drift velocity
    auto& gasParam = ParameterGas::Instance();
    float vDrift = gasParam.DriftV; // drift velocity in cm/us
    /// TODO fix hard coded values (ezField, t1, t2): export to Constants.h or get from somewhere?
    const float t1 = 1.;
    const float t2 = 1.;
    /// TODO use this parameterization or fixed value(s) from Magboltz calculations?
    const float omegaTau = -10. * bzField * vDrift / std::abs(sEzField);
    setOmegaTauT1T2(omegaTau, t1, t2);
    if (mUseInitialSCDensity) {
      calculateLookupTables();
    }
  }
}

float SpaceCharge::calculateLookupTables()
{
  // Potential, E field and electron distortion and correction lookup tables
  TStopwatch timer;
  mLookUpTableCalculator.ForceInitSpaceCharge3DPoissonIntegralDz(mNR, mNZ, mNPhi, 300, 1e-8);
  float tRealCalc = static_cast<float>(timer.RealTime());

  // Lookup tables for local ion drift along E field
  if (mSCDistortionType == SCDistortionType::SCDistortionsRealistic) {
    if (!mMemoryAllocated) {
      allocateMemory();
    }
    TMatrixD* matrixDriftZ = nullptr;
    TMatrixD* matrixDriftRPhi = nullptr;
    TMatrixD* matrixDriftR = nullptr;
    for (int iside = 0; iside < 2; ++iside) {
      const int sign = 1 - iside * 2;
      for (int iphi = 0; iphi < mNPhi; ++iphi) {
        const float phi = static_cast<float>(mCoordPhi[iphi]);
        if (iside == 0) {
          matrixDriftZ = mMatrixIonDriftZA[iphi].get();
          matrixDriftRPhi = mMatrixIonDriftRPhiA[iphi].get();
          matrixDriftR = mMatrixIonDriftRA[iphi].get();
        } else {
          matrixDriftZ = mMatrixIonDriftZC[iphi].get();
          matrixDriftRPhi = mMatrixIonDriftRPhiC[iphi].get();
          matrixDriftR = mMatrixIonDriftRC[iphi].get();
        }
        int roc = iside == 0 ? o2::utils::Angle2Sector(phi) : o2::utils::Angle2Sector(phi) + 18;
        for (int ir = 0; ir < mNR; ++ir) {
          const float radius = static_cast<float>(mCoordR[ir]);
          /// TODO: what is the electric field stored in the LUTs at iz=0 and iz=mNZSlices-1
          for (int iz = 0; iz < mNZ; ++iz) {
            const float z = static_cast<float>(mCoordZ[iz]);
            float x0[3] = {radius, phi, sign * (z + static_cast<float>(mVoxelSizeZ))}; // iphi, ir, iz+1
            float x1[3] = {radius, phi, sign * z};                                     // iphi, ir, iz
            if (iside == 1) {
              x0[2] *= -1;
              x1[2] *= -1;
            }
            double eVector0[3] = {0., 0., 0.};
            double eVector1[3] = {0., 0., 0.};
            mLookUpTableCalculator.GetElectricFieldCyl(x0, roc, eVector0); // returns correct sign for Ez
            mLookUpTableCalculator.GetElectricFieldCyl(x1, roc, eVector1); // returns correct sign for Ez

            // drift of ions along E field
            (*matrixDriftR)(ir, iz) = -1 * sign * mVoxelSizeZ * 0.5 * (eVector0[0] + eVector1[0]) / (sign * sEzField + eVector0[2]);
            (*matrixDriftRPhi)(ir, iz) = -1 * sign * mVoxelSizeZ * 0.5 * (eVector0[1] + eVector1[1]) / (sign * sEzField + eVector0[2]);
            (*matrixDriftZ)(ir, iz) = -1 * sign * mVoxelSizeZ + DvDEoverv0 * mVoxelSizeZ * 0.5 * (eVector0[2] + eVector1[2]);
          }
        }
      }
      if (iside == 0) {
        mLookUpIonDriftA->CopyFromMatricesToInterpolator();
      } else {
        mLookUpIonDriftC->CopyFromMatricesToInterpolator();
      }
    }

    // TODO: Propagate current SC density along E field by one time bin for next update
    // propagateSpaceCharge();
  }

  mInitLookUpTables = true;
  return tRealCalc;
}

float SpaceCharge::updateLookupTables(float eventTime)
{
  // TODO: only update after update time interval
  // if (mTimeInit < 0.) {
  //   mTimeInit = eventTime; // set the time of first initialization
  // }
  // if (std::abs(eventTime - mTimeInit) < mDriftTimeVoxel) {
  //   return 0.f; // update only after one time bin has passed
  // }
  // mTimeInit = eventTime;

  std::unique_ptr<std::unique_ptr<TMatrixD>[]> spaceChargeA = std::make_unique<std::unique_ptr<TMatrixD>[]>(mNPhi);
  std::unique_ptr<std::unique_ptr<TMatrixD>[]> spaceChargeC = std::make_unique<std::unique_ptr<TMatrixD>[]>(mNPhi);
  for (int iphi = 0; iphi < mNPhi; ++iphi) {
    spaceChargeA[iphi] = std::make_unique<TMatrixD>(mNR, mNZ);
    spaceChargeC[iphi] = std::make_unique<TMatrixD>(mNR, mNZ);
  }
  for (int iside = 0; iside < 2; ++iside) {
    for (int iphi = 0; iphi < mNPhi; ++iphi) {
      TMatrixD& chargeDensity = iside == 0 ? *spaceChargeA[iphi] : *spaceChargeC[iphi];
      for (int ir = 0; ir < mNR; ++ir) {
        for (int iz = 0; iz < mNZ; ++iz) {
          if (iside == 0) {
            chargeDensity(ir, iz) = mSpaceChargeDensityA[iphi * mNR * mNZ + ir * mNZ + iz];
          } else {
            chargeDensity(ir, iz) = mSpaceChargeDensityC[iphi * mNR * mNZ + ir * mNZ + iz];
          }
        }
      }
    }
  }
  mLookUpTableCalculator.SetInputSpaceChargeA((TMatrixD**)spaceChargeA.get());
  mLookUpTableCalculator.SetInputSpaceChargeC((TMatrixD**)spaceChargeC.get());
  float tRealCalc = calculateLookupTables();
  return tRealCalc;
}

void SpaceCharge::setOmegaTauT1T2(float omegaTau, float t1, float t2)
{
  mLookUpTableCalculator.SetOmegaTauT1T2(omegaTau, t1, t2);
}

void SpaceCharge::setInitialSpaceChargeDensity(const TH3* hisSCDensity)
{
  std::unique_ptr<std::unique_ptr<TMatrixD>[]> spaceChargeA = std::make_unique<std::unique_ptr<TMatrixD>[]>(mNPhi);
  std::unique_ptr<std::unique_ptr<TMatrixD>[]> spaceChargeC = std::make_unique<std::unique_ptr<TMatrixD>[]>(mNPhi);
  for (int iphi = 0; iphi < mNPhi; ++iphi) {
    spaceChargeA[iphi] = std::make_unique<TMatrixD>(mNR, mNZ);
    spaceChargeC[iphi] = std::make_unique<TMatrixD>(mNR, mNZ);
  }
  mLookUpTableCalculator.GetChargeDensity((TMatrixD**)spaceChargeA.get(), (TMatrixD**)spaceChargeC.get(), hisSCDensity, mNR, mNZ, mNPhi);
  for (int iside = 0; iside < 2; ++iside) {
    for (int iphi = 0; iphi < mNPhi; ++iphi) {
      TMatrixD& chargeDensity = iside == 0 ? *spaceChargeA[iphi] : *spaceChargeC[iphi];
      for (int ir = 0; ir < mNR; ++ir) {
        for (int iz = 0; iz < mNZ; ++iz) {
          if (iside == 0) {
            mSpaceChargeDensityA[iphi * mNR * mNZ + ir * mNZ + iz] = chargeDensity(ir, iz);
          } else {
            mSpaceChargeDensityC[iphi * mNR * mNZ + ir * mNZ + iz] = chargeDensity(ir, iz);
          }
        }
      }
    }
  }
  mLookUpTableCalculator.SetInputSpaceChargeA((TMatrixD**)spaceChargeA.get());
  mLookUpTableCalculator.SetInputSpaceChargeC((TMatrixD**)spaceChargeC.get());
  mUseInitialSCDensity = true;
}

void SpaceCharge::fillPrimaryIons(double r, double phi, double z, int nIons)
{
  Side side = z > 0 ? Side::A : Side::C;
  double dr = 0.;
  double drphi = 0.;
  double dz = 0.;
  getIonDrift(side, r, phi, z, dr, drphi, dz);
  double rdist = r + dr;
  double phidist = phi + drphi / r;
  double zdist = z + dz;
  const int zBin = TMath::BinarySearch(mNZ, mCoordZ.data(), std::abs(zdist));
  const int phiBin = TMath::BinarySearch(mNPhi, mCoordPhi.data(), phidist);
  const int rBin = TMath::BinarySearch(mNR, mCoordR.data(), rdist);
  /// TODO: protection against ions ending up outside the volume
  if (z > 0 && zdist > 0) {
    mSpaceChargeDensityA[phiBin * mNR * mNZ + rBin * mNZ + zBin] += ions2Charge(rBin, nIons);
  } else if (z < 0 && zdist < 0) {
    mSpaceChargeDensityC[phiBin * mNR * mNZ + rBin * mNZ + zBin] += ions2Charge(rBin, nIons);
  }
}

void SpaceCharge::fillPrimaryCharge(double r, double phi, double z, float charge)
{
  Side side = z > 0 ? Side::A : Side::C;
  double dr = 0.;
  double drphi = 0.;
  double dz = 0.;
  getIonDrift(side, r, phi, z, dr, drphi, dz);
  double rdist = r + dr;
  double phidist = phi + drphi / r;
  double zdist = z + dz;
  const int zBin = TMath::BinarySearch(mNZ, mCoordZ.data(), std::abs(zdist));
  const int phiBin = TMath::BinarySearch(mNPhi, mCoordPhi.data(), phidist);
  const int rBin = TMath::BinarySearch(mNR, mCoordR.data(), rdist);
  /// TODO: protection against ions ending up outside the volume
  if (z > 0 && zdist > 0) {
    mSpaceChargeDensityA[phiBin * mNR * mNZ + rBin * mNZ + zBin] += charge;
  } else if (z < 0 && zdist < 0) {
    mSpaceChargeDensityC[phiBin * mNR * mNZ + rBin * mNZ + zBin] += charge;
  }
}

void SpaceCharge::fillIBFIons(double r, double phi, Side side, int nIons)
{
  const int phiBin = TMath::BinarySearch(mNPhi, mCoordPhi.data(), phi);
  const int rBin = TMath::BinarySearch(mNR, mCoordR.data(), r);
  const int zBin = mNZ - 1;
  /// TODO: distribution of amplification ions instead of placing all of them in one point
  /// TODO: protection against ions ending up outside the volume
  if (side == Side::A) {
    mSpaceChargeDensityA[phiBin * mNR * mNZ + rBin * mNZ + zBin] += ions2Charge(rBin, nIons);
  } else {
    mSpaceChargeDensityC[phiBin * mNR * mNZ + rBin * mNZ + zBin] += ions2Charge(rBin, nIons);
  }
}

void SpaceCharge::fillIBFCharge(double r, double phi, Side side, float charge)
{
  const int phiBin = TMath::BinarySearch(mNPhi, mCoordPhi.data(), phi);
  const int rBin = TMath::BinarySearch(mNR, mCoordR.data(), r);
  const int zBin = mNZ - 1;
  /// TODO: distribution of amplification ions instead of placing all of them in one point
  /// TODO: protection against ions ending up outside the volume
  if (side == Side::A) {
    mSpaceChargeDensityA[phiBin * mNR * mNZ + rBin * mNZ + zBin] += charge;
  } else {
    mSpaceChargeDensityC[phiBin * mNR * mNZ + rBin * mNZ + zBin] += charge;
  }
}

void SpaceCharge::propagateSpaceCharge()
{
  //
  // Continuity  equation, assuming that the charge is conserved:
  //   delta(rho_0) / delta(t) =  - div(flux) = - div(rho_0 * u) = - ( rho_0 * div(u) + u * grad(rho_0) )
  //       u: velocity vector of space-charge density
  //       rho_0: space-charge density at time t0
  //
  //
  // Calculate the change of the space-charge density rho_0 at time t0 after time delta(t) = mVoxelSizeZ / v_driftIon:
  //   delta(rho_0) = - ( rho_0 * div(u) + u * grad(rho_0) ) * delta(t)
  //                = - ( rho_0 * div(d) + d * grad(rho_0) )
  //       d: drift vector (dr, dphi, mVoxelSizeZ + dz)
  //
  //   div(d) = 1/r del(r*d_r)/del(r) + 1/r del(d_phi)/del(phi) + del(d_z)/del(z)
  //          = d_r/r + del(d_r)/del(r) + 1/r del(d_phi)/del(phi) + del(d_z)/del(z)
  //
  //   grad(rho_0) = del(rho_0)/del(r) * e_r + 1/r * del(rho_0)/del(phi) * e_phi + del(rho_0)/del(z) * e_z
  //       e_i: unit vectors in i = {r, phi, z} direction
  //

  // Finite difference interpolation coefficients
  const float coeffCent40 = 1.f / 12.f;
  const float coeffCent41 = -2.f / 3.f;
  const float coeffCent42 = -1.f * coeffCent41;
  const float coeffCent43 = -1.f * coeffCent40;
  const float coeffFwd30 = -11.f / 6.f;
  const float coeffFwd31 = 3.f;
  const float coeffFwd32 = -1.5;
  const float coeffFwd33 = 1.f / 3.f;
  const float coeffFwd20 = -1.5;
  const float coeffFwd21 = 2.f;
  const float coeffFwd22 = -0.5;

  std::vector<float>* scDensity = nullptr;
  TMatrixD** matrixDriftZ = nullptr;
  TMatrixD** matrixDriftRPhi = nullptr;
  TMatrixD** matrixDriftR = nullptr;
  for (int iside = 0; iside < 2; ++iside) {
    const int signZ = iside == 0 ? 1 : -1;
    if (iside == 0) {
      scDensity = &mSpaceChargeDensityA;
      matrixDriftZ = (TMatrixD**)mMatrixIonDriftZA.get();
      matrixDriftRPhi = (TMatrixD**)mMatrixIonDriftRPhiA.get();
      matrixDriftR = (TMatrixD**)mMatrixIonDriftRA.get();
    } else {
      scDensity = &mSpaceChargeDensityC;
      matrixDriftZ = (TMatrixD**)mMatrixIonDriftZC.get();
      matrixDriftRPhi = (TMatrixD**)mMatrixIonDriftRPhiC.get();
      matrixDriftR = (TMatrixD**)mMatrixIonDriftRC.get();
    }
    /// TODO: is there a better way than to create a copy for the new SC density?
    std::vector<float> newSCDensity(mNPhi * mNR * mNZ);
    for (int iphi0 = 0; iphi0 < mNPhi; ++iphi0) {

      for (int ir0 = 0; ir0 < mNR; ++ir0) {
        const float r0 = static_cast<float>(mCoordR[ir0]);

        for (int iz0 = 0; iz0 < mNZ; ++iz0) {

          // rho_0 * div(d)
          float ddrdr = 0.f;
          if (ir0 > 1 && ir0 < mNR - 2) {
            ddrdr = (coeffCent40 * (*matrixDriftR[iphi0])(ir0 - 2, iz0) + coeffCent41 * (*matrixDriftR[iphi0])(ir0 - 1, iz0) + coeffCent42 * (*matrixDriftR[iphi0])(ir0 + 1, iz0) + coeffCent43 * (*matrixDriftR[iphi0])(ir0 + 2, iz0)) / static_cast<float>(mVoxelSizeR);
          } else if (ir0 < 2) {
            ddrdr = (coeffFwd30 * (*matrixDriftR[iphi0])(ir0, iz0) + coeffFwd31 * (*matrixDriftR[iphi0])(ir0 + 1, iz0) + coeffFwd32 * (*matrixDriftR[iphi0])(ir0 + 2, iz0) + coeffFwd33 * (*matrixDriftR[iphi0])(ir0 + 3, iz0)) / static_cast<float>(mVoxelSizeR);
          } else if (ir0 > (mNR - 3)) {
            ddrdr = -1 * (coeffFwd30 * (*matrixDriftR[iphi0])(ir0, iz0) + coeffFwd31 * (*matrixDriftR[iphi0])(ir0 - 1, iz0) + coeffFwd32 * (*matrixDriftR[iphi0])(ir0 - 2, iz0) + coeffFwd33 * (*matrixDriftR[iphi0])(ir0 - 3, iz0)) / static_cast<float>(mVoxelSizeR);
          }

          const int iphiCent0 = iphi0 - 2 + (mNPhi) * (iphi0 < 2);
          const int iphiCent1 = iphi0 - 1 + (mNPhi) * (iphi0 < 1);
          const int iphiCent3 = iphi0 + 1 - (mNPhi) * (iphi0 > (mNPhi - 2));
          const int iphiCent4 = iphi0 + 2 - (mNPhi) * (iphi0 > (mNPhi - 3));
          const float ddphidphi = (coeffCent40 * (*matrixDriftRPhi[iphiCent0])(ir0, iz0) + coeffCent41 * (*matrixDriftRPhi[iphiCent1])(ir0, iz0) + coeffCent42 * (*matrixDriftRPhi[iphiCent3])(ir0, iz0) + coeffCent43 * (*matrixDriftRPhi[iphiCent4])(ir0, iz0)) / static_cast<float>(mVoxelSizePhi);

          float ddzdz = 0.f;
          if (iz0 > 1 && iz0 < mNZ - 2) {
            ddzdz = signZ * (coeffCent40 * (*matrixDriftZ[iphi0])(ir0, iz0 - 2) + coeffCent41 * (*matrixDriftZ[iphi0])(ir0, iz0 - 1) + coeffCent42 * (*matrixDriftZ[iphi0])(ir0, iz0 + 1) + coeffCent43 * (*matrixDriftZ[iphi0])(ir0, iz0 + 2)) / static_cast<float>(mVoxelSizeR);
          } else if (iz0 == 1 || iz0 == (mNZ - 2)) {
            ddzdz = signZ * (-0.5 * (*matrixDriftZ[iphi0])(ir0, iz0 - 1) + 0.5 * (*matrixDriftZ[iphi0])(ir0, iz0 + 1)) / static_cast<float>(mVoxelSizeR);
          } else if (iz0 == 0) {
            ddzdz = signZ * (coeffFwd20 * (*matrixDriftZ[iphi0])(ir0, iz0) + coeffFwd21 * (*matrixDriftZ[iphi0])(ir0, iz0 + 1) + coeffFwd22 * (*matrixDriftZ[iphi0])(ir0, iz0 + 2)) / static_cast<float>(mVoxelSizeR);
          } else if (iz0 == (mNZ - 1)) {
            ddzdz = -1 * signZ * (coeffFwd20 * (*matrixDriftZ[iphi0])(ir0, iz0) + coeffFwd21 * (*matrixDriftZ[iphi0])(ir0, iz0 - 1) + coeffFwd22 * (*matrixDriftZ[iphi0])(ir0, iz0 - 2)) / static_cast<float>(mVoxelSizeR);
          }

          const float qdivd = (*scDensity)[iphi0 * mNR * mNZ + ir0 * mNZ + iz0] * (((*matrixDriftR[iphi0])(ir0, iz0) + ddphidphi) / r0 + ddrdr + ddzdz);

          // - d * grad(rho_0) = - d_drift * grad(rho_0(x)) + d_dist * grad(rho_0(x+d_drift)) = (charge0 - charge1) + d_dist * grad(rho_0(x-d_drift))
          if (iz0 < (mNZ - 1)) {
            const float dr = (*matrixDriftR[iphi0])(ir0, iz0);
            const float drphi = (*matrixDriftRPhi[iphi0])(ir0, iz0);
            const float dz = (*matrixDriftZ[iphi0])(ir0, iz0) + mVoxelSizeZ * signZ;

            const int ir1 = dr < 0 ? ir0 - 1 * (ir0 == (mNR - 1)) : ir0 - 1 + 1 * (ir0 == 0);
            const int ir2 = dr < 0 ? ir0 + 1 - 1 * (ir0 == (mNR - 1)) : ir0 + 1 * (ir0 == 0);
            const int iphi1 = drphi < 0 ? iphi0 : iphi0 - 1 + (mNPhi) * (iphi0 == 0);
            const int iphi2 = drphi < 0 ? iphi0 + 1 - (mNPhi) * (iphi0 == (mNPhi - 1)) : iphi0;
            const int iz1 = dz < 0 ? iz0 + 1 - 1 * (iside == 0) * (iz0 == (mNZ - 2)) : iz0 + 2 * (iside == 1) - 1 * (iside == 1) * (iz0 == (mNZ - 2));
            const int iz2 = dz < 0 ? iz0 + 2 - 2 * (iside == 1) - 1 * (iside == 0) * (iz0 == (mNZ - 2)) : iz0 + 1 - 1 * (iside == 1) * (iz0 == (mNZ - 2));

            const float dqdr = ((*scDensity)[iphi0 * mNR * mNZ + ir2 * mNZ + (iz0 + 1)] - (*scDensity)[iphi0 * mNR * mNZ + ir1 * mNZ + (iz0 + 1)]) / static_cast<float>(mVoxelSizeR);
            const float dqdphi = ((*scDensity)[iphi2 * mNR * mNZ + ir0 * mNZ + (iz0 + 1)] - (*scDensity)[iphi1 * mNR * mNZ + ir0 * mNZ + (iz0 + 1)]) / static_cast<float>(mVoxelSizePhi);
            const float dqdz = ((*scDensity)[iphi0 * mNR * mNZ + ir0 * mNZ + iz2] - (*scDensity)[iphi0 * mNR * mNZ + ir0 * mNZ + iz1]) / static_cast<float>(mVoxelSizeZ);

            const float dgradq = dr * dqdr + drphi / r0 * dqdphi + dz * dqdz;

            newSCDensity[iphi0 * mNR * mNZ + ir0 * mNZ + iz0] = (*scDensity)[iphi0 * mNR * mNZ + ir0 * mNZ + (iz0 + 1)] - (qdivd + dgradq);
          } else {
            newSCDensity[iphi0 * mNR * mNZ + ir0 * mNZ + iz0] = -qdivd;
          }

          if (newSCDensity[iphi0 * mNR * mNZ + ir0 * mNZ + iz0] < 0.f) {
            newSCDensity[iphi0 * mNR * mNZ + ir0 * mNZ + iz0] = 0.f;
          }
        }
      }
    }
    (*scDensity).swap(newSCDensity);
  }
}

void SpaceCharge::propagateIons()
{
  std::vector<float>* scDensity = nullptr;
  AliTPCLookUpTable3DInterpolatorD* lookUpIonDrift = nullptr;
  for (int iside = 0; iside < 2; ++iside) {
    const int signZ = iside == 0 ? 1 : -1;
    if (iside == 0) {
      scDensity = &mSpaceChargeDensityA;
      lookUpIonDrift = mLookUpIonDriftA.get();
    } else {
      scDensity = &mSpaceChargeDensityC;
      lookUpIonDrift = mLookUpIonDriftC.get();
    }
    std::vector<float> newSCDensity(mNPhi * mNR * mNZ);

    for (int iphi0 = 0; iphi0 < mNPhi; ++iphi0) {
      const double phi0 = static_cast<float>(mCoordPhi[iphi0]);

      for (int ir0 = 0; ir0 < mNR; ++ir0) {
        const double r0 = static_cast<float>(mCoordR[ir0]);
        const double r1 = r0 + mVoxelSizeR;

        for (int iz0 = 0; iz0 < mNZ; ++iz0) {
          const double z0 = mCoordZ[iz0] * signZ;

          const float ionDensity = (*scDensity)[iphi0 * mNR * mNZ + ir0 * mNZ + iz0] * AliTPCPoissonSolver::fgke0 / TMath::Qe(); // #ions / cm^3
          const float nIons = std::round(ionDensity * mVoxelSizeZ * 0.5 * mVoxelSizePhi * (r1 * r1 - r0 * r0));                  // absolute #ions in the voxel

          for (int iion = 0; iion < nIons; ++iion) {
            double phiIon = phi0 + mRandomFlat.getNextValue() * mVoxelSizePhi;
            double rIon = r0 + mRandomFlat.getNextValue() * mVoxelSizeR;
            double zIon = z0 + mRandomFlat.getNextValue() * mVoxelSizeZ * signZ;

            double drphi = 0.f;
            double dr = 0.f;
            double dz = 0.f;
            lookUpIonDrift->GetValue(rIon, phiIon, std::abs(zIon), dr, drphi, dz);
            float phiIonF = static_cast<float>(phiIon + (drphi / rIon));
            o2::utils::BringTo02PiGen(phiIonF);
            rIon += dr;
            zIon += dz;

            // continue if ion is outside the TPC boundaries in r or z
            if (rIon > (mCoordR[mNR - 1] + mVoxelSizeR) || rIon < mCoordR[0]) {
              continue;
            }
            if ((zIon * signZ) > (mCoordZ[mNZ - 1] + mVoxelSizeZ) || (zIon * signZ) < mCoordZ[0]) {
              continue;
            }

            const int iphiDrift = TMath::BinarySearch(mNPhi, mCoordPhi.data(), static_cast<double>(phiIonF));
            const int irDrift = TMath::BinarySearch(mNR, mCoordR.data(), rIon);
            const int izDrift = TMath::BinarySearch(mNZ, mCoordZ.data(), std::abs(zIon));
            newSCDensity[iphiDrift * mNR * mNZ + irDrift * mNZ + izDrift] += ions2Charge(irDrift, 1);
          }
        }
      }
    }
    (*scDensity).swap(newSCDensity);
  }
}

void SpaceCharge::getIonDrift(Side side, double r, double phi, double z, double& dr, double& drphi, double& dz) const
{
  if (!mInitLookUpTables) {
    return;
  }
  if (side == Side::A) {
    mLookUpIonDriftA->GetValue(r, phi, z, dr, drphi, dz);
  } else if (side == Side::C) {
    mLookUpIonDriftC->GetValue(r, phi, -1 * z, dr, drphi, dz);
  } else {
    LOG(INFO) << "TPC side undefined! Cannot calculate local ion drift correction...";
  }
}

void SpaceCharge::correctElectron(GlobalPosition3D& point)
{
  if (!mInitLookUpTables) {
    return;
  }
  const float x[3] = {point.X(), point.Y(), point.Z()};
  float dx[3] = {0.f, 0.f, 0.f};
  float phi = point.phi();
  o2::utils::BringTo02PiGen(phi);
  int roc = o2::utils::Angle2Sector(phi);
  if (x[2] < 0) {
    roc += 18;
  }
  mLookUpTableCalculator.GetCorrection(x, roc, dx);
  point.SetXYZ(x[0] + dx[0], x[1] + dx[1], x[2] + dx[2]);
}

void SpaceCharge::distortElectron(GlobalPosition3D& point) const
{
  if (!mInitLookUpTables) {
    return;
  }
  const float x[3] = {point.X(), point.Y(), point.Z()};
  float dx[3] = {0.f, 0.f, 0.f};
  float phi = point.phi();
  o2::utils::BringTo02PiGen(phi);
  int roc = o2::utils::Angle2Sector(phi);
  if (x[2] < 0) {
    roc += 18;
  }
  mLookUpTableCalculator.GetDistortion(x, roc, dx);
  point.SetXYZ(x[0] + dx[0], x[1] + dx[1], x[2] + dx[2]);
}

double SpaceCharge::getChargeDensity(Side side, const GlobalPosition3D& point) const
{
  Float_t x[3] = {point.rho(), point.phi(), point.z()};
  o2::utils::BringTo02PiGen(x[1]);
  const int roc = side == Side::A ? o2::utils::Angle2Sector(x[1]) : o2::utils::Angle2Sector(x[1]) + 18;
  return mLookUpTableCalculator.GetChargeCylAC(x, roc);
}

float SpaceCharge::getChargeDensity(Side side, int ir, int iphi, int iz) const
{
  if (side == Side::A) {
    return mSpaceChargeDensityA[iphi * mNR * mNZ + ir * mNZ + iz];
  } else if (side == Side::C) {
    return mSpaceChargeDensityC[iphi * mNR * mNZ + ir * mNZ + iz];
  } else {
    return -1.f;
  }
}

void SpaceCharge::setUseIrregularLUTs(int useIrrLUTs)
{
  mLookUpTableCalculator.SetCorrectionType(useIrrLUTs);
}

void SpaceCharge::setUseFastDistIntegration(int useFastInt)
{
  mLookUpTableCalculator.SetIntegrationStrategy(useFastInt);
}

float SpaceCharge::ions2Charge(int rBin, int nIons)
{
  float rInner = mCoordR[rBin];
  float rOuter = mCoordR[rBin] + mVoxelSizeR;
  return nIons * TMath::Qe() / (mVoxelSizeZ * 0.5 * mVoxelSizePhi * (rOuter * rOuter - rInner * rInner)) / AliTPCPoissonSolver::fgke0;
}
