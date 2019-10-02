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

#include "CommonConstants/MathConstants.h"
#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTPC/Defs.h"
#include "Field/MagneticField.h"
#include "MathUtils/Utils.h"
#include "TPCBase/ParameterGas.h"
#include "TPCSimulation/SpaceCharge.h"

using namespace o2::tpc;

const float o2::tpc::SpaceCharge::sEzField = (AliTPCPoissonSolver::fgkCathodeV - AliTPCPoissonSolver::fgkGG) / AliTPCPoissonSolver::fgkTPCZ0;

SpaceCharge::SpaceCharge()
  : mNZSlices(MaxZSlices),
    mNPhiBins(MaxPhiBins),
    mNRBins(Constants::MAXGLOBALPADROW),
    mLengthZSlice(AliTPCPoissonSolver::fgkTPCZ0 / (MaxZSlices - 1)),
    mDriftTimeZSlice(IonDriftTime / (MaxZSlices - 1)),
    mWidthPhiBin(TWOPI / MaxPhiBins),
    mLengthRBin((AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (Constants::MAXGLOBALPADROW - 1)),
    mCoordZ(MaxZSlices),
    mCoordPhi(MaxPhiBins),
    mCoordR(Constants::MAXGLOBALPADROW),
    mInterpolationOrder(2),
    mUseInitialSCDensity(false),
    mInitLookUpTables(false),
    mTimeInit(-1),
    mSCDistortionType(SpaceCharge::SCDistortionType::SCDistortionsRealistic),
    mLookUpTableCalculator(Constants::MAXGLOBALPADROW, MaxZSlices, MaxPhiBins, 2, 3, 0),
    mSpaceChargeDensityA(MaxPhiBins * Constants::MAXGLOBALPADROW * MaxZSlices),
    mSpaceChargeDensityC(MaxPhiBins * Constants::MAXGLOBALPADROW * MaxZSlices),
    mRandomFlat(RandomRing<>::RandomType::Flat)
{
  mLookUpTableCalculator.SetCorrectionType(0);
  mLookUpTableCalculator.SetIntegrationStrategy(0);
  allocateMemory();
}

SpaceCharge::SpaceCharge(int nZSlices, int nPhiBins, int nRBins)
  : mNZSlices(nZSlices),
    mNPhiBins(nPhiBins),
    mNRBins(nRBins),
    mLengthZSlice(AliTPCPoissonSolver::fgkTPCZ0 / (nZSlices - 1)),
    mDriftTimeZSlice(IonDriftTime / (nZSlices - 1)),
    mWidthPhiBin(TWOPI / nPhiBins),
    mLengthRBin((AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nRBins - 1)),
    mCoordZ(nZSlices),
    mCoordPhi(nPhiBins),
    mCoordR(nRBins),
    mInterpolationOrder(2),
    mUseInitialSCDensity(false),
    mInitLookUpTables(false),
    mTimeInit(-1),
    mSCDistortionType(SpaceCharge::SCDistortionType::SCDistortionsRealistic),
    mLookUpTableCalculator(nRBins, nZSlices, nPhiBins, 2, 3, 0),
    mSpaceChargeDensityA(nPhiBins * nRBins * nZSlices),
    mSpaceChargeDensityC(nPhiBins * nRBins * nZSlices),
    mRandomFlat(RandomRing<>::RandomType::Flat)
{
  mLookUpTableCalculator.SetCorrectionType(0);
  mLookUpTableCalculator.SetIntegrationStrategy(0);
  allocateMemory();
}

SpaceCharge::SpaceCharge(int nZSlices, int nPhiBins, int nRBins, int interpolationOrder)
  : mNZSlices(nZSlices),
    mNPhiBins(nPhiBins),
    mNRBins(nRBins),
    mLengthZSlice(AliTPCPoissonSolver::fgkTPCZ0 / (nZSlices - 1)),
    mDriftTimeZSlice(IonDriftTime / (nZSlices - 1)),
    mWidthPhiBin(TWOPI / nPhiBins),
    mLengthRBin((AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nRBins - 1)),
    mCoordZ(nZSlices),
    mCoordPhi(nPhiBins),
    mCoordR(nRBins),
    mInterpolationOrder(interpolationOrder),
    mUseInitialSCDensity(false),
    mInitLookUpTables(false),
    mTimeInit(-1),
    mSCDistortionType(SpaceCharge::SCDistortionType::SCDistortionsRealistic),
    mLookUpTableCalculator(nRBins, nZSlices, nPhiBins, interpolationOrder, 3, 0),
    mSpaceChargeDensityA(nPhiBins * nRBins * nZSlices),
    mSpaceChargeDensityC(nPhiBins * nRBins * nZSlices),
    mRandomFlat(RandomRing<>::RandomType::Flat)
{
  mLookUpTableCalculator.SetCorrectionType(0);
  mLookUpTableCalculator.SetIntegrationStrategy(0);
  allocateMemory();
}

void SpaceCharge::allocateMemory()
{
  for (int iz = 0; iz < mNZSlices; ++iz) {
    mCoordZ[iz] = iz * mLengthZSlice;
  }
  for (int iphi = 0; iphi < mNPhiBins; ++iphi) {
    mCoordPhi[iphi] = iphi * mWidthPhiBin;
  }
  for (int ir = 0; ir < mNRBins; ++ir) {
    mCoordR[ir] = AliTPCPoissonSolver::fgkIFCRadius + ir * mLengthRBin;
  }

  mMatrixIonDriftZA = new TMatrixD*[mNPhiBins];
  mMatrixIonDriftRPhiA = new TMatrixD*[mNPhiBins];
  mMatrixIonDriftRA = new TMatrixD*[mNPhiBins];
  mMatrixIonDriftZC = new TMatrixD*[mNPhiBins];
  mMatrixIonDriftRPhiC = new TMatrixD*[mNPhiBins];
  mMatrixIonDriftRC = new TMatrixD*[mNPhiBins];
  for (int iphi = 0; iphi < mNPhiBins; ++iphi) {
    mMatrixIonDriftZA[iphi] = new TMatrixD(mNRBins, mNZSlices);
    mMatrixIonDriftRPhiA[iphi] = new TMatrixD(mNRBins, mNZSlices);
    mMatrixIonDriftRA[iphi] = new TMatrixD(mNRBins, mNZSlices);
    mMatrixIonDriftZC[iphi] = new TMatrixD(mNRBins, mNZSlices);
    mMatrixIonDriftRPhiC[iphi] = new TMatrixD(mNRBins, mNZSlices);
    mMatrixIonDriftRC[iphi] = new TMatrixD(mNRBins, mNZSlices);
  }
  mLookUpIonDriftA = std::make_unique<AliTPCLookUpTable3DInterpolatorD>(mNRBins, mMatrixIonDriftRA, mCoordR.data(), mNPhiBins, mMatrixIonDriftRPhiA, mCoordPhi.data(), mNZSlices, mMatrixIonDriftZA, mCoordZ.data(), mInterpolationOrder);
  mLookUpIonDriftC = std::make_unique<AliTPCLookUpTable3DInterpolatorD>(mNRBins, mMatrixIonDriftRC, mCoordR.data(), mNPhiBins, mMatrixIonDriftRPhiC, mCoordPhi.data(), mNZSlices, mMatrixIonDriftZC, mCoordZ.data(), mInterpolationOrder);
}

void SpaceCharge::init()
{
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

float SpaceCharge::calculateLookupTables()
{
  // Potential, E field and electron distortion and correction lookup tables
  TStopwatch timer;
  mLookUpTableCalculator.ForceInitSpaceCharge3DPoissonIntegralDz(mNRBins, mNZSlices, mNPhiBins, 300, 1e-8);
  float tRealCalc = static_cast<float>(timer.RealTime());

  // Lookup tables for local ion drift along E field
  if (mSCDistortionType == SCDistortionType::SCDistortionsRealistic) {
    TMatrixD* matrixDriftZ = nullptr;
    TMatrixD* matrixDriftRPhi = nullptr;
    TMatrixD* matrixDriftR = nullptr;
    for (int iside = 0; iside < 2; ++iside) {
      const int sign = 1 - iside * 2;
      for (int iphi = 0; iphi < mNPhiBins; ++iphi) {
        const float phi = static_cast<float>(mCoordPhi[iphi]);
        if (iside == 0) {
          matrixDriftZ = mMatrixIonDriftZA[iphi];
          matrixDriftRPhi = mMatrixIonDriftRPhiA[iphi];
          matrixDriftR = mMatrixIonDriftRA[iphi];
        } else {
          matrixDriftZ = mMatrixIonDriftZC[iphi];
          matrixDriftRPhi = mMatrixIonDriftRPhiC[iphi];
          matrixDriftR = mMatrixIonDriftRC[iphi];
        }
        int roc = iside == 0 ? o2::utils::Angle2Sector(phi) : o2::utils::Angle2Sector(phi) + 18;
        for (int ir = 0; ir < mNRBins; ++ir) {
          const float radius = static_cast<float>(mCoordR[ir]);
          /// TODO: what is the electric field stored in the LUTs at iz=0 and iz=mNZSlices-1
          for (int iz = 0; iz < mNZSlices; ++iz) {
            const float z = static_cast<float>(mCoordZ[iz]);
            float x0[3] = {radius, phi, sign * (z + static_cast<float>(mLengthZSlice))}; // iphi, ir, iz+1
            float x1[3] = {radius, phi, sign * z};                                       // iphi, ir, iz
            if (iside == 1) {
              x0[2] *= -1;
              x1[2] *= -1;
            }
            double eVector0[3] = {0., 0., 0.};
            double eVector1[3] = {0., 0., 0.};
            mLookUpTableCalculator.GetElectricFieldCyl(x0, roc, eVector0); // returns correct sign for Ez
            mLookUpTableCalculator.GetElectricFieldCyl(x1, roc, eVector1); // returns correct sign for Ez

            // drift of ions along E field
            (*matrixDriftR)(ir, iz) = -1 * sign * mLengthZSlice * 0.5 * (eVector0[0] + eVector1[0]) / (sign * sEzField + eVector0[2]);
            (*matrixDriftRPhi)(ir, iz) = -1 * sign * mLengthZSlice * 0.5 * (eVector0[1] + eVector1[1]) / (sign * sEzField + eVector0[2]);
            (*matrixDriftZ)(ir, iz) = -1 * sign * mLengthZSlice + DvDEoverv0 * mLengthZSlice * 0.5 * (eVector0[2] + eVector1[2]);
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
  // if (std::abs(eventTime - mTimeInit) < mDriftTimeZSlice) {
  //   return 0.f; // update only after one time bin has passed
  // }
  // mTimeInit = eventTime;

  std::unique_ptr<std::unique_ptr<TMatrixD>[]> spaceChargeA = std::make_unique<std::unique_ptr<TMatrixD>[]>(mNPhiBins);
  std::unique_ptr<std::unique_ptr<TMatrixD>[]> spaceChargeC = std::make_unique<std::unique_ptr<TMatrixD>[]>(mNPhiBins);
  for (int iphi = 0; iphi < mNPhiBins; ++iphi) {
    spaceChargeA[iphi] = std::make_unique<TMatrixD>(mNRBins, mNZSlices);
    spaceChargeC[iphi] = std::make_unique<TMatrixD>(mNRBins, mNZSlices);
  }
  for (int iside = 0; iside < 2; ++iside) {
    for (int iphi = 0; iphi < mNPhiBins; ++iphi) {
      TMatrixD& chargeDensity = iside == 0 ? *spaceChargeA[iphi] : *spaceChargeC[iphi];
      for (int ir = 0; ir < mNRBins; ++ir) {
        for (int iz = 0; iz < mNZSlices; ++iz) {
          if (iside == 0) {
            chargeDensity(ir, iz) = mSpaceChargeDensityA[iphi * mNRBins * mNZSlices + ir * mNZSlices + iz];
          } else {
            chargeDensity(ir, iz) = mSpaceChargeDensityC[iphi * mNRBins * mNZSlices + ir * mNZSlices + iz];
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

void SpaceCharge::setInitialSpaceChargeDensity(TH3* hisSCDensity)
{
  std::unique_ptr<std::unique_ptr<TMatrixD>[]> spaceChargeA = std::make_unique<std::unique_ptr<TMatrixD>[]>(mNPhiBins);
  std::unique_ptr<std::unique_ptr<TMatrixD>[]> spaceChargeC = std::make_unique<std::unique_ptr<TMatrixD>[]>(mNPhiBins);
  for (int iphi = 0; iphi < mNPhiBins; ++iphi) {
    spaceChargeA[iphi] = std::make_unique<TMatrixD>(mNRBins, mNZSlices);
    spaceChargeC[iphi] = std::make_unique<TMatrixD>(mNRBins, mNZSlices);
  }
  mLookUpTableCalculator.GetChargeDensity((TMatrixD**)spaceChargeA.get(), (TMatrixD**)spaceChargeC.get(), hisSCDensity, mNRBins, mNZSlices, mNPhiBins);
  for (int iside = 0; iside < 2; ++iside) {
    for (int iphi = 0; iphi < mNPhiBins; ++iphi) {
      TMatrixD& chargeDensity = iside == 0 ? *spaceChargeA[iphi] : *spaceChargeC[iphi];
      for (int ir = 0; ir < mNRBins; ++ir) {
        for (int iz = 0; iz < mNZSlices; ++iz) {
          if (iside == 0) {
            mSpaceChargeDensityA[iphi * mNRBins * mNZSlices + ir * mNZSlices + iz] = chargeDensity(ir, iz);
          } else {
            mSpaceChargeDensityC[iphi * mNRBins * mNZSlices + ir * mNZSlices + iz] = chargeDensity(ir, iz);
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
  const int zBin = TMath::BinarySearch(mNZSlices, mCoordZ.data(), std::abs(zdist));
  const int phiBin = TMath::BinarySearch(mNPhiBins, mCoordPhi.data(), phidist);
  const int rBin = TMath::BinarySearch(mNRBins, mCoordR.data(), rdist);
  /// TODO: protection against ions ending up outside the volume
  if (z > 0 && zdist > 0) {
    mSpaceChargeDensityA[phiBin * mNRBins * mNZSlices + rBin * mNZSlices + zBin] += ions2Charge(rBin, nIons);
  } else if (z < 0 && zdist < 0) {
    mSpaceChargeDensityC[phiBin * mNRBins * mNZSlices + rBin * mNZSlices + zBin] += ions2Charge(rBin, nIons);
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
  const int zBin = TMath::BinarySearch(mNZSlices, mCoordZ.data(), std::abs(zdist));
  const int phiBin = TMath::BinarySearch(mNPhiBins, mCoordPhi.data(), phidist);
  const int rBin = TMath::BinarySearch(mNRBins, mCoordR.data(), rdist);
  /// TODO: protection against ions ending up outside the volume
  if (z > 0 && zdist > 0) {
    mSpaceChargeDensityA[phiBin * mNRBins * mNZSlices + rBin * mNZSlices + zBin] += charge;
  } else if (z < 0 && zdist < 0) {
    mSpaceChargeDensityC[phiBin * mNRBins * mNZSlices + rBin * mNZSlices + zBin] += charge;
  }
}

void SpaceCharge::fillIBFIons(double r, double phi, Side side, int nIons)
{
  const int phiBin = TMath::BinarySearch(mNPhiBins, mCoordPhi.data(), phi);
  const int rBin = TMath::BinarySearch(mNRBins, mCoordR.data(), r);
  const int zBin = mNZSlices - 1;
  /// TODO: distribution of amplification ions instead of placing all of them in one point
  /// TODO: protection against ions ending up outside the volume
  if (side == Side::A) {
    mSpaceChargeDensityA[phiBin * mNRBins * mNZSlices + rBin * mNZSlices + zBin] += ions2Charge(rBin, nIons);
  } else {
    mSpaceChargeDensityC[phiBin * mNRBins * mNZSlices + rBin * mNZSlices + zBin] += ions2Charge(rBin, nIons);
  }
}

void SpaceCharge::fillIBFCharge(double r, double phi, Side side, float charge)
{
  const int phiBin = TMath::BinarySearch(mNPhiBins, mCoordPhi.data(), phi);
  const int rBin = TMath::BinarySearch(mNRBins, mCoordR.data(), r);
  const int zBin = mNZSlices - 1;
  /// TODO: distribution of amplification ions instead of placing all of them in one point
  /// TODO: protection against ions ending up outside the volume
  if (side == Side::A) {
    mSpaceChargeDensityA[phiBin * mNRBins * mNZSlices + rBin * mNZSlices + zBin] += charge;
  } else {
    mSpaceChargeDensityC[phiBin * mNRBins * mNZSlices + rBin * mNZSlices + zBin] += charge;
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
  // Calculate the change of the space-charge density rho_0 at time t0 after time delta(t) = mLengthZSlice / v_driftIon:
  //   delta(rho_0) = - ( rho_0 * div(u) + u * grad(rho_0) ) * delta(t)
  //                = - ( rho_0 * div(d) + d * grad(rho_0) )
  //       d: drift vector (dr, dphi, mLengthZSlice + dz)
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

  for (int iside = 0; iside < 2; ++iside) {
    const int signZ = iside == 0 ? 1 : -1;
    std::vector<float>* scDensity = nullptr;
    TMatrixD** matrixDriftZ = nullptr;
    TMatrixD** matrixDriftRPhi = nullptr;
    TMatrixD** matrixDriftR = nullptr;
    if (iside == 0) {
      scDensity = &mSpaceChargeDensityA;
      matrixDriftZ = mMatrixIonDriftZA;
      matrixDriftRPhi = mMatrixIonDriftRPhiA;
      matrixDriftR = mMatrixIonDriftRA;
    } else {
      scDensity = &mSpaceChargeDensityC;
      matrixDriftZ = mMatrixIonDriftZC;
      matrixDriftRPhi = mMatrixIonDriftRPhiC;
      matrixDriftR = mMatrixIonDriftRC;
    }
    /// TODO: is there a better way than to create a copy for the new SC density?
    std::vector<float> newSCDensity(mNPhiBins * mNRBins * mNZSlices);
    for (int iphi0 = 0; iphi0 < mNPhiBins; ++iphi0) {

      for (int ir0 = 0; ir0 < mNRBins; ++ir0) {
        const float r0 = static_cast<float>(mCoordR[ir0]);

        for (int iz0 = 0; iz0 < mNZSlices; ++iz0) {

          // rho_0 * div(d)
          float ddrdr = 0.f;
          if (ir0 > 1 && ir0 < mNRBins - 2) {
            ddrdr = (coeffCent40 * (*matrixDriftR[iphi0])(ir0 - 2, iz0) + coeffCent41 * (*matrixDriftR[iphi0])(ir0 - 1, iz0) + coeffCent42 * (*matrixDriftR[iphi0])(ir0 + 1, iz0) + coeffCent43 * (*matrixDriftR[iphi0])(ir0 + 2, iz0)) / static_cast<float>(mLengthRBin);
          } else if (ir0 < 2) {
            ddrdr = (coeffFwd30 * (*matrixDriftR[iphi0])(ir0, iz0) + coeffFwd31 * (*matrixDriftR[iphi0])(ir0 + 1, iz0) + coeffFwd32 * (*matrixDriftR[iphi0])(ir0 + 2, iz0) + coeffFwd33 * (*matrixDriftR[iphi0])(ir0 + 3, iz0)) / static_cast<float>(mLengthRBin);
          } else if (ir0 > (mNRBins - 3)) {
            ddrdr = -1 * (coeffFwd30 * (*matrixDriftR[iphi0])(ir0, iz0) + coeffFwd31 * (*matrixDriftR[iphi0])(ir0 - 1, iz0) + coeffFwd32 * (*matrixDriftR[iphi0])(ir0 - 2, iz0) + coeffFwd33 * (*matrixDriftR[iphi0])(ir0 - 3, iz0)) / static_cast<float>(mLengthRBin);
          }

          const int iphiCent0 = iphi0 - 2 + (mNPhiBins) * (iphi0 < 2);
          const int iphiCent1 = iphi0 - 1 + (mNPhiBins) * (iphi0 < 1);
          const int iphiCent3 = iphi0 + 1 - (mNPhiBins) * (iphi0 > (mNPhiBins - 2));
          const int iphiCent4 = iphi0 + 2 - (mNPhiBins) * (iphi0 > (mNPhiBins - 3));
          const float ddphidphi = (coeffCent40 * (*matrixDriftRPhi[iphiCent0])(ir0, iz0) + coeffCent41 * (*matrixDriftRPhi[iphiCent1])(ir0, iz0) + coeffCent42 * (*matrixDriftRPhi[iphiCent3])(ir0, iz0) + coeffCent43 * (*matrixDriftRPhi[iphiCent4])(ir0, iz0)) / static_cast<float>(mWidthPhiBin);

          float ddzdz = 0.f;
          if (iz0 > 1 && iz0 < mNZSlices - 2) {
            ddzdz = signZ * (coeffCent40 * (*matrixDriftZ[iphi0])(ir0, iz0 - 2) + coeffCent41 * (*matrixDriftZ[iphi0])(ir0, iz0 - 1) + coeffCent42 * (*matrixDriftZ[iphi0])(ir0, iz0 + 1) + coeffCent43 * (*matrixDriftZ[iphi0])(ir0, iz0 + 2)) / static_cast<float>(mLengthRBin);
          } else if (iz0 == 1 || iz0 == (mNZSlices - 2)) {
            ddzdz = signZ * (-0.5 * (*matrixDriftZ[iphi0])(ir0, iz0 - 1) + 0.5 * (*matrixDriftZ[iphi0])(ir0, iz0 + 1)) / static_cast<float>(mLengthRBin);
          } else if (iz0 == 0) {
            ddzdz = signZ * (coeffFwd20 * (*matrixDriftZ[iphi0])(ir0, iz0) + coeffFwd21 * (*matrixDriftZ[iphi0])(ir0, iz0 + 1) + coeffFwd22 * (*matrixDriftZ[iphi0])(ir0, iz0 + 2)) / static_cast<float>(mLengthRBin);
          } else if (iz0 == (mNZSlices - 1)) {
            ddzdz = -1 * signZ * (coeffFwd20 * (*matrixDriftZ[iphi0])(ir0, iz0) + coeffFwd21 * (*matrixDriftZ[iphi0])(ir0, iz0 - 1) + coeffFwd22 * (*matrixDriftZ[iphi0])(ir0, iz0 - 2)) / static_cast<float>(mLengthRBin);
          }

          const float qdivd = (*scDensity)[iphi0 * mNRBins * mNZSlices + ir0 * mNZSlices + iz0] * (((*matrixDriftR[iphi0])(ir0, iz0) + ddphidphi) / r0 + ddrdr + ddzdz);

          // - d * grad(rho_0) = - d_drift * grad(rho_0(x)) + d_dist * grad(rho_0(x+d_drift)) = (charge0 - charge1) + d_dist * grad(rho_0(x-d_drift))
          if (iz0 < (mNZSlices - 1)) {
            const float dr = (*matrixDriftR[iphi0])(ir0, iz0);
            const float drphi = (*matrixDriftRPhi[iphi0])(ir0, iz0);
            const float dz = (*matrixDriftZ[iphi0])(ir0, iz0) + mLengthZSlice * signZ;

            const int ir1 = dr < 0 ? ir0 - 1 * (ir0 == (mNRBins - 1)) : ir0 - 1 + 1 * (ir0 == 0);
            const int ir2 = dr < 0 ? ir0 + 1 - 1 * (ir0 == (mNRBins - 1)) : ir0 + 1 * (ir0 == 0);
            const int iphi1 = drphi < 0 ? iphi0 : iphi0 - 1 + (mNPhiBins) * (iphi0 == 0);
            const int iphi2 = drphi < 0 ? iphi0 + 1 - (mNPhiBins) * (iphi0 == (mNPhiBins - 1)) : iphi0;
            const int iz1 = dz < 0 ? iz0 + 1 - 1 * (iside == 0) * (iz0 == (mNZSlices - 2)) : iz0 + 2 * (iside == 1) - 1 * (iside == 1) * (iz0 == (mNZSlices - 2));
            const int iz2 = dz < 0 ? iz0 + 2 - 2 * (iside == 1) - 1 * (iside == 0) * (iz0 == (mNZSlices - 2)) : iz0 + 1 - 1 * (iside == 1) * (iz0 == (mNZSlices - 2));

            const float dqdr = ((*scDensity)[iphi0 * mNRBins * mNZSlices + ir2 * mNZSlices + (iz0 + 1)] - (*scDensity)[iphi0 * mNRBins * mNZSlices + ir1 * mNZSlices + (iz0 + 1)]) / static_cast<float>(mLengthRBin);
            const float dqdphi = ((*scDensity)[iphi2 * mNRBins * mNZSlices + ir0 * mNZSlices + (iz0 + 1)] - (*scDensity)[iphi1 * mNRBins * mNZSlices + ir0 * mNZSlices + (iz0 + 1)]) / static_cast<float>(mWidthPhiBin);
            const float dqdz = ((*scDensity)[iphi0 * mNRBins * mNZSlices + ir0 * mNZSlices + iz2] - (*scDensity)[iphi0 * mNRBins * mNZSlices + ir0 * mNZSlices + iz1]) / static_cast<float>(mLengthZSlice);

            const float dgradq = dr * dqdr + drphi / r0 * dqdphi + dz * dqdz;

            newSCDensity[iphi0 * mNRBins * mNZSlices + ir0 * mNZSlices + iz0] = (*scDensity)[iphi0 * mNRBins * mNZSlices + ir0 * mNZSlices + (iz0 + 1)] - (qdivd + dgradq);
          } else {
            newSCDensity[iphi0 * mNRBins * mNZSlices + ir0 * mNZSlices + iz0] = -qdivd;
          }

          if (newSCDensity[iphi0 * mNRBins * mNZSlices + ir0 * mNZSlices + iz0] < 0.f) {
            newSCDensity[iphi0 * mNRBins * mNZSlices + ir0 * mNZSlices + iz0] = 0.f;
          }
        }
      }
    }
    if (iside == 0) {
      mSpaceChargeDensityA = newSCDensity;
    } else {
      mSpaceChargeDensityC = newSCDensity;
    }
  }
}

void SpaceCharge::propagateIons()
{
  for (int iside = 0; iside < 2; ++iside) {
    const int signZ = iside == 0 ? 1 : -1;
    std::vector<float>* scDensity = nullptr;
    AliTPCLookUpTable3DInterpolatorD* lookUpIonDrift = nullptr;
    if (iside == 0) {
      scDensity = &mSpaceChargeDensityA;
      lookUpIonDrift = mLookUpIonDriftA.get();
    } else {
      scDensity = &mSpaceChargeDensityC;
      lookUpIonDrift = mLookUpIonDriftC.get();
    }
    std::vector<float> newSCDensity(mNPhiBins * mNRBins * mNZSlices);

    for (int iphi0 = 0; iphi0 < mNPhiBins; ++iphi0) {
      const double phi0 = static_cast<float>(mCoordPhi[iphi0]);

      for (int ir0 = 0; ir0 < mNRBins; ++ir0) {
        const double r0 = static_cast<float>(mCoordR[ir0]);
        const double r1 = r0 + mLengthRBin;

        for (int iz0 = 0; iz0 < mNZSlices; ++iz0) {
          const double z0 = mCoordZ[iz0] * signZ;

          const float ionDensity = (*scDensity)[iphi0 * mNRBins * mNZSlices + ir0 * mNZSlices + iz0] * AliTPCPoissonSolver::fgke0 / TMath::Qe();  // #ions / cm^3
          const float nIons = std::round(ionDensity * mLengthZSlice * 0.5 * mWidthPhiBin * (r1 * r1 - r0 * r0));  // absolute #ions in the voxel

          for (int iion=0; iion<nIons; ++iion){
            double phiIon = phi0 + mRandomFlat.getNextValue() * mWidthPhiBin;
            double rIon = r0 + mRandomFlat.getNextValue() * mLengthRBin;
            double zIon = z0 + mRandomFlat.getNextValue() * mLengthZSlice * signZ;

            double drphi = 0.f;
            double dr = 0.f;
            double dz = 0.f;
            lookUpIonDrift->GetValue(rIon, phiIon, std::abs(zIon), dr, drphi, dz);
            float phiIonF = static_cast<float>(phiIon + (drphi / rIon));
            o2::utils::BringTo02PiGen(phiIonF);
            rIon += dr;
            zIon += dz;

            // continue if ion is outside the TPC boundaries in r or z
            if (rIon > (mCoordR[mNRBins-1]+mLengthRBin) || rIon < mCoordR[0]){
              continue;
            }
            if ((zIon * signZ) > (mCoordZ[mNZSlices-1]+mLengthZSlice) || (zIon * signZ) < mCoordZ[0]){
              continue;
            }

            const int iphiDrift = TMath::BinarySearch(mNPhiBins, mCoordPhi.data(), static_cast<double>(phiIonF));
            const int irDrift = TMath::BinarySearch(mNRBins, mCoordR.data(), rIon);
            const int izDrift = TMath::BinarySearch(mNZSlices, mCoordZ.data(), std::abs(zIon));
            newSCDensity[iphiDrift * mNRBins * mNZSlices + irDrift * mNZSlices + izDrift] += ions2Charge(irDrift, 1);
          }

        }
      }
    }
    if (iside == 0) {
      mSpaceChargeDensityA = newSCDensity;
    } else {
      mSpaceChargeDensityC = newSCDensity;
    }
  }
}

void SpaceCharge::getIonDrift(Side side, double r, double phi, double z, double& dr, double& drphi, double& dz)
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

void SpaceCharge::distortElectron(GlobalPosition3D& point)
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

double SpaceCharge::getChargeDensity(Side side, GlobalPosition3D& point)
{
  Float_t x[3] = {point.rho(), point.phi(), point.z()};
  o2::utils::BringTo02PiGen(x[1]);
  const int roc = side == Side::A ? o2::utils::Angle2Sector(x[1]) : o2::utils::Angle2Sector(x[1]) + 18;
  return mLookUpTableCalculator.GetChargeCylAC(x, roc);
}

float SpaceCharge::getChargeDensity(Side side, int iphi, int ir, int iz)
{
  if (side == Side::A) {
    return mSpaceChargeDensityA[iphi * mNRBins * mNZSlices + ir * mNZSlices + iz];
  } else if (side == Side::C) {
    return mSpaceChargeDensityC[iphi * mNRBins * mNZSlices + ir * mNZSlices + iz];
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
  float rOuter = mCoordR[rBin] + mLengthRBin;
  return nIons * TMath::Qe() / (mLengthZSlice * 0.5 * mWidthPhiBin * (rOuter * rOuter - rInner * rInner)) / AliTPCPoissonSolver::fgke0;
}
