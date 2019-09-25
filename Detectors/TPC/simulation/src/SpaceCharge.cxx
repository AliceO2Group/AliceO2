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

#include "AliTPCPoissonSolver.h"

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
    mLengthZSlice(DriftLength / MaxZSlices),
    mLengthTimeSlice(IonDriftTime / MaxZSlices),
    mWidthPhiBin(o2::constants::math::TwoPI / MaxPhiBins),
    mLengthRBin((RadiusOuter - RadiusInner) / Constants::MAXGLOBALPADROW),
    mCoordZ(MaxZSlices),
    mCoordPhi(MaxPhiBins),
    mCoordR(Constants::MAXGLOBALPADROW),
    mInterpolationOrder(2),
    mUseInitialSCDensity(false),
    mInitLookUpTables(false),
    mTimeInit(-1),
    mSCDistortionType(SpaceCharge::SCDistortionType::SCDistortionsRealistic),
    mLookUpTableCalculator(Constants::MAXGLOBALPADROW, MaxZSlices, MaxPhiBins, 2, 3, 0),
    mSpaceChargeDensityA(MaxZSlices),
    mSpaceChargeDensityC(MaxZSlices)
{
  allocateMemory();
}

SpaceCharge::SpaceCharge(int nZSlices, int nPhiBins, int nRBins)
  : mNZSlices(nZSlices),
    mNPhiBins(nPhiBins),
    mNRBins(nRBins),
    mLengthZSlice(DriftLength / nZSlices),
    mLengthTimeSlice(IonDriftTime / nZSlices),
    mWidthPhiBin(o2::constants::math::TwoPI / nPhiBins),
    mLengthRBin((RadiusOuter - RadiusInner) / nRBins),
    mCoordZ(nZSlices),
    mCoordPhi(nPhiBins),
    mCoordR(nRBins),
    mInterpolationOrder(2),
    mUseInitialSCDensity(false),
    mInitLookUpTables(false),
    mTimeInit(-1),
    mSCDistortionType(SpaceCharge::SCDistortionType::SCDistortionsRealistic),
    mLookUpTableCalculator(nRBins, nZSlices, nPhiBins, 2, 3, 0),
    mSpaceChargeDensityA(nZSlices),
    mSpaceChargeDensityC(nZSlices)
{
  allocateMemory();
}

SpaceCharge::SpaceCharge(int nZSlices, int nPhiBins, int nRBins, int interpolationOrder)
  : mNZSlices(nZSlices),
    mNPhiBins(nPhiBins),
    mNRBins(nRBins),
    mLengthZSlice(DriftLength / nZSlices),
    mLengthTimeSlice(IonDriftTime / nZSlices),
    mWidthPhiBin(o2::constants::math::TwoPI / nPhiBins),
    mLengthRBin((RadiusOuter - RadiusInner) / nRBins),
    mCoordZ(nZSlices),
    mCoordPhi(nPhiBins),
    mCoordR(nRBins),
    mInterpolationOrder(interpolationOrder),
    mUseInitialSCDensity(false),
    mInitLookUpTables(false),
    mTimeInit(-1),
    mSCDistortionType(SpaceCharge::SCDistortionType::SCDistortionsRealistic),
    mLookUpTableCalculator(nRBins, nZSlices, nPhiBins, interpolationOrder, 3, 0),
    mSpaceChargeDensityA(nZSlices),
    mSpaceChargeDensityC(nZSlices)
{
  allocateMemory();
}

void SpaceCharge::allocateMemory()
{
  for (auto i = 0; i < mNZSlices; ++i) {
    mSpaceChargeDensityA[i].resize(mNPhiBins * mNRBins);
    mSpaceChargeDensityC[i].resize(mNPhiBins * mNRBins);
  }

  for (int iz = 0; iz < mNZSlices; ++iz)
    mCoordZ[iz] = (iz + 1) * mLengthZSlice;
  for (int iphi = 0; iphi < mNPhiBins; ++iphi)
    mCoordPhi[iphi] = (iphi + 1) * mWidthPhiBin;
  for (int ir = 0; ir < mNRBins; ++ir)
    mCoordR[ir] = (ir + 1) * mLengthRBin;

  mMatrixLocalIonDriftDzA = new TMatrixD*[mNPhiBins];
  mMatrixLocalIonDriftDrphiA = new TMatrixD*[mNPhiBins];
  mMatrixLocalIonDriftDrA = new TMatrixD*[mNPhiBins];
  mMatrixLocalIonDriftDzC = new TMatrixD*[mNPhiBins];
  mMatrixLocalIonDriftDrphiC = new TMatrixD*[mNPhiBins];
  mMatrixLocalIonDriftDrC = new TMatrixD*[mNPhiBins];
  for (int iphi = 0; iphi < mNPhiBins; ++iphi) {
    mMatrixLocalIonDriftDzA[iphi] = new TMatrixD(mNRBins, mNZSlices);
    mMatrixLocalIonDriftDrphiA[iphi] = new TMatrixD(mNRBins, mNZSlices);
    mMatrixLocalIonDriftDrA[iphi] = new TMatrixD(mNRBins, mNZSlices);
    mMatrixLocalIonDriftDzC[iphi] = new TMatrixD(mNRBins, mNZSlices);
    mMatrixLocalIonDriftDrphiC[iphi] = new TMatrixD(mNRBins, mNZSlices);
    mMatrixLocalIonDriftDrC[iphi] = new TMatrixD(mNRBins, mNZSlices);
  }
  mLookUpLocalIonDriftA = std::make_unique<AliTPCLookUpTable3DInterpolatorD>(mNRBins, mMatrixLocalIonDriftDrA, mCoordR.data(), mNPhiBins, mMatrixLocalIonDriftDrphiA, mCoordPhi.data(), mNZSlices, mMatrixLocalIonDriftDzA, mCoordZ.data(), mInterpolationOrder);
  mLookUpLocalIonDriftC = std::make_unique<AliTPCLookUpTable3DInterpolatorD>(mNRBins, mMatrixLocalIonDriftDrC, mCoordR.data(), mNPhiBins, mMatrixLocalIonDriftDrphiC, mCoordPhi.data(), mNZSlices, mMatrixLocalIonDriftDzC, mCoordZ.data(), mInterpolationOrder);
}

void SpaceCharge::init()
{
  auto o2field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  float bzField = o2field->solenoidField(); // magnetic field in kGauss
  /// TODO is there a faster way to get the drift velocity
  auto& gasParam = ParameterGas::Instance();
  float vDrift = gasParam.DriftV; // drift velocity in cm/us
  /// TODO fix hard coded values (ezField, t1, t2): export to Constants.h or get from somewhere?
  float t1 = 1.;
  float t2 = 1.;
  /// TODO use this parameterization or fixed value(s) from Magboltz calculations?
  float omegaTau = -10. * bzField * vDrift / TMath::Abs(sEzField);
  setOmegaTauT1T2(omegaTau, t1, t2);
  if (mUseInitialSCDensity) {
    calculateLookupTables();
  }
}

void SpaceCharge::calculateLookupTables()
{
  // Potential, E field and electron distortion and correction lookup tables
  mLookUpTableCalculator.ForceInitSpaceCharge3DPoissonIntegralDz(mNRBins, mNZSlices, mNPhiBins, 300, 1e-8);

  // Lookup tables for local ion drift along E field
  if (mSCDistortionType == SCDistortionType::SCDistortionsRealistic) {
    for (int iphi = 0; iphi < mNPhiBins; ++iphi) {
      float phi = mCoordPhi[iphi];
      TMatrixD& matrixDzA = *mMatrixLocalIonDriftDzA[iphi];
      TMatrixD& matrixDzC = *mMatrixLocalIonDriftDzC[iphi];
      TMatrixD& matrixDrphiA = *mMatrixLocalIonDriftDrphiA[iphi];
      TMatrixD& matrixDrphiC = *mMatrixLocalIonDriftDrphiC[iphi];
      TMatrixD& matrixDrA = *mMatrixLocalIonDriftDrA[iphi];
      TMatrixD& matrixDrC = *mMatrixLocalIonDriftDrC[iphi];
      int roc = o2::utils::Angle2Sector(phi);
      for (int ir = 0; ir < mNRBins; ++ir) {
        float radius = mCoordR[ir];
        for (int iz = 0; iz < mNZSlices; ++iz) {
          // A side
          float z = mCoordZ[iz];
          float x0[3] = {radius, phi, z};
          float x1[3] = {radius, phi, z - mLengthZSlice};
          double eVector0[3] = {0.f, 0.f, 0.f};
          double eVector1[3] = {0.f, 0.f, 0.f};
          mLookUpTableCalculator.GetElectricFieldCyl(x0, roc, eVector0);
          mLookUpTableCalculator.GetElectricFieldCyl(x1, roc, eVector1);
          matrixDzA(ir, iz) = DvDEoverv0 * (mLengthZSlice / 2.f) * (eVector0[2] + eVector1[2]);
          matrixDrphiA(ir, iz) = (mLengthZSlice / 2.f) * (eVector0[1] + eVector1[1]) / sEzField;
          matrixDrA(ir, iz) = (mLengthZSlice / 2.f) * (eVector0[0] + eVector1[0]) / sEzField;
          // C side
          x0[2] *= -1;
          x1[2] *= -1;
          mLookUpTableCalculator.GetElectricFieldCyl(x0, roc + 18, eVector0);
          mLookUpTableCalculator.GetElectricFieldCyl(x1, roc + 18, eVector1);
          matrixDzC(ir, iz) = DvDEoverv0 * (mLengthZSlice / 2.f) * (eVector0[2] + eVector1[2]);
          matrixDrphiC(ir, iz) = (mLengthZSlice / 2.f) * (eVector0[1] + eVector1[1]) / sEzField;
          matrixDrC(ir, iz) = (mLengthZSlice / 2.f) * (eVector0[0] + eVector1[0]) / sEzField;
        }
      }
    }
    mLookUpLocalIonDriftA->CopyFromMatricesToInterpolator();
    mLookUpLocalIonDriftC->CopyFromMatricesToInterpolator();

    /// TODO: Propagate current SC density along E field by one time bin for next update
  }

  mInitLookUpTables = true;
}

void SpaceCharge::updateLookupTables(float eventTime)
{
  if (mTimeInit < 0.) {
    mTimeInit = eventTime; // set the time of first initialization
  }
  if (TMath::Abs(eventTime - mTimeInit) < mLengthTimeSlice) {
    return; // update only after one time bin has passed
  }

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
            chargeDensity(ir, iz) = mSpaceChargeDensityA[iz][ir + iphi * mNRBins];
          } else {
            chargeDensity(ir, iz) = mSpaceChargeDensityC[iz][ir + iphi * mNRBins];
          }
        }
      }
    }
  }
  mLookUpTableCalculator.SetInputSpaceChargeA((TMatrixD**)spaceChargeA.get());
  mLookUpTableCalculator.SetInputSpaceChargeC((TMatrixD**)spaceChargeC.get());
  calculateLookupTables();
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
            mSpaceChargeDensityA[iz][ir + iphi * mNRBins] = chargeDensity(ir, iz);
          } else {
            mSpaceChargeDensityC[iz][ir + iphi * mNRBins] = chargeDensity(ir, iz);
          }
        }
      }
    }
  }
  mLookUpTableCalculator.SetInputSpaceChargeA((TMatrixD**)spaceChargeA.get());
  mLookUpTableCalculator.SetInputSpaceChargeC((TMatrixD**)spaceChargeC.get());
  mUseInitialSCDensity = true;
}

void SpaceCharge::fillSCDensity(float zPos, float phiPos, float rPos, int nIons)
{
  const int zBin = TMath::Abs(zPos) / mLengthZSlice;
  const int phiBin = phiPos / mWidthPhiBin;
  const int rBin = rPos / mLengthRBin;
  if (zPos > 0) {
    mSpaceChargeDensityA[zBin][rBin + phiBin * mNRBins] += ions2Charge(nIons);
  } else {
    mSpaceChargeDensityC[zBin][rBin + phiBin * mNRBins] += ions2Charge(nIons);
  }
}

/// TODO
// void SpaceCharge::propagateSpaceCharge()
// {
//
// }

// GlobalPosition3D SpaceCharge::driftIon(GlobalPosition3D &point);
// {
//   GlobalPosition3D posIonDistorted(point.X(),point.Y(),point.Z());
//   if (!mInitLookUpTables) return posIonDistorted;
//
// }

void SpaceCharge::correctElectron(GlobalPosition3D& point)
{
  if (!mInitLookUpTables) {
    return;
  }
  const float x[3] = {point.X(), point.Y(), point.Z()};
  float dx[3] = {0.f, 0.f, 0.f};
  float phi = TMath::ATan2(x[1], x[0]);
  if (phi < 0) {
    phi += TMath::TwoPi();
  }
  int roc = phi / TMath::Pi() * 9;
  if (x[2] < 0) {
    roc += 18;
  }
  mLookUpTableCalculator.GetCorrection(x, roc, dx);
  point = GlobalPosition3D(x[0] + dx[0], x[1] + dx[1], x[2] + dx[2]);
}

void SpaceCharge::distortElectron(GlobalPosition3D& point)
{
  if (!mInitLookUpTables) {
    return;
  }
  const float x[3] = {point.X(), point.Y(), point.Z()};
  float dx[3] = {0.f, 0.f, 0.f};
  float phi = TMath::ATan2(x[1], x[0]);
  if (phi < 0) {
    phi += TMath::TwoPi();
  }
  int roc = phi / TMath::Pi() * 9;
  if (x[2] < 0) {
    roc += 18;
  }
  mLookUpTableCalculator.GetDistortion(x, roc, dx);
  point = GlobalPosition3D(x[0] + dx[0], x[1] + dx[1], x[2] + dx[2]);
}

float SpaceCharge::ions2Charge(int nIons)
{
  return 1.e6f * nIons * TMath::Qe() / (mLengthZSlice * (mWidthPhiBin * mLengthRBin) * mLengthRBin);
}
