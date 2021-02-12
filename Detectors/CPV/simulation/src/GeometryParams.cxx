// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CPVSimulation/GeometryParams.h"
#include "TMath.h"

using namespace o2::cpv;

ClassImp(GeometryParams);

GeometryParams* GeometryParams::sGeomParam = nullptr;

//____________________________________________________________________________
GeometryParams::GeometryParams(const std::string_view name)
  : // Set zeros to the variables: most of them should be calculated
    // and it is more clear to set them in the text
    mNModules(5),
    mNumberOfCPVPadsPhi(128),
    mNumberOfCPVPadsZ(60),
    mCPVPadSizePhi(1.13),
    mCPVPadSizeZ(2.1093),
    mNumberOfCPVChipsPhi(8),
    mNumberOfCPVChipsZ(8),
    mCPVGasThickness(1.3),
    mCPVTextoliteThickness(0.1),
    mCPVCuNiFoilThickness(56.e-04)

{
  // Initializes the EMC parameters
  // Coordinate system chosen: x across beam, z along beam, y out of beam.
  // Reference point for all volumes incide module is
  // center of module in x,z on the upper surface of support beam

  // Distance from IP to front surface of CPV
  mIPtoCPVSurface = 449.310 - 5.2 - 2.61; //Distance to PHOS fron sutface - CPV size

  // Calculate distance from IP to upper cover
  // mIPtoOuterCoverDistance = mIPtoCrystalSurface - mAirGapLed - mInnerThermoWidthY - mAirGapWidthY - mCoolerWidthY -
  //                           mOuterThermoWidthY - mAlCoverThickness - mzAirTightBoxToTopModuleDist - mATBoxWall;

  // double tanA = mOuterThermoWidthXUp / (2. * mIPtoOuterCoverDistance);
  // tan(a) where A = angle between IP to center and IP to side across beam

  // Initializes the CPV parameters
  mCPVFrameSize[0] = 2.5;
  mCPVFrameSize[1] = 5.1;
  mCPVFrameSize[2] = 2.5;
  mGassiplexChipSize[0] = 4.2;
  mGassiplexChipSize[1] = 0.1;
  mGassiplexChipSize[2] = 6.0;
  mFTPosition[0] = 0.7;
  mFTPosition[1] = 2.2;
  mFTPosition[2] = 3.6;
  mFTPosition[3] = 5.1;

  mCPVActiveSize[0] = mNumberOfCPVPadsPhi * mCPVPadSizePhi;
  mCPVActiveSize[1] = mNumberOfCPVPadsZ * mCPVPadSizeZ;
  mCPVBoxSize[0] = mCPVActiveSize[0] + 2 * mCPVFrameSize[0];
  mCPVBoxSize[1] = mCPVFrameSize[1] + 0.1;
  mCPVBoxSize[2] = mCPVActiveSize[1] + 2 * mCPVFrameSize[2];

  double const moduleAngle = 20.;
  double const kRADDEG = 180.0 / TMath::Pi();

  double r = mIPtoCPVSurface + mCPVBoxSize[1];
  for (Int_t iModule = 2; iModule < mNModules; iModule++) {
    double angle = moduleAngle * (iModule - 2); //Module 2 just below IP
    mCPVAngle[iModule] = -angle;
    mModuleCenter[iModule][0] = r * TMath::Sin(-angle / kRADDEG);
    mModuleCenter[iModule][1] = -r * TMath::Cos(-angle / kRADDEG);
    mModuleCenter[iModule][2] = 0.;

    mModuleAngle[iModule][0][0] = 90;         //thetaX polar angle for axis X
    mModuleAngle[iModule][0][1] = -angle;     //phiX azimuthal angle for axis X
    mModuleAngle[iModule][1][0] = 90;         //thetaY polar angle for axis Y
    mModuleAngle[iModule][1][1] = 90 - angle; //phiY azimuthal angle for axis Y
    mModuleAngle[iModule][2][0] = 0;          //thetaZ polar angle for axis Z
    mModuleAngle[iModule][2][1] = 0;          //phiZ azimuthal angle for axis Z
  }
}
