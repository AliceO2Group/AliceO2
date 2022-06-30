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

// $Id$
//
//-----------------------------------------------------------------------------
/// \class MisAligner
///
/// This performs the misalignment on an existing muon arm geometry
/// based on the standard definition of the detector elements in
/// $ALICE_ROOT/MUON/data
///
/// --> User has to specify the magnitude of the alignments, in the Cartesian
/// co-ordiantes (which are used to apply translation misalignments) and in the
/// spherical co-ordinates (which are used to apply angular displacements)
///
/// --> If the constructor is used with no arguments, user has to set
/// misalignment ranges by hand using the methods :
/// SetApplyMisAlig, SetMaxCartMisAlig, SetMaxAngMisAlig, SetXYAngMisAligFactor
/// (last method takes account of the fact that the misalingment is greatest in
/// the XY plane, since the detection elements are fixed to a support structure
/// in this plane. Misalignments in the XZ and YZ plane will be very small
/// compared to those in the XY plane, which are small already - of the order
/// of microns)
///
/// Note : If the detection elements are allowed to be misaligned in all
/// directions, this has consequences for the alignment algorithm
/// (AliMUONAlignment), which needs to know the number of free parameters.
/// Eric only allowed 3 :  x,y,theta_xy, but in principle z and the other
/// two angles are alignable as well.
///
/// \author Bruce Becker, Javier Castillo
//-----------------------------------------------------------------------------

#include "MCHGeometryMisAligner/MisAligner.h"

#include <TGeoMatrix.h>
#include <TMath.h>
#include <TRandom.h>
#include <Riostream.h>

#include "DetectorsCommonDataFormats/AlignParam.h"

#include "Framework/Logger.h"

ClassImp(o2::mch::geo::MisAligner);

namespace o2::mch::geo
{

bool MisAligner::isMatrixConvertedToAngles(const double* rot, double& psi, double& theta, double& phi) const
{
  /// Calculates the Euler angles in "x y z" notation
  /// using the rotation matrix
  /// Returns false in case the rotation angles can not be
  /// extracted from the matrix
  //
  if (std::abs(rot[0]) < 1e-7 || std::abs(rot[8]) < 1e-7) {
    LOG(error) << "Failed to extract roll-pitch-yall angles!";
    return false;
  }
  psi = std::atan2(-rot[5], rot[8]);
  theta = std::asin(rot[2]);
  phi = std::atan2(-rot[1], rot[0]);
  return true;
}
//______________________________________________________________________________
MisAligner::MisAligner(double cartXMisAligM, double cartXMisAligW, double cartYMisAligM, double cartYMisAligW, double angMisAligM, double angMisAligW)
  : TObject(), mUseUni(kFALSE), mUseGaus(kTRUE), mXYAngMisAligFactor(0.0), mZCartMisAligFactor(0.0)
{
  /// Standard constructor
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 2; j++) {
      mDetElemMisAlig[i][j] = 0.0;
      mModuleMisAlig[i][j] = 0.0;
    }
  }
  mDetElemMisAlig[0][0] = cartXMisAligM;
  mDetElemMisAlig[0][1] = cartXMisAligW;
  mDetElemMisAlig[1][0] = cartYMisAligM;
  mDetElemMisAlig[1][1] = cartYMisAligW;
  mDetElemMisAlig[5][0] = angMisAligM;
  mDetElemMisAlig[5][1] = angMisAligW;
}

//______________________________________________________________________________
MisAligner::MisAligner(double cartMisAligM, double cartMisAligW, double angMisAligM, double angMisAligW)
  : TObject(), mUseUni(kFALSE), mUseGaus(kTRUE), mXYAngMisAligFactor(0.0), mZCartMisAligFactor(0.0)
{
  /// Standard constructor
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 2; j++) {
      mDetElemMisAlig[i][j] = 0.0;
      mModuleMisAlig[i][j] = 0.0;
    }
  }
  mDetElemMisAlig[0][0] = cartMisAligM;
  mDetElemMisAlig[0][1] = cartMisAligW;
  mDetElemMisAlig[1][0] = cartMisAligM;
  mDetElemMisAlig[1][1] = cartMisAligW;
  mDetElemMisAlig[5][0] = angMisAligM;
  mDetElemMisAlig[5][1] = angMisAligW;
}

//______________________________________________________________________________
MisAligner::MisAligner(double cartMisAlig, double angMisAlig)
  : TObject(),
    mUseUni(kTRUE),
    mUseGaus(kFALSE),
    mXYAngMisAligFactor(0.0),
    mZCartMisAligFactor(0.0)
{
  /// Standard constructor
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 2; j++) {
      mDetElemMisAlig[i][j] = 0.0;
      mModuleMisAlig[i][j] = 0.0;
    }
  }
  mDetElemMisAlig[0][1] = cartMisAlig;
  mDetElemMisAlig[1][1] = cartMisAlig;
  mDetElemMisAlig[5][1] = angMisAlig;
}

//_____________________________________________________________________________
MisAligner::MisAligner()
  : TObject(),
    mUseUni(kFALSE),
    mUseGaus(kTRUE),
    mXYAngMisAligFactor(0.0),
    mZCartMisAligFactor(0.0)
{
  /// Default constructor
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 2; j++) {
      mDetElemMisAlig[i][j] = 0.0;
      mModuleMisAlig[i][j] = 0.0;
    }
  }
}

/*
//______________________________________________________________________________
MisAligner::~MisAligner()
{
  /// Destructor
}
 */

//_________________________________________________________________________
void MisAligner::setXYAngMisAligFactor(double factor)
{
  /// Set XY angular misalign factor

  if (TMath::Abs(factor) > 1.0 && factor > 0.) {
    mXYAngMisAligFactor = factor;
    mDetElemMisAlig[3][0] = mDetElemMisAlig[5][0] * factor; // These lines were
    mDetElemMisAlig[3][1] = mDetElemMisAlig[5][1] * factor; // added to keep
    mDetElemMisAlig[4][0] = mDetElemMisAlig[5][0] * factor; // backward
    mDetElemMisAlig[4][1] = mDetElemMisAlig[5][1] * factor; // compatibility
  } else {
    LOG(error) << "Invalid XY angular misalign factor, " << factor;
  }
}

//_________________________________________________________________________
void MisAligner::setZCartMisAligFactor(double factor)
{
  /// Set XY angular misalign factor
  if (TMath::Abs(factor) < 1.0 && factor > 0.) {
    mZCartMisAligFactor = factor;
    mDetElemMisAlig[2][0] = mDetElemMisAlig[0][0];          // These lines were added to
    mDetElemMisAlig[2][1] = mDetElemMisAlig[0][1] * factor; // keep backward compatibility
  } else {
    LOG(error) << Form("Invalid Z cartesian misalign factor, %f", factor);
  }
}

//_________________________________________________________________________
void MisAligner::getUniMisAlign(double cartMisAlig[3], double angMisAlig[3], const double lParMisAlig[6][2]) const
{
  /// Misalign using uniform distribution
  /**
   misalign the centre of the local transformation
   rotation axes :
   fAngMisAlig[1,2,3] = [x,y,z]
   Assume that misalignment about the x and y axes (misalignment of z plane)
   is much smaller, since the entire detection plane has to be moved (the
   detection elements are on a support structure), while rotation of the x-y
   plane is more free.
   */
  cartMisAlig[0] = gRandom->Uniform(-lParMisAlig[0][1] + lParMisAlig[0][0], lParMisAlig[0][0] + lParMisAlig[0][1]);
  cartMisAlig[1] = gRandom->Uniform(-lParMisAlig[1][1] + lParMisAlig[1][0], lParMisAlig[1][0] + lParMisAlig[1][1]);
  cartMisAlig[2] = gRandom->Uniform(-lParMisAlig[2][1] + lParMisAlig[2][0], lParMisAlig[2][0] + lParMisAlig[2][1]);

  angMisAlig[0] = gRandom->Uniform(-lParMisAlig[3][1] + lParMisAlig[3][0], lParMisAlig[3][0] + lParMisAlig[3][1]);
  angMisAlig[1] = gRandom->Uniform(-lParMisAlig[4][1] + lParMisAlig[4][0], lParMisAlig[4][0] + lParMisAlig[4][1]);
  angMisAlig[2] = gRandom->Uniform(-lParMisAlig[5][1] + lParMisAlig[5][0], lParMisAlig[5][0] + lParMisAlig[5][1]); // degrees
}

//_________________________________________________________________________
void MisAligner::getGausMisAlign(double cartMisAlig[3], double angMisAlig[3], const double lParMisAlig[6][2]) const
{
  /// Misalign using gaussian distribution
  /**
   misalign the centre of the local transformation
   rotation axes :
   fAngMisAlig[1,2,3] = [x,y,z]
   Assume that misalignment about the x and y axes (misalignment of z plane)
   is much smaller, since the entire detection plane has to be moved (the
   detection elements are on a support structure), while rotation of the x-y
   plane is more free.
   */
  cartMisAlig[0] = gRandom->Gaus(lParMisAlig[0][0], lParMisAlig[0][1]); //, 3. * lParMisAlig[0][1]);
  cartMisAlig[1] = gRandom->Gaus(lParMisAlig[1][0], lParMisAlig[1][1]); //, 3. * lParMisAlig[1][1]);
  cartMisAlig[2] = gRandom->Gaus(lParMisAlig[2][0], lParMisAlig[2][1]); //, 3. * lParMisAlig[2][1]);

  angMisAlig[0] = gRandom->Gaus(lParMisAlig[3][0], lParMisAlig[3][1]); //, 3. * lParMisAlig[3][1]);
  angMisAlig[1] = gRandom->Gaus(lParMisAlig[4][0], lParMisAlig[4][1]); //, 3. * lParMisAlig[4][1]);
  angMisAlig[2] = gRandom->Gaus(lParMisAlig[5][0], lParMisAlig[5][1]); //, 3. * lParMisAlig[5][1]); // degrees
}

//_________________________________________________________________________
TGeoCombiTrans MisAligner::misAlignDetElem() const
{
  /// Returns a local delta transformation for a detection element. This is
  /// meant to be a local delta transformation in the ALICE alignment
  /// framework known and will be used to create an AlignParam, which
  /// be applied to the geometry after misaligning the module
  /// (see MisAligner::misAlign)

  double cartMisAlig[3] = {0, 0, 0};
  double angMisAlig[3] = {0, 0, 0};

  if (mUseUni) {
    getUniMisAlign(cartMisAlig, angMisAlig, mDetElemMisAlig);
  } else {
    if (!mUseGaus) {
      LOG(warn) << Form("Neither uniform nor gausian distribution is set! Will use gausian...");
    }
    getGausMisAlign(cartMisAlig, angMisAlig, mDetElemMisAlig);
  }

  TGeoTranslation deltaTrans(cartMisAlig[0], cartMisAlig[1], cartMisAlig[2]);
  TGeoRotation deltaRot;
  deltaRot.RotateX(angMisAlig[0]);
  deltaRot.RotateY(angMisAlig[1]);
  deltaRot.RotateZ(angMisAlig[2]);

  TGeoCombiTrans deltaTransf(deltaTrans, deltaRot);

  LOG(info) << Form("Rotated DE by %f about Z axis.", angMisAlig[2]);

  return TGeoCombiTrans(deltaTransf);
}

//_________________________________________________________________________
TGeoCombiTrans MisAligner::misAlignModule() const
{
  /// Returns a local delta transformation for a half chamber
  /// (aka a module). This is meant to be a local delta transformation in
  /// the ALICE alignment framework known and will be used to create an
  /// AlignParam (see MisAligner::misAlign)

  double cartMisAlig[3] = {0, 0, 0};
  double angMisAlig[3] = {0, 0, 0};

  if (mUseUni) {
    getUniMisAlign(cartMisAlig, angMisAlig, mModuleMisAlig);
  } else {
    if (!mUseGaus) {
      LOG(warn) << Form("Neither uniform nor gausian distribution is set! Will use gausian...");
    }
    getGausMisAlign(cartMisAlig, angMisAlig, mModuleMisAlig);
  }

  TGeoTranslation deltaTrans(cartMisAlig[0], cartMisAlig[1], cartMisAlig[2]);
  TGeoRotation deltaRot;
  deltaRot.RotateX(angMisAlig[0]);
  deltaRot.RotateY(angMisAlig[1]);
  deltaRot.RotateZ(angMisAlig[2]);

  TGeoCombiTrans deltaTransf(deltaTrans, deltaRot);

  LOG(info) << Form("Rotated Module by %f about Z axis.", angMisAlig[2]);

  return TGeoCombiTrans(deltaTransf);
}

//______________________________________________________________________
// void MisAligner::MisAlign(Bool_t verbose) const
void MisAligner::misAlign(std::vector<o2::detectors::AlignParam>& params, Bool_t verbose) const
{
  /// Generates local delta transformations for the modules and their
  /// detection elements, creates AlignParams, and applies them to the
  /// current geometry.
  /// The AlignParams are stored in a std::vector.

  std::vector<std::vector<int>> DEofHC{{100, 103},
                                       {101, 102},
                                       {200, 203},
                                       {201, 202},
                                       {300, 303},
                                       {301, 302},
                                       {400, 403},
                                       {401, 402},
                                       {500, 501, 502, 503, 504, 514, 515, 516, 517},
                                       {505, 506, 507, 508, 509, 510, 511, 512, 513},
                                       {600, 601, 602, 603, 604, 614, 615, 616, 617},
                                       {605, 606, 607, 608, 609, 610, 611, 612, 613},
                                       {700, 701, 702, 703, 704, 705, 706, 720, 721, 722, 723, 724, 725},
                                       {707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719},
                                       {800, 801, 802, 803, 804, 805, 806, 820, 821, 822, 823, 824, 825},
                                       {807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819},
                                       {900, 901, 902, 903, 904, 905, 906, 920, 921, 922, 923, 924, 925},
                                       {907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919},
                                       {1000, 1001, 1002, 1003, 1004, 1005, 1006, 1020, 1021, 1022, 1023, 1024, 1025},
                                       {1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019}};

  o2::detectors::AlignParam lAP;
  for (int hc = 0; hc < 20; hc++) {

    TGeoCombiTrans localDeltaTransform = misAlignModule();
    // localDeltaTransform.Print();

    std::string sname = fmt::format("MCH/HC{}", hc);
    lAP.setSymName(sname.c_str());

    double lPsi, lTheta, lPhi = 0.;
    if (!isMatrixConvertedToAngles(localDeltaTransform.GetRotationMatrix(), lPsi, lTheta, lPhi)) {
      LOG(error) << "Problem extracting angles!";
    }

    LOG(debug) << fmt::format("Module {} is {} : {} : Local Delta | X: {:+f} Y: {:+f} Z: {:+f} | pitch: {:+f} roll: {:+f} yaw: {:+f}\n", hc, lAP.getSymName(), lAP.getAlignableID(), localDeltaTransform.GetTranslation()[0],
                              localDeltaTransform.GetTranslation()[1], localDeltaTransform.GetTranslation()[2], lPsi, lTheta, lPhi);
    if (!lAP.setLocalParams(localDeltaTransform)) {
      LOG(error) << "Could not set local params for " << sname.c_str();
    }
    LOG(debug) << fmt::format("Module {} is {} : {} : Global Delta | X: {:+f} Y: {:+f} Z: {:+f} | pitch: {:+f} roll: {:+f} yaw: {:+f}\n", hc, lAP.getSymName(), lAP.getAlignableID(), lAP.getX(),
                              lAP.getY(), lAP.getZ(), lAP.getPsi(), lAP.getTheta(), lAP.getPhi());
    // lAP.print();
    lAP.applyToGeometry();
    params.emplace_back(lAP);
    for (int de = 0; de < DEofHC[hc].size(); de++) {

      localDeltaTransform = misAlignDetElem();

      sname = fmt::format("MCH/HC{}/DE{}", hc, DEofHC[hc][de]);
      lAP.setSymName(sname.c_str());

      if (!isMatrixConvertedToAngles(localDeltaTransform.GetRotationMatrix(), lPsi, lTheta, lPhi)) {
        LOG(error) << "Problem extracting angles for " << sname.c_str();
      }
      LOG(debug) << fmt::format("DetElem {} is {} : {} : Local Delta| X: {:+f} Y: {:+f} Z: {:+f} | pitch: {:+f} roll: {:+f} yaw: {:+f}\n", de, lAP.getSymName(), lAP.getAlignableID(), localDeltaTransform.GetTranslation()[0],
                                localDeltaTransform.GetTranslation()[1], localDeltaTransform.GetTranslation()[2], lPsi, lTheta, lPhi);
      if (!lAP.setLocalParams(localDeltaTransform)) {
        LOG(error) << "  Could not set local params for " << sname.c_str();
      }
      LOG(debug) << fmt::format("DetElem {} is {} : {} : Global Delta | X: {:+f} Y: {:+f} Z: {:+f} | pitch: {:+f} roll: {:+f} yaw: {:+f}\n", de, lAP.getSymName(), lAP.getAlignableID(), lAP.getX(),
                                lAP.getY(), lAP.getZ(), lAP.getPsi(), lAP.getTheta(), lAP.getPhi());
      lAP.applyToGeometry();
      params.emplace_back(lAP);
    }

    if (verbose) {
      LOG(info) << "MisAligned half chamber " << hc;
    }
  }
}

void MisAligner::setAlignmentResolution(const TClonesArray* misAlignArray, int rChId, double rChResX, double rChResY, double rDeResX, double rDeResY)
{
  /// In AliRoot we could also store the alignment resolution in the alignment objects
  /// but we never used that.
}

} // namespace o2::mch::geo
