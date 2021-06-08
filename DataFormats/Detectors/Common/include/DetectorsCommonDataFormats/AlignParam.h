// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AlignParam.h
/// \brief Definition of the base alignment parameters class
/// @author Rafaelle Grosso, Raffaele.Grosso@cern.ch (original code in AliRoot)
///         Ruben Shahoyan, ruben.shahoyan@cern.ch (porting to O2)

#ifndef ALICEO2_BASE_ALIGNPARAM_H_
#define ALICEO2_BASE_ALIGNPARAM_H_

class TGeoMatrix;
class TGeoHMatrix;

#include <string>

namespace o2
{
namespace detectors
{
/// Base class for alignment parameters, containing global delta of rotation and translation
/// For the detail of alignment framework check http://alice-offline.web.cern.ch/Activities/Alignment

class AlignParam
{
 public:
  AlignParam() = default;
  ~AlignParam() = default;
  AlignParam(const char* symname, int algID,       // volume symbolic name and its alignable ID
             double x, double y, double z,         // delta translation
             double psi, double theta, double phi, // delta rotation
             bool global = true);                  // global (preferable) or local delta definition

  /// return symbolic name of the volume
  const std::string& getSymName() const { return mSymName; }
  /// iparamater's getters
  double getPhi() const { return mPhi; }
  double getPsi() const { return mPsi; }
  double getTheta() const { return mTheta; }
  double getX() const { return mX; }
  double getY() const { return mY; }
  double getZ() const { return mZ; }

  /// apply object to geoemetry with optional check for the overlaps
  bool applyToGeometry(bool ovlpcheck = false, double ovlToler = 1e-3) const;

  /// extract global delta matrix
  TGeoHMatrix createMatrix() const;

  /// extract local delta matrix
  bool createLocalMatrix(TGeoHMatrix& m) const;

  /// set symbolic name of the volume
  void setSymName(const char* m) { mSymName = m; }

  /// return alignable entry ID
  int getAlignableID() const { return mAlignableID; }

  /// set alignable entry ID
  void setAlignableID(int id) { mAlignableID = id; }
  /// ================ methods for direct setting of delta params

  /// set parameters of global delta
  void setGlobalParams(double x, double y, double z, double psi, double theta, double phi);

  /// set global delta rotations angles in radian
  void setRotation(double psi, double theta, double phi);

  /// set global delta displacements in cm
  void setTranslation(double x, double y, double z);

  /// set params from the matrix of global delta
  void setGlobalParams(const TGeoMatrix& m);

  /// set translation from the matrix of global delta
  void setTranslation(const TGeoMatrix& m);

  // set rotation from the matrix of global delta
  bool setRotation(const TGeoMatrix& m);

  /// ================ methods for setting global delta params from local delta

  /// set global delta params from the local delta params
  bool setLocalParams(double x, double y, double z, double psi, double theta, double phi);

  /// set global delta translation from the local delta translation
  bool setLocalTranslation(double x, double y, double z);

  /// set global delta rotation from the local delta rotation
  bool setLocalRotation(double psi, double theta, double phi);

  /// set global delta params from the local delta matrix
  bool setLocalParams(const TGeoMatrix& m);

  /// set the global delta transformation from translation part of local delta matrix
  bool setLocalTranslation(const TGeoMatrix& m);

  /// set the global delta rotation from rotation part of local delta matrix
  bool setLocalRotation(const TGeoMatrix& m);

  int getLevel() const;

  void print() const;

 protected:
  bool matrixToAngles(const double* rot, double& psi, double& theta, double& phi) const;
  void anglesToMatrix(double psi, double theta, double phi, double* rot) const;
  void setMatrixRotation(double psi, double theta, double phi, TGeoHMatrix& dest) const;
  void setMatrixTranslation(double x, double y, double z, TGeoHMatrix& dest) const;

 private:
  std::string mSymName{};

  int mAlignableID = -1; /// alignable ID (set for sensors only)

  double mX = 0.; ///< X translation of global delta
  double mY = 0.; ///< Y translation of global delta
  double mZ = 0.; ///< Z translation of global delta

  double mPsi = 0.;   ///< "pitch" : Euler angle of rotation around final X axis (radians)
  double mTheta = 0.; ///< "roll"  : Euler angle of rotation around Y axis after 1st rotation (radians)
  double mPhi = 0.;   ///< "yaw"   : Euler angle of rotation around Z axis (radians)

  ClassDefNV(AlignParam, 1);
};

}
}

#endif
