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

#include <TGeoMatrix.h>
#include <TNamed.h>

namespace o2
{
namespace detectors
{
/// Base class for alignment parameters, containing global delta of rotation and translation
/// For the detail of alignment framework check http://alice-offline.web.cern.ch/Activities/Alignment
/// Since some detectors may foresee to derive from this class to implement more complex alignment
/// objects (e.g. staves sagging for ITS), we define this class virtual and it should be stored in
/// the polymorphic container
class AlignParam : public TNamed
{
 public:
  AlignParam() = default;
  ~AlignParam() override = default;
  AlignParam(const char* symname, int algID,       // volume symbolic name and its alignable ID
             double x, double y, double z,         // delta translation
             double psi, double theta, double phi, // delta rotation
             bool global);                         // global (preferable) or local delta definition

  /// return symbolic name of the volume
  const char* getSymName() const { return GetName(); }
  /// paramater's getters
  double getPhi() const { return mPhi; }
  double getPsi() const { return mPsi; }
  double getTheta() const { return mTheta; }
  double getX() const { return mX; }
  double getY() const { return mY; }
  double getZ() const { return mZ; }
  /// apply object to geoemetry with optional check for the overlaps
  bool applyToGeometry(bool ovlpcheck = false, double ovlToler = 1e-3);

  /// extract global delta matrix
  virtual TGeoHMatrix createMatrix() const;

  /// extract local delta matrix
  virtual bool createLocalMatrix(TGeoHMatrix& m) const;

  /// set symbolic name of the volume
  void setSymName(const char* m) { return SetName(m); }
  /// return alignable entry ID
  int getAlignableID() const { return mAlignableID; }
  /// set alignable entry ID
  void setAlignableID(int id) { mAlignableID = id; }
  /// ================ methods for direct setting of global delta params

  /// set parameters of global delta
  void setParams(double x, double y, double z, double psi, double theta, double phi);

  /// set global delta rotations angles in radian
  void setRotation(double psi, double theta, double phi);

  /// set global delta displacements in cm
  void setTranslation(double x, double y, double z);

  /// set params from the matrix of global delta
  void setParams(const TGeoMatrix& m);

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

  /// method for sorting according to affected object depth
  bool IsSortable() const override { return true; }
  int Compare(const TObject* obj) const override;
  int getLevel() const;

  void Print(const Option_t* opt = "") const override;

 protected:
  bool matrixToAngles(const double* rot, double& psi, double& theta, double& phi);
  void setMatrixRotation(double psi, double theta, double phi, TGeoHMatrix& dest) const;
  void setMatrixTranslation(double x, double y, double z, TGeoHMatrix& dest) const;

 private:
  int mAlignableID = -1; /// alignable ID (set for sensors only)

  double mX = 0.; ///< X translation of global delta
  double mY = 0.; ///< Y translation of global delta
  double mZ = 0.; ///< Z translation of global delta

  double mPsi = 0.;   ///< "pitch" : Euler angle of rotation around final X axis (radians)
  double mTheta = 0.; ///< "roll"  : Euler angle of rotation around Y axis after 1st rotation (radians)
  double mPhi = 0.;   ///< "yaw"   : Euler angle of rotation around Z axis (radians)

  ClassDefOverride(AlignParam, 1);
};

//_____________________________________________________________________________
inline void AlignParam::setParams(double x, double y, double z, double psi, double theta, double phi)
{
  /// set parameters of global delta
  setTranslation(x, y, z);
  setRotation(psi, theta, phi);
}

//_____________________________________________________________________________
inline void AlignParam::setRotation(double psi, double theta, double phi)
{
  /// set global delta rotations angles in radian
  mPsi = psi;
  mTheta = theta;
  mPhi = phi;
}

//_____________________________________________________________________________
inline void AlignParam::setTranslation(double x, double y, double z)
{
  /// set global delta displacements in cm
  mX = x;
  mY = y;
  mZ = z;
}

//_____________________________________________________________________________
inline void AlignParam::setParams(const TGeoMatrix& m)
{
  /// set params from the matrix of global delta
  setTranslation(m);
  setRotation(m);
}

//___________________________________________________
inline void AlignParam::setMatrixTranslation(double x, double y, double z, TGeoHMatrix& dest) const
{
  /// apply translation to matrix
  double tra[3] = {x, y, z};
  dest.SetTranslation(tra);
}

//_____________________________________________________________________________
inline bool AlignParam::setLocalTranslation(double x, double y, double z)
{
  /// Set the global delta transformation by passing the three shifts giving
  /// the translation in the local reference system of the alignable
  /// volume (known by TGeo geometry).
  /// In case that the TGeo was not initialized or not closed,
  /// returns false and the object parameters are not set.

  TGeoHMatrix m;
  Double_t tr[3] = {x, y, z};
  m.SetTranslation(tr);

  return setLocalParams(m);
}

//_____________________________________________________________________________
inline bool AlignParam::setLocalTranslation(const TGeoMatrix& m)
{
  /// Set the global delta transformation by passing the matrix of
  /// the local delta transformation and taking its translational part
  /// In case that the TGeo was not initialized or not closed,
  /// returns false and the object parameters are not set.

  TGeoHMatrix mtr;
  mtr.SetTranslation(m.GetTranslation());
  return setLocalParams(mtr);
}

//_____________________________________________________________________________
inline bool AlignParam::setLocalRotation(double psi, double theta, double phi)
{
  /// Set the global delta transformation by passing the three angles giving
  /// the rotation in the local reference system of the alignable
  /// volume (known by TGeo geometry).
  /// In case that the TGeo was not initialized or not closed,
  /// returns false and the object parameters are not set.

  TGeoHMatrix m;
  setMatrixRotation(psi, theta, phi, m);
  return setLocalParams(m);
}

//_____________________________________________________________________________
inline bool AlignParam::setLocalRotation(const TGeoMatrix& m)
{
  /// Set the global delta transformation by passing the matrix of
  /// the local delta transformation and taking its rotational part
  /// In case that the TGeo was not initialized or not closed,
  /// returns false and the object parameters are not set.

  TGeoHMatrix rotm;
  rotm.SetRotation(m.GetRotationMatrix());
  return setLocalParams(rotm);
}
} // namespace detectors
} // namespace o2

#endif
