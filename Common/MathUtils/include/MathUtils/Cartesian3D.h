// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @brief O2 interface class to ROOT::Math::Transform3D for Cartesian transformations
/// @author Sandro Wenzel, sandro.wenzel@cern.ch
/// @author Ruben Shahoyan, ruben.shahoyan@cern.ch

#ifndef ALICEO2_CARTESIAN3D_H
#define ALICEO2_CARTESIAN3D_H

#include <Math/GenVector/DisplacementVector3D.h>
#include <Math/GenVector/PositionVector3D.h>
#include <Math/GenVector/Rotation3D.h>
#include <Math/GenVector/Transform3D.h>
#include <Math/GenVector/Translation3D.h>
#include <Rtypes.h>
#include <TGeoMatrix.h>
#include <iosfwd>
#include "MathUtils/Cartesian2D.h"

template <typename T>
using Point3D = ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<T>, ROOT::Math::DefaultCoordinateSystemTag>;
template <typename T>
using Vector3D = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<T>, ROOT::Math::DefaultCoordinateSystemTag>;
// more typedefs can follow

namespace o2
{
/// predefined transformations: Tracking->Local, Tracking->Global, Local->Global etc
/// The IDs must be < 32

struct TransformType {
  static constexpr int L2G = 0;
  static constexpr int T2L = 1;
  static constexpr int T2G = 2;
  static constexpr int T2GRot = 3;
}; /// transformation types

class Rotation2D
{
  //
  // class to perform rotation of 3D (around Z) and 2D points

 public:
  Rotation2D() = default;
  Rotation2D(float cs, float sn) : mCos(cs), mSin(sn) {}
  Rotation2D(float phiZ) : mCos(cos(phiZ)), mSin(sin(phiZ)) {}
  ~Rotation2D() = default;
  Rotation2D(const Rotation2D& src) = default;
  Rotation2D& operator=(const Rotation2D& src) = default;

  void set(float phiZ)
  {
    mCos = cos(phiZ);
    mSin = sin(phiZ);
  }

  void set(float cs, float sn)
  {
    mCos = cs;
    mSin = sn;
  }

  void getComponents(float& cs, float& sn) const
  {
    cs = mCos;
    sn = mSin;
  }

  template <typename T>
  Point3D<T> operator()(const Point3D<T>& v) const
  { // local->master
    return Point3D<T>(v.X() * mCos - v.Y() * mSin, v.X() * mSin + v.Y() * mCos, v.Z());
  }

  template <typename T>
  Point3D<T> operator^(const Point3D<T>& v) const
  { // master->local
    return Point3D<T>(v.X() * mCos + v.Y() * mSin, -v.X() * mSin + v.Y() * mCos, v.Z());
  }

  template <typename T>
  Vector3D<T> operator()(const Vector3D<T>& v) const
  { // local->master
    return Vector3D<T>(v.X() * mCos - v.Y() * mSin, v.X() * mSin + v.Y() * mCos, v.Z());
  }

  template <typename T>
  Vector3D<T> operator^(const Vector3D<T>& v) const
  { // master->local
    return Vector3D<T>(v.X() * mCos + v.Y() * mSin, -v.X() * mSin + v.Y() * mCos, v.Z());
  }

  template <typename T>
  Point2D<T> operator()(const Point2D<T>& v) const
  { // local->master
    return Point2D<T>(v.X() * mCos - v.Y() * mSin, v.X() * mSin + v.Y() * mCos);
  }

  template <typename T>
  Point2D<T> operator^(const Point2D<T>& v) const
  { // master->local
    return Point2D<T>(v.X() * mCos + v.Y() * mSin, -v.X() * mSin + v.Y() * mCos);
  }

  template <typename T>
  Vector2D<T> operator()(const Vector2D<T>& v) const
  { // local->master
    return Vector2D<T>(v.X() * mCos - v.Y() * mSin, v.X() * mSin + v.Y() * mCos);
  }

  template <typename T>
  Vector2D<T> operator^(const Vector2D<T>& v) const
  { // master->local
    return Vector2D<T>(v.X() * mCos + v.Y() * mSin, -v.X() * mSin + v.Y() * mCos);
  }

 private:
  float mCos = 1.f; ///< cos of rotation angle
  float mSin = 0.f; ///< sin of rotation angle

  ClassDefNV(Rotation2D, 1);
};

class Transform3D : public ROOT::Math::Transform3D
{
  //
  // Class to perform geom.transformations (rotation and displacements only) in
  // double precision over the cartesian points and vectors (float or double).
  // Adds to the base ROOT::Math::Transform3D<double> class a convertor from
  // TGeoMatrix.
  // To be used instead of TGeoHMatrix for all transformations of hits,
  // clusters etc.
  //

 public:
  Transform3D() = default;
  Transform3D(const TGeoMatrix& m);
  ~Transform3D() = default;

  // inherit assignment operators of the base class
  using ROOT::Math::Transform3D::operator=;

  // to avoid conflict between the base Transform3D(const ForeignMatrix & m) and
  // Transform3D(const TGeoMatrix &m) constructors we cannot inherit base c-tors,
  // therefore we redefine them here
  Transform3D(const ROOT::Math::Rotation3D& r, const Vector& v) : ROOT::Math::Transform3D(r, v) {}
  Transform3D(const ROOT::Math::Rotation3D& r, const ROOT::Math::Translation3D& t) : ROOT::Math::Transform3D(r, t) {}
  template <class IT>
  Transform3D(IT begin, IT end) : ROOT::Math::Transform3D(begin, end)
  {
  }

  // conversion operator to TGeoHMatrix
  operator TGeoHMatrix&() const
  {
    static TGeoHMatrix tmp;
    double rot[9], tra[3];
    GetComponents(rot[0], rot[1], rot[2], tra[0], rot[3], rot[4], rot[5], tra[1], rot[6], rot[7], rot[8], tra[2]);
    tmp.SetRotation(rot);
    tmp.SetTranslation(tra);
    return tmp;
  }

  void set(const TGeoMatrix& m); // set parameters from TGeoMatrix

  using ROOT::Math::Transform3D::operator();
  // the local->master transformation for points and vectors can be
  // done in operator form (inherited from base Transform3D) as
  // Point3D pnt;
  // Transform3D trans;
  // auto pntTr0 = trans(pnt); // 1st version
  // auto pntTr1 = trans*pnt;  // 2nd version
  //
  // For the inverse transform we define our own operator^

  template <typename T>
  Point3D<T> operator^(const Point3D<T>& p) const
  { // master->local
    return ApplyInverse(p);
  }

  template <typename T>
  Vector3D<T> operator^(const Vector3D<T>& v) const
  { // local->master
    return ApplyInverse(v);
  }

  // TGeoHMatrix-like aliases
  template <typename T>
  void LocalToMaster(const Point3D<T>& loc, Point3D<T>& mst) const
  {
    mst = operator()(loc);
  }

  template <typename T>
  void MasterToLocal(const Point3D<T>& mst, Point3D<T>& loc) const
  {
    loc = operator^(mst);
  }

  template <typename T>
  void LocalToMasterVect(const Point3D<T>& loc, Point3D<T>& mst) const
  {
    mst = operator()(loc);
  }

  template <typename T>
  void MasterToLocalVect(const Point3D<T>& mst, Point3D<T>& loc) const
  {
    loc = operator^(mst);
  }

  void print() const;

  ClassDefNV(Transform3D, 1);
};
} // namespace o2

std::ostream& operator<<(std::ostream& os, const o2::Rotation2D& t);

#endif
