// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MathUtils/Cartesian3D.h"

using namespace o2;

ClassImp(o2::Transform3D);
ClassImp(o2::Rotation2D);

//_________________________________________________
Transform3D::Transform3D(const TGeoMatrix& m)
{
  /*
   * Construct from TGeoMatrix or its derived classes
   */
  set(m);
}

//_________________________________________________
void Transform3D::set(const TGeoMatrix& m)
{
  /*
   * Set parameters from TGeoMatrix
   */
  const double* t = m.GetTranslation();
  const double* r = m.GetRotationMatrix();
  SetComponents(r[0], r[1], r[2], t[0], r[3], r[4], r[5], t[1], r[6], r[7], r[8], t[2]);
}

//_________________________________________________
void Transform3D::print() const
{
  /*
   * print itself
   */
  std::cout << *this << std::endl;
}

//_________________________________________________
std::ostream &operator<<(std::ostream &os, const o2::Rotation2D &t)
{
  float cs,sn;
  t.getComponents(cs,sn);
  os << "cos: " << cs << " sin: " << sn ;
  return os;
}
