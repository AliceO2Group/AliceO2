// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <cfloat>
#include <cmath>
#include <iostream>
#include "DataFormatsEMCAL/Cluster.h"

using namespace o2::emcal;

Cluster::Cluster(Point3D<Float_t> pos, Float_t time, Float_t energy) : o2::dataformats::TimeStamp<Float16_t>(time),
                                                                       Point3D<Float16_t>(pos),
                                                                       mEnergy(energy),
                                                                       mDispersion(0),
                                                                       mDistanceToBadCell(1000),
                                                                       mM02(0),
                                                                       mM20(0),
                                                                       mIsExotic(false),
                                                                       mCellIndices()
{
}

Cluster::Cluster(Float_t x, Float_t y, Float_t z, Float_t time, Float_t energy) : o2::dataformats::TimeStamp<Float16_t>(time),
                                                                                  Point3D<Float16_t>(x, y, z),
                                                                                  mEnergy(energy),
                                                                                  mDispersion(0),
                                                                                  mDistanceToBadCell(1000),
                                                                                  mM02(0),
                                                                                  mM20(0),
                                                                                  mIsExotic(false),
                                                                                  mCellIndices()
{
}

Vector3D<Float_t> Cluster::getMomentum(const Point3D<Float_t>* vertex) const
{
  std::array<Float_t, 3> pos = {{this->X(), this->Y(), this->Z()}};

  if (vertex) { //calculate direction from vertex
    pos[0] -= vertex->X();
    pos[1] -= vertex->Y();
    pos[2] -= vertex->Z();
  }

  Float_t r = sqrt(pos[0] * pos[0] +
                   pos[1] * pos[1] +
                   pos[2] * pos[2]);

  if (std::abs(r) < DBL_EPSILON)
    throw InvalidPositionException(pos);
  return {mEnergy * pos[0] / r, mEnergy * pos[1] / r, mEnergy * pos[2] / r};
}

void Cluster::PrintStream(std::ostream& stream) const
{
  stream << "Pos: (" << X() << ", " << Y() << ", " << Z() << "), time: " << getTimeStamp() << ", energy " << mEnergy << " GeV" << std::endl;
  stream << "M02 " << mM02 << ", M20 " << mM20 << ", exotic: " << (mIsExotic ? "yes" : "no") << std::endl;
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const o2::emcal::Cluster& cluster)
{
  cluster.PrintStream(stream);
  return stream;
}