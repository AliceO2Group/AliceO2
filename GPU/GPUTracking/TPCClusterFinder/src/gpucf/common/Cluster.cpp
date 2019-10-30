// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Cluster.h"

#include <gpucf/common/float.h>
#include <gpucf/common/RowInfo.h>

#include <ostream>

using namespace gpucf;

static_assert(Cluster::Field_all == 0b00111111, "");

Cluster::Cluster()
  : Cluster(0, 0, 0, 0, 0, 0, 0, 0)
{
}

Cluster::Cluster(int _cru, int _row, float q, float qmax, float _padMean,
                 float _timeMean, float _padSigma, float _timeSigma)
{
  cru = _cru;
  row = _row;
  Q = q;
  QMax = qmax;
  padMean = _padMean;
  timeMean = _timeMean;
  padSigma = _padSigma;
  timeSigma = _timeSigma;
}

Cluster::Cluster(const Digit& peak, const ClusterNative& cn)
  : Cluster(peak.cru(), peak.localRow(), cn)
{
}

Cluster::Cluster(int globalRow, const ClusterNative& cn)
  : Cluster(
      RowInfo::instance().globalRowToCru(globalRow),
      RowInfo::instance().globalToLocal(globalRow),
      cn)
{
}

Cluster::Cluster(int cru, int row, const ClusterNative& cn)
  : Cluster(
      cru,
      row,
      cn.qtot,
      cn.qmax,
      cnGetPad(&cn),
      cnGetTime(&cn),
      cnGetSigmaPad(&cn),
      cnGetSigmaTime(&cn))
{
  unsigned char flags = cnGetFlags(&cn);

  atEdge = flags & CN_FLAG_IS_EDGE_CLUSTER;
  splitInTime = flags & CN_FLAG_SPLIT_IN_TIME;
  splitInPad = flags & CN_FLAG_SPLIT_IN_PAD;
}

Object Cluster::serialize() const
{
  Object obj("Cluster");

  SET_FIELD(obj, cru);
  SET_FIELD(obj, row);
  SET_FIELD(obj, Q);
  SET_FIELD(obj, QMax);
  SET_FIELD(obj, padMean);
  SET_FIELD(obj, timeMean);
  SET_FIELD(obj, padSigma);
  SET_FIELD(obj, timeSigma);

  return obj;
}

void Cluster::deserialize(const Object& obj)
{
  GET_INT(obj, cru);
  GET_INT(obj, row);
  GET_FLOAT(obj, Q);
  GET_FLOAT(obj, QMax);
  GET_FLOAT(obj, padMean);
  GET_FLOAT(obj, timeMean);
  GET_FLOAT(obj, padSigma);
  GET_FLOAT(obj, timeSigma);
}

bool Cluster::hasNaN() const
{
  return std::isnan(cru) || std::isnan(row) || std::isnan(Q) || std::isnan(QMax) || std::isnan(padMean) || std::isnan(timeMean) || std::isnan(padSigma) || std::isnan(timeSigma);
}

int Cluster::globalRow() const
{
  return RowInfo::instance().localToGlobal(cru, row);
}

bool Cluster::hasNegativeEntries() const
{
  return cru < 0 || row < 0 || Q < 0 || QMax < 0 || padMean < 0 || timeMean < 0 || padSigma < 0 || timeSigma < 0;
}

bool Cluster::operator==(const Cluster& other) const
{
  return eq(other, FEQ_EPSILON_SMALL, FEQ_EPSILON_BIG, Field_all);
}

bool Cluster::eq(
  const Cluster& other,
  float /*epsilonSmall*/,
  float /*epsilonBig*/,
  FieldMask mask) const
{
  return cru == other.cru && row == other.row && (almostEqual(Q, other.Q) || !(mask & Field_Q)) && (almostEqual(QMax, other.QMax) || !(mask & Field_QMax)) && (almostEqual(timeMean, other.timeMean) || !(mask & Field_timeMean)) && (almostEqual(padMean, other.padMean) || !(mask & Field_padMean)) && (almostEqual(timeSigma, other.timeSigma) || !(mask & Field_timeSigma)) && (almostEqual(padSigma, other.padSigma) || !(mask & Field_padSigma));
}

float Cluster::dist(const Cluster& other) const
{
  float dt = timeMean - other.timeMean;
  float dp = padMean - other.padMean;
  return std::sqrt(dp * dp + dt * dt);
}

std::ostream& gpucf::operator<<(std::ostream& os, const Cluster& c)
{
  return os << c.serialize();
}

// vim: set ts=4 sw=4 sts=4 expandtab:
