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

/// \file Cluster.h
/// \brief Definition of the TOF cluster

#ifndef ALICEO2_TOF_CLUSTERCALINFOCLASS_H
#define ALICEO2_TOF_CLUSTERCALINFOCLASS_H

#include <vector>
#include "Rtypes.h"

namespace o2
{
namespace tof
{
/// \class CalibInfoCluster
/// \brief CalibInfoCluster for TOF
///
class CalibInfoCluster
{
  int ch = 0;
  int8_t deltach = 0;
  float deltat = 0;
  short tot1 = 0;
  short tot2 = 0;

 public:
  int getCH() const { return ch; }
  int8_t getDCH() const { return deltach; }
  float getDT() const { return deltat; }
  short getTOT1() const { return tot1; }
  short getTOT2() const { return tot2; }

  CalibInfoCluster() = default;
  CalibInfoCluster(int ich, int8_t ideltach, float dt, short t1, short t2) : ch(ich), deltach(ideltach), deltat(dt), tot1(t1), tot2(t2) {}
  ClassDefNV(CalibInfoCluster, 1);
};
} // namespace tof

} // namespace o2

#endif
