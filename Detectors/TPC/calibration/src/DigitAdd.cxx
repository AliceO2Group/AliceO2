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

#include "GPUTPCGeometry.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/CRU.h"
#include "TPCCalibration/DigitAdd.h"

using namespace o2::tpc;

int DigitAdd::sector() const
{
  return CRU(mCRU).sector();
}

float DigitAdd::lx() const
{
  const GPUCA_NAMESPACE::gpu::GPUTPCGeometry gpuGeom;
  return gpuGeom.Row2X(mRow);
}

float DigitAdd::ly() const
{
  const GPUCA_NAMESPACE::gpu::GPUTPCGeometry gpuGeom;
  return gpuGeom.LinearPad2Y(sector(), mRow, getPad());
}

float DigitAdd::gx() const
{
  const LocalPosition2D l2D{lx(), ly()};
  const auto g2D = Mapper::LocalToGlobal(l2D, Sector(sector()));
  return g2D.x();
}

float DigitAdd::gy() const
{
  const LocalPosition2D l2D{lx(), ly()};
  const auto g2D = Mapper::LocalToGlobal(l2D, Sector(sector()));
  return g2D.y();
}

float DigitAdd::cpad() const
{
  const GPUCA_NAMESPACE::gpu::GPUTPCGeometry gpuGeom;
  return getPad() - gpuGeom.NPads(mRow) / 2.f;
}
