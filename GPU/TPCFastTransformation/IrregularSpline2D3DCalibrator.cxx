// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file IrregularSpline2D3DCalibrator.cxx
/// \brief Implementation of IrregularSpline2D3DCalibrator class
///
/// \author  Oscar Lange <langeoscar96@googlemail.com>
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#include "IrregularSpline2D3D.h"
#include "IrregularSpline2D3DCalibrator.h"
#include <cmath>
#include <iostream>

namespace GPUCA_NAMESPACE
{
namespace gpu
{

IrregularSpline2D3DCalibrator::IrregularSpline2D3DCalibrator()
{
  /// Default constructor.
  setRasterSize(5, 5);
  setMaxNKnots(5, 5);
}

void IrregularSpline2D3DCalibrator::setRasterSize(int nKnotsU, int nKnotsV)
{
  /// set maximal size of the spline grid

  int n[2] = {nKnotsU, nKnotsV};

  for (int uv = 0; uv < 2; ++uv) {
    if (n[uv] < mMaxNKnots[uv])
      n[uv] = mMaxNKnots[uv];
  }

  mRaster.constructRegular(n[0], n[1]);
}

void IrregularSpline2D3DCalibrator::setMaxNKnots(int nKnotsU, int nKnotsV)
{
  /// set maximal size of the spline grid

  mMaxNKnots[0] = nKnotsU;
  mMaxNKnots[1] = nKnotsV;

  for (int uv = 0; uv < 2; ++uv) {
    if (mMaxNKnots[uv] < 5)
      mMaxNKnots[uv] = 5;
  }
}

void IrregularSpline2D3DCalibrator::startCalibration(std::function<void(float, float, float&, float&, float&)> F)
{
  // initialize everything for the calibration

  // fill the raster data
  mRasterData.resize(mRaster.getNumberOfKnots() * 3);

  for (int i = 0; i < mRaster.getNumberOfKnots(); ++i) {
    float u = 0, v = 0, fx = 0, fy = 0, fz = 0;
    mRaster.getKnotUV(i, u, v);
    F(u, v, fx, fy, fz);
    mRasterData[3 * i + 0] = fx;
    mRasterData[3 * i + 1] = fy;
    mRasterData[3 * i + 2] = fz;
  }

  mRaster.correctEdges(mRasterData.data());

  // create current spline
  for (int uv = 0; uv < 2; ++uv) {
    //std::cout<<"n Raster knots: "<<mRaster.getGrid(uv).getNumberOfKnots()<<std::endl;
    mKnots[uv].clear();
    double du = 1. / (mMaxNKnots[uv] - 1);
    int lastKnot = 0;
    for (int i = 1; i < mMaxNKnots[uv] - 1; ++i) {
      KnotData d;
      d.uv = uv;
      double u = i * du;
      d.rasterKnot = nearbyint(u * (mRaster.getGrid(uv).getNumberOfKnots() - 1));
      //std::cout<<"uv "<<uv<<" i "<<d.rasterKnot<<" u "<<u<<std::endl;
      if (d.rasterKnot <= lastKnot)
        continue;
      if (d.rasterKnot >= mRaster.getGrid(uv).getNumberOfKnots() - 1)
        continue;
      mKnots[uv].push_back(d);
      lastKnot = d.rasterKnot;
    }
  }

  createCurrentSpline();
}

void IrregularSpline2D3DCalibrator::createCurrentSpline()
{
  createSpline(mSpline, mSplineData);
}

void IrregularSpline2D3DCalibrator::createActionSpline()
{
  createSpline(mActionSpline, mActionSplineData);
}

void IrregularSpline2D3DCalibrator::createSpline(IrregularSpline2D3D& sp, std::vector<float>& data)
{
  // recreate a spline with  respect to knots in mKnots[] lists

  for (int uv = 0; uv < 2; ++uv) {
    mTemp[uv].reserve(mMaxNKnots[uv]);
    mTemp[uv].clear();
    mTemp[uv].push_back(0.f);
    for (std::list<KnotData>::iterator i = mKnots[uv].begin(); i != mKnots[uv].end(); ++i) {
      //std::cout<<"uv "<<uv<<" i "<<i->rasterKnot<<" u "<<mRaster.getGrid(uv).getKnot(i->rasterKnot).u<<std::endl;
      mTemp[uv].push_back(mRaster.getGrid(uv).getKnot(i->rasterKnot).u);
    }
    mTemp[uv].push_back(1.f);
  }

  sp.construct(mTemp[0].size(), mTemp[0].data(), mRaster.getGrid(0).getNumberOfKnots(),
               mTemp[1].size(), mTemp[1].data(), mRaster.getGrid(1).getNumberOfKnots());

  data.resize(sp.getNumberOfKnots() * 3);
  for (int i = 0; i < sp.getNumberOfKnots(); ++i) {
    float u = 0, v = 0, fx = 0, fy = 0, fz = 0;
    sp.getKnotUV(i, u, v);
    mRaster.getSplineVec(mRasterData.data(), u, v, fx, fy, fz);
    data[3 * i + 0] = fx;
    data[3 * i + 1] = fy;
    data[3 * i + 2] = fz;
  }
  sp.correctEdges(data.data());
}

IrregularSpline2D3DCalibrator::Action IrregularSpline2D3DCalibrator::checkActionShift(std::list<KnotData>::iterator& knot)
{
  // estimate the cost change when the knot i is shifted up or down

  Action ret;
  ret.action = Action::Move::No;
  ret.cost = mMaxDeviation + 1.e10;
  ret.iter = knot;

  int uv = knot->uv;

  bool isSpaceUp = 0;

  if (knot->rasterKnot < mRaster.getGrid(uv).getNumberOfKnots() - 2) {
    isSpaceUp = 1;
    std::list<KnotData>::iterator next = knot;
    ++next;
    if (next != mKnots[uv].end()) {
      if (next->rasterKnot <= knot->rasterKnot + 1)
        isSpaceUp = 0;
    }
  }

  bool isSpaceDn = 0;

  if (knot->rasterKnot > 1) {
    isSpaceDn = 1;
    std::list<KnotData>::iterator prev = knot;
    if (prev != mKnots[uv].begin()) {
      --prev;
      if (prev->rasterKnot >= knot->rasterKnot - 1)
        isSpaceDn = 0;
    }
  }

  if (!isSpaceUp && !isSpaceDn)
    return ret;

  // get the area of interest

  int regionKnotFirst = knot->rasterKnot;
  int regionKnotLast = knot->rasterKnot;
  getRegionOfInfluence(knot, regionKnotFirst, regionKnotLast);

  // get the current cost

  double costCurrent = getIntegralDeviationArea(mSpline, mSplineData, uv, regionKnotFirst, regionKnotLast);

  // get the cost when moving up

  if (isSpaceUp) {
    knot->rasterKnot++;
    createActionSpline();
    knot->rasterKnot--;
    ret.action = Action::Move::Up;
    ret.cost = getIntegralDeviationArea(mActionSpline, mActionSplineData, uv, regionKnotFirst, regionKnotLast) - costCurrent;
  }

  if (isSpaceDn) {
    knot->rasterKnot--;
    createActionSpline();
    knot->rasterKnot++;
    double costDn = getIntegralDeviationArea(mActionSpline, mActionSplineData, uv, regionKnotFirst, regionKnotLast) - costCurrent;
    if (costDn < ret.cost) {
      ret.action = Action::Move::Down;
      ret.cost = costDn;
    }
  }
  //if( ret.cost<0 )  std::cout<<"knot "<<knot->rasterKnot<<" area: "<<regionKnotFirst<<"<->"<<regionKnotLast<<" costCurrent "<<costCurrent<<std::endl;

  return ret;
}

IrregularSpline2D3DCalibrator::Action IrregularSpline2D3DCalibrator::checkActionRemove(std::list<KnotData>::iterator& knot)
{
  // estimate the cost change when the knot i is shifted up or down

  Action ret;
  ret.action = Action::Move::No;
  ret.cost = mMaxDeviation + 1.e10;
  ret.iter = knot;

  int uv = knot->uv;

  if (mSpline.getGrid(uv).getNumberOfKnots() <= 5)
    return ret;

  // get the area of interest

  int regionKnotFirst = knot->rasterKnot;
  int regionKnotLast = knot->rasterKnot;

  getRegionOfInfluence(knot, regionKnotFirst, regionKnotLast);

  // std::cout<<"uv "<<uv<<" knot "<<knot->rasterKnot<<" region: "<<regionKnotFirst<<" <-> "<<regionKnotLast<<std::endl;

  KnotData tmp = *knot;
  std::list<KnotData>::iterator next = mKnots[uv].erase(knot);
  createActionSpline();
  knot = mKnots[uv].insert(next, tmp);

  // get the cost

  ret.action = Action::Move::Remove;
  ret.cost = getMaxDeviationArea(mActionSpline, mActionSplineData, uv, regionKnotFirst, regionKnotLast);
  ret.iter = knot;

  return ret;
}

void IrregularSpline2D3DCalibrator::getRegionOfInfluence(std::list<KnotData>::iterator knot, int& regionKnotFirst, int& regionKnotLast) const
{
  int uv = knot->uv;
  regionKnotFirst = knot->rasterKnot;
  regionKnotLast = knot->rasterKnot;
  std::list<KnotData>::iterator next = knot;
  std::list<KnotData>::iterator prev = knot;
  for (int i = 0; i < 3; ++i) {
    if (prev != mKnots[uv].begin()) {
      --prev;
      regionKnotFirst = prev->rasterKnot;
    } else {
      regionKnotFirst = 0;
    }

    if (next != mKnots[uv].end()) {
      ++next;
      if (next != mKnots[uv].end()) {
        regionKnotLast = next->rasterKnot;
      } else {
        regionKnotLast = mRaster.getGrid(uv).getNumberOfKnots() - 1;
      }
    }
  }
}

bool IrregularSpline2D3DCalibrator::doCalibrationStep()
{
  // perform one step of the calibration

  // first, try to move a knot
  //std::cout<<"do step.. "<<std::endl;
  Action bestAction;
  bestAction.action = Action::Move::No;
  bestAction.cost = 1.e10;

  for (int uv = 0; uv < 2; ++uv) {
    for (std::list<KnotData>::iterator i = mKnots[uv].begin(); i != mKnots[uv].end(); ++i) {
      Action a = checkActionShift(i);
      if (a.cost < bestAction.cost)
        bestAction = a;
    }
  }

  //std::cout<<"move cost: "<<bestAction.cost<<std::endl;
  if (bestAction.cost < 0) { // shift the best knot
    if (bestAction.action == Action::Move::Up) {
      //std::cout<<"move Up uv "<< bestAction.iter->uv<<" knot "<<bestAction.iter->rasterKnot<<" -> "<<bestAction.iter->rasterKnot+1<<std::endl;
      bestAction.iter->rasterKnot++;
    } else if (bestAction.action == Action::Move::Down) {
      //std::cout<<"move Down uv "<< bestAction.iter->uv<<" knot "<<bestAction.iter->rasterKnot<<" -> "<<bestAction.iter->rasterKnot-1<<std::endl;
      bestAction.iter->rasterKnot--;
    } else {
      std::cerr << "Internal error!!!" << std::endl;
      return 0;
    }
    createCurrentSpline();
    return 1;
  }

  // second, try to remove a knot

  //for (int axis = 0; axis < 2; axis++) {
  bestAction.action = Action::Move::No;
  bestAction.cost = mMaxDeviation + 1.e10;

  for (int uv = 0; uv < 2; ++uv) {

    for (std::list<KnotData>::iterator i = mKnots[uv].begin(); i != mKnots[uv].end(); ++i) {
      Action a = checkActionRemove(i);
      if (a.cost < bestAction.cost)
        bestAction = a;
    }
  }
  bestAction.cost = sqrt(bestAction.cost / 3.);

  //std::cout<<"remove cost: "<<bestAction.cost<<std::endl;

  if (bestAction.cost <= mMaxDeviation) { // move the best knot
    if (bestAction.action == Action::Move::Remove) {
      //std::cout<<"remove uv "<< bestAction.iter->uv<<" knot "<<bestAction.iter->rasterKnot<<std::endl;
      mKnots[bestAction.iter->uv].erase(bestAction.iter);
    } else {
      std::cout << "Internal error!!!" << std::endl;
      return 0;
    }
    createCurrentSpline();
    return 1;
  }

  return 0;
}

std::unique_ptr<float[]> IrregularSpline2D3DCalibrator::calibrateSpline(IrregularSpline2D3D& spline_uv,
                                                                        std::function<void(float, float, float&, float&, float&)> F)
{
  // main method: spline calibration

  startCalibration(F);
  while (doCalibrationStep())
    ;
  createCurrentSpline();
  spline_uv.cloneFromObject(mSpline, nullptr);
  std::unique_ptr<float[]> tmp(new float[mSpline.getNumberOfKnots()]);
  for (int i = 0; i < mSpline.getNumberOfKnots(); ++i) {
    tmp[i] = mSplineData[i];
  }
  return tmp;
}

double IrregularSpline2D3DCalibrator::getMaxDeviationLine(const IrregularSpline2D3D& spline, const std::vector<float>& data, int axis0, int knot0) const
{
  int axis1 = (axis0 == 0) ? 1 : 0;
  float u[2];
  u[axis0] = mRaster.getGrid(axis0).getKnot(knot0).u;

  double dMax2 = 0.;

  for (int knot1 = 0; knot1 < mRaster.getGrid(axis1).getNumberOfKnots(); ++knot1) {
    u[axis1] = mRaster.getGrid(axis1).getKnot(knot1).u;
    float fx0, fy0, fz0, fx, fy, fz;
    mRaster.getSplineVec(mRasterData.data(), u[0], u[1], fx0, fy0, fz0);
    spline.getSplineVec(data.data(), u[0], u[1], fx, fy, fz);
    double dx = fx - fx0;
    double dy = fy - fy0;
    double dz = fz - fz0;
    double d2 = dx * dx + dy * dy + dz * dz;
    if (dMax2 < d2)
      dMax2 = d2;
  }
  return dMax2;
}

double IrregularSpline2D3DCalibrator::getMaxDeviationArea(const IrregularSpline2D3D& spline, const std::vector<float>& data,
                                                          int axis, int knotFirst, int knotLast) const
{
  double dMax = 0.;
  for (int knot = knotFirst; knot <= knotLast; ++knot) {
    double d = getMaxDeviationLine(spline, data, axis, knot);
    if (dMax < d)
      dMax = d;
  }
  return dMax;
}

double IrregularSpline2D3DCalibrator::getIntegralDeviationLine(const IrregularSpline2D3D& spline, const std::vector<float>& data, int axis0, int knot0) const
{
  int axis1 = (axis0 == 0) ? 1 : 0;
  float u[2];
  u[axis0] = mRaster.getGrid(axis0).getKnot(knot0).u;

  double sum = 0.;

  for (int knot1 = 0; knot1 < mRaster.getGrid(axis1).getNumberOfKnots(); ++knot1) {
    u[axis1] = mRaster.getGrid(axis1).getKnot(knot1).u;
    float fx0, fy0, fz0, fx, fy, fz;
    mRaster.getSplineVec(mRasterData.data(), u[0], u[1], fx0, fy0, fz0);
    spline.getSplineVec(data.data(), u[0], u[1], fx, fy, fz);
    double dx = fx - fx0;
    double dy = fy - fy0;
    double dz = fz - fz0;
    double d2 = dx * dx + dy * dy + dz * dz;
    sum += sqrt(d2 / 3.);
  }
  //sum = sqrt(sum/3.);
  return sum;
}

double IrregularSpline2D3DCalibrator::getIntegralDeviationArea(const IrregularSpline2D3D& spline, const std::vector<float>& data,
                                                               int axis, int knotFirst, int knotLast) const
{
  double sum = 0.;
  for (int knot = knotFirst; knot <= knotLast; ++knot) {
    sum += getIntegralDeviationLine(spline, data, axis, knot);
  }
  return sum;
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE
