// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  IrregularSpline2D3DCalibrator.h
/// \brief Definition of IrregularSpline2D3DCalibrator class
///
/// \author  Oscar Lange
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_IRREGULARSPLINE2D3DCALIBRATOR_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_IRREGULARSPLINE2D3DCALIBRATOR_H

#include "GPUCommonDef.h"
#include "IrregularSpline2D3D.h"
#include <memory>
#include <list>
#include <functional>

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class IrregularSpline2D3DCalibrator
{
 public:
  struct KnotData {
    int uv;         // is the knot on U or V coordinate axis
    int rasterKnot; // index of the raster knot
  };

  struct Action {
    enum Move { No,
                Remove,
                Up,
                Down };
    Move action;                        // action type
    float cost;                         // deviation between the input function and the spline, which happens when applying this action
    std::list<KnotData>::iterator iter; // pointer to the knot data
    bool operator<(const Action& a)
    {
      return (cost < a.cost);
    }
  };

  /// _____________  Constructors / destructors __________________________

  /// Default constructor.
  IrregularSpline2D3DCalibrator();

  /// Destructor
  ~IrregularSpline2D3DCalibrator() CON_DEFAULT;

  /// set size of the raster grid
  void setRasterSize(int nKnotsU, int nKnotsV);

  /// set maximal size of the spline grid
  void setMaxNKnots(int nKnotsU, int nKnotsV);

  /// set maximal tolerated deviation between the spline and the input function
  void setMaximalDeviation(float maxDeviation)
  {
    mMaxDeviation = maxDeviation;
  }

  /// function constructs a calibrated Spline
  /// \param spline_uv - calibrated spline
  /// \param         F - input function

  std::unique_ptr<float[]> calibrateSpline(IrregularSpline2D3D& spline_uv, std::function<void(float, float, float&, float&, float&)> F);

  // some getters and step-by-step calibration methods. Only for debugging.

  const IrregularSpline2D3D& getRaster() const
  {
    return mRaster;
  }
  const float* getRasterData() const
  {
    return mRasterData.data();
  }

  const IrregularSpline2D3D& getSpline() const
  {
    return mSpline;
  }

  const float* getSplineData() const
  {
    return mSplineData.data();
  }

  void startCalibration(std::function<void(float, float, float&, float&, float&)> F);
  bool doCalibrationStep();

 private:
  /// Methods

  void createCurrentSpline();
  void createActionSpline();
  void createSpline(IrregularSpline2D3D& sp, std::vector<float>& data);

  Action checkActionShift(std::list<KnotData>::iterator& knot);

  Action checkActionRemove(std::list<KnotData>::iterator& knot);

  void getRegionOfInfluence(std::list<KnotData>::iterator knot, int& regionKnotFirst, int& regionKnotLast) const;

  double getMaxDeviationLine(const IrregularSpline2D3D& spline, const std::vector<float>& data, int axis, int knot) const;
  double getMaxDeviationArea(const IrregularSpline2D3D& spline, const std::vector<float>& data,
                             int axis, int knotFirst, int knotLast) const;
  double getIntegralDeviationLine(const IrregularSpline2D3D& spline, const std::vector<float>& data, int axis, int knot) const;
  double getIntegralDeviationArea(const IrregularSpline2D3D& spline, const std::vector<float>& data,
                                  int axis, int knotFirst, int knotLast) const;

  /// Class members

  int mMaxNKnots[2] = {5, 5}; ///< max N knots, U / V axis

  std::list<KnotData> mKnots[2]; ///< list of knots for U/V axis

  IrregularSpline2D3D mRaster;    ///< a spline of the maximal size which represents the input function
  std::vector<float> mRasterData; ///< function values for the mRaster, with corrected edges

  IrregularSpline2D3D mSpline;    ///< current spline
  std::vector<float> mSplineData; ///< function values for the mSpline, with corrected edges

  IrregularSpline2D3D mActionSpline;    ///< spline to test the cost of the current action
  std::vector<float> mActionSplineData; ///< function values for the mSpline, with corrected edges

  std::vector<float> mTemp[2]; ///< temp. arrays

  float mMaxDeviation = 0.1; ///< maximal tolerated deviation between the spline and the input function
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
