// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  SemiregularSpline2D3D.h
/// \brief Definition of SemiregularSpline2D3D class
///
/// \author  Oscar Lange
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_IRREGULARSPLINE2D3DCALIBRATORV0_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_IRREGULARSPLINE2D3DCALIBRATORV0_H

#include "GPUCommonDef.h"
#include "IrregularSpline2D3D.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class IrregularSpline2D3DCalibratorV0
{
 public:
  /// _____________  Constructors / destructors __________________________

  /// Default constructor. Creates an empty uninitialised object
  IrregularSpline2D3DCalibratorV0();

  /// Destructor
  ~IrregularSpline2D3DCalibratorV0() CON_DEFAULT;

  /*function generates a spline from a given function
   *           input spline => the spline that needs to be aproximated
   *           F => input function
   */
  float* generate_input_spline(IrregularSpline2D3D& input_spline, void (*F)(float, float, float&, float&, float&));

  /*function generates a spline for comparison to the original, a so called compare spline
   *the compare spline is later used to find the spots where the aproximation is not accurate enough
   *parameters: 
   *           compare spline => the spline needs to be initialized outside and given to the function
   *           input spline => the spline that needs to be aproximated
   *           input data => the data of the input spline
   */
  float* generate_compare_spline(IrregularSpline2D3D& compare_spline, IrregularSpline2D3D& input_spline, float* input_data);

  /*function generates a spline that is calibrated on the u axis
   *parameters: 
   *           input spline => the spline that needs to be aproximated
   *           compare spline => the spline used for comparison
   *           spline_u => the resulting calibrated spline
   *           input data => the data of the input spline
   *           compare data => the data of the compare spline
   */
  float* generate_spline_u(IrregularSpline2D3D& input_spline, IrregularSpline2D3D& compare_spline, IrregularSpline2D3D& spline_u, float* input_data, float* compare_data);

  /*function generates a spline that is calibrated on the u and v axis 
   *parameters: 
   *           input spline => the spline that needs to be aproximated
   *           compare spline => the spline used for comparison
   *           spline_u => the resulting calibrated spline
   *           input data => the data of the input spline
   *           compare data => the data of the compare spline
   */
  float* generate_spline_uv(IrregularSpline2D3D& input_spline, IrregularSpline2D3D& spline_u, IrregularSpline2D3D& spline_uv, float* input_data, float* data_u);

  /*function uses all three function above to construct a calibrated Spline
   *parameters: 
   *           spline_uv => calibrated spline
   *           F => input function
   */
  float* calibrateSpline(IrregularSpline2D3D& spline_uv, void (*F)(float, float, float&, float&, float&));

  // the number of knots of the starting grid
  int nKnotsU, nKnotsV;
  // the number of knots of the comparison grid
  int minKnotsU, minKnotsV;
  // check IrregularSpline2D3D.cxx for documentation
  int nAxisTicksU, nAxisTicksV;
  // the number of times a value inbetween two existing knots is checked
  int u_ticks, v_ticks;
  // the threshold value of placing a new value
  double max_tolerated_deviation;
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
