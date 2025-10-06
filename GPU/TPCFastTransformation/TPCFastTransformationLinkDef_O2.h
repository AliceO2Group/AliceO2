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

/// \file TPCFastTransformationLinkDef_O2.h
/// \author Sergey Gorbunov

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedef;

#pragma link C++ namespace o2::gpu;

#pragma link C++ class o2::gpu::Spline1DContainer < float> + ;
#pragma link C++ class o2::gpu::Spline1DContainer < double> + ;
#pragma link C++ class o2::gpu::Spline1D < float> + ;
#pragma link C++ class o2::gpu::Spline1D < double> + ;
#pragma link C++ class o2::gpu::Spline1DHelperOld < float> + ;
#pragma link C++ class o2::gpu::Spline1DHelperOld < double> + ;
#pragma link C++ class o2::gpu::Spline1DHelper < float> + ;
#pragma link C++ class o2::gpu::Spline1DHelper < double> + ;
#pragma link C++ class o2::gpu::Spline1DSpec < float, 0, 2> + ;
#pragma link C++ class o2::gpu::Spline1DSpec < double, 0, 2> + ;

#pragma link C++ class o2::gpu::Spline2DContainer < float> + ;
#pragma link C++ class o2::gpu::Spline2DContainer < double> + ;
#pragma link C++ class o2::gpu::Spline2D < float> + ;
#pragma link C++ class o2::gpu::Spline2D < double> + ;
#pragma link C++ class o2::gpu::Spline2DHelper < float> + ;
#pragma link C++ class o2::gpu::Spline2DHelper < double> + ;

#pragma link C++ class o2::gpu::SplineContainer < float> + ;
#pragma link C++ class o2::gpu::SplineContainer < double> + ;
#pragma link C++ class o2::gpu::Spline < float> + ;
#pragma link C++ class o2::gpu::Spline < double> + ;
#pragma link C++ class o2::gpu::SplineHelper < float> + ;
#pragma link C++ class o2::gpu::SplineHelper < double> + ;

#pragma link C++ class o2::gpu::ChebyshevFit1D + ;
#pragma link C++ class o2::gpu::SymMatrixSolver + ;
#pragma link C++ class o2::gpu::BandMatrixSolver < 0> + ;

#pragma link C++ class o2::gpu::RegularSpline1D + ;
#pragma link C++ class o2::gpu::IrregularSpline1D + ;
#pragma link C++ class o2::gpu::IrregularSpline2D3D + ;
#pragma link C++ class o2::gpu::SemiregularSpline2D3D + ;
#pragma link C++ class o2::gpu::IrregularSpline2D3DCalibrator + ;

#pragma link C++ class o2::gpu::TPCFastTransformGeo::SliceInfo + ;
#pragma link C++ class o2::gpu::TPCFastTransformGeo::SectorInfo + ;

#pragma link C++ class o2::gpu::TPCFastTransformGeo + ;
#pragma read \
  sourceClass = "o2::gpu::TPCFastTransformGeo" targetClass = "o2::gpu::TPCFastTransformGeo" source = "float mTPCzLengthA; float mTPCzLengthC; float mTPCalignmentZ; float mScaleVtoSVsideA; float mScaleVtoSVsideC; float mScaleSVtoVsideA; float mScaleSVtoVsideC;" version = "[-1]" target = "mTPCzLength" code = "{ mTPCzLength = onfile.mTPCzLengthA; }";

#pragma read \
  sourceClass = "o2::gpu::TPCFastTransformGeo" targetClass = "o2::gpu::TPCFastTransformGeo" source = "o2::gpu::TPCFastTransformGeo::SliceInfo mSliceInfos[37]" version = "[1-]" target = "" code = "{}";

#pragma link C++ class o2::gpu::TPCFastTransformGeo::RowInfo + ;
#pragma read \
  sourceClass = "o2::gpu::TPCFastTransformGeo::RowInfo" targetClass = "o2::gpu::TPCFastTransformGeo::RowInfo" source = "float u0; float scaleUtoSU; float scaleSUtoU" version = "[-2]" target = "yMin" code = "{ yMin = onfile.u0; }"

#pragma link C++ class o2::gpu::TPCFastTransform + ;

#pragma link C++ class o2::gpu::TPCFastSpaceChargeCorrectionMap + ;

#pragma link C++ class o2::gpu::TPCFastSpaceChargeCorrection + ;
#pragma link C++ class o2::gpu::TPCFastSpaceChargeCorrection::SliceInfo + ;
#pragma link C++ class o2::gpu::TPCFastSpaceChargeCorrection::SectorRowInfo + ;
#pragma link C++ class o2::gpu::TPCFastSpaceChargeCorrection::GridInfo + ;
#pragma read \
  sourceClass = "o2::gpu::TPCFastSpaceChargeCorrection" targetClass = "o2::gpu::TPCFastSpaceChargeCorrection" source = "o2::gpu::TPCFastSpaceChargeCorrection::SliceInfo mSliceInfo[36]" version = "[-3]" target = "" code = "{}";

#pragma read \
  sourceClass = "o2::gpu::TPCFastSpaceChargeCorrection" targetClass = "o2::gpu::TPCFastSpaceChargeCorrection" source = "size_t mSliceDataSizeBytes[3]" version = "[-3]" target = "mCorrectionDataSize" code = "{ for (int i=0; i<3; i++) mCorrectionDataSize[i] = onfile.mSliceDataSizeBytes[i] * 36; }";

#pragma read \
  sourceClass = "o2::gpu::TPCFastSpaceChargeCorrection" targetClass = "o2::gpu::TPCFastSpaceChargeCorrection" source = "float fInterpolationSafetyMargin" version = "[-3]" target = "" code = "{}";

#pragma link C++ class o2::gpu::CorrectionMapsHelper + ;
#pragma link C++ struct o2::gpu::MultivariatePolynomialContainer + ;
#pragma link C++ struct o2::gpu::NDPiecewisePolynomialContainer + ;
#pragma link C++ struct o2::gpu::TPCSlowSpaceChargeCorrection + ;

#endif
