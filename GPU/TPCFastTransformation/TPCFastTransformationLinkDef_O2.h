// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

//#pragma link C++ function o2::gpu::initSplineLibrary;

#pragma link C++ namespace o2::gpu;
//#pragma link C++ namespace o2::gpu::test+;
//#pragma link C++ class o2::gpu::test::A<float>+;

#pragma link C++ class o2::gpu::Spline1D < float> + ;
#pragma link C++ class o2::gpu::Spline1D < double> + ;
#pragma link C++ class o2::gpu::Spline2DBase < float, false> + ;
#pragma link C++ class o2::gpu::Spline2DBase < double, false> + ;
#pragma link C++ class o2::gpu::Spline2DBase < float, true> + ;
#pragma link C++ class o2::gpu::Spline2DBase < double, true> + ;

#pragma link C++ class o2::gpu::Spline2D < float, 1> - ;

#pragma link C++ class o2::gpu::SplineHelper1D < float>;
#pragma link C++ class o2::gpu::SplineHelper1D < double>;
#pragma link C++ class o2::gpu::SplineHelper2D < float>;
#pragma link C++ class o2::gpu::SplineHelper2D < double>;

#pragma link C++ class o2::gpu::ChebyshevFit1D;

#pragma link C++ class o2::gpu::RegularSpline1D + ;
#pragma link C++ class o2::gpu::IrregularSpline1D + ;
#pragma link C++ class o2::gpu::IrregularSpline2D3D + ;
#pragma link C++ class o2::gpu::SemiregularSpline2D3D + ;
#pragma link C++ class o2::gpu::IrregularSpline2D3DCalibrator + ;
#pragma link C++ class o2::gpu::TPCFastTransformGeo + ;
#pragma link C++ class o2::gpu::TPCFastTransformGeo::SliceInfo + ;
#pragma link C++ class o2::gpu::TPCFastTransformGeo::RowInfo + ;
#pragma link C++ class o2::gpu::TPCFastTransform + ;
#pragma link C++ class o2::gpu::TPCFastSpaceChargeCorrection::SliceInfo + ;
#pragma link C++ class o2::gpu::TPCFastSpaceChargeCorrection + ;

#endif
