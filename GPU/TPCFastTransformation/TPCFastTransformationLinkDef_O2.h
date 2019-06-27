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

#pragma link C++ class o2::gpu::RegularSpline1D+;
#pragma link C++ class o2::gpu::IrregularSpline1D+;
#pragma link C++ class o2::gpu::IrregularSpline2D3D+;
#pragma link C++ class o2::gpu::SemiregularSpline2D3D+;
#pragma link C++ class o2::gpu::IrregularSpline2D3DCalibrator+;
#pragma link C++ class o2::gpu::TPCFastTransform+;
#pragma link C++ class o2::gpu::TPCDistortionIRS+;

#endif
