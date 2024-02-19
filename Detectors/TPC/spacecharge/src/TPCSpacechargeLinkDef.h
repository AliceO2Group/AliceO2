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

/// \file TPCSpacechargeLinkDef.h
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedef;

#pragma link C++ struct o2::tpc::MGParameters + ;
#pragma link C++ struct o2::tpc::TPCParameters < double> + ;
#pragma link C++ class o2::tpc::AnalyticalFields < double> + ;
#pragma link C++ class o2::tpc::RegularGrid3D < double> + ;
#pragma link C++ struct o2::tpc::RegularGridHelper < double> + ;
#pragma link C++ class o2::tpc::TriCubicInterpolator < double> + ;
#pragma link C++ class o2::tpc::DataContainer3D < double> + ;
#pragma link C++ class o2::tpc::PoissonSolver < double> + ;
#pragma link C++ class o2::tpc::SpaceCharge < double> + ;
#pragma link C++ class o2::tpc::NumericalFields < double> + ;
#pragma link C++ class o2::tpc::DistCorrInterpolator < double> + ;
#pragma link C++ class o2::tpc::GridProperties < double> + ;
#pragma link C++ struct o2::tpc::TPCParameters < float> + ;
#pragma link C++ class o2::tpc::AnalyticalFields < float> + ;
#pragma link C++ class o2::tpc::RegularGrid3D < float> + ;
#pragma link C++ struct o2::tpc::RegularGridHelper < float> + ;
#pragma link C++ class o2::tpc::TriCubicInterpolator < float> + ;
#pragma link C++ class o2::tpc::DataContainer3D < float> + ;
#pragma link C++ class o2::tpc::PoissonSolver < float> + ;
#pragma link C++ class o2::tpc::SpaceCharge < float> + ;
#pragma link C++ class o2::tpc::NumericalFields < float> + ;
#pragma link C++ class o2::tpc::DistCorrInterpolator < float> + ;
#pragma link C++ class o2::tpc::GridProperties < float> + ;
#pragma link C++ class o2::tpc::AnalyticalDistCorr < double> + ;
#pragma link C++ class o2::tpc::AnalyticalDistCorr < float> + ;
#pragma link C++ struct o2::tpc::ParamSpaceCharge + ;
#pragma link C++ struct o2::tpc::SCMetaData + ;
#endif
