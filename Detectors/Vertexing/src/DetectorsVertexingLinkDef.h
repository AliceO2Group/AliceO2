// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

// this aliase are defined in the DCAFitterN.h as o2::vertexing::DCAFitterN<2,o2::track::TrackParCov>
#pragma link C++ class o2::vertexing::DCAFitter2 + ;
// this aliase are defined in the DCAFitterN.h as o2::vertexing::DCAFitterN<3,o2::track::TrackParCov>
#pragma link C++ class o2::vertexing::DCAFitter3 + ;

#pragma link C++ class o2::vertexing::PVertexerParams + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::vertexing::PVertexerParams> + ;

#pragma link C++ class o2::vertexing::SVertexHypothesis + ;
#pragma link C++ class o2::vertexing::SVertexerParams + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::vertexing::SVertexerParams> + ;

#pragma link C++ class o2::track::TrackAuxPar + ;
#pragma link C++ class o2::track::CrossInfo + ;

#pragma link C++ function o2::vertexing::DCAFitter2::process(const o2::track::TrackParCov&, const o2::track::TrackParCov&);
#pragma link C++ function o2::vertexing::DCAFitter3::process(const o2::track::TrackParCov&, const o2::track::TrackParCov&, const o2::track::TrackParCov&);

#pragma link C++ class o2::vertexing::TrackVFDump + ;
#pragma link C++ class std::vector < o2::vertexing::TrackVFDump> + ;

#pragma link C++ class o2::vertexing::PVtxCompDump + ;
#pragma link C++ class std::vector < o2::vertexing::PVtxCompDump> + ;

#endif
