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

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::globaltracking::MatchTOF + ;
#pragma link C++ class o2::globaltracking::TrackLocTPC + ;
#pragma link C++ class o2::globaltracking::TrackLocITS + ;
#pragma link C++ class o2::globaltracking::MatchTPCITSParams + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::globaltracking::MatchTPCITSParams> + ;

#pragma link C++ class o2::globaltracking::MatchCosmics + ;
#pragma link C++ class o2::globaltracking::MatchCosmicsParams + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::globaltracking::MatchCosmicsParams> + ;

#pragma link C++ class o2::globaltracking::MatchTOFParams + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::globaltracking::MatchTOFParams> + ;

#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::globaltracking::ITSTPCMatchingQCParams> + ;

#pragma link C++ class o2::globaltracking::GloFwdAssessment + ;

#pragma link C++ class o2::globaltracking::GlobalFwdMatchingParam + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::globaltracking::GlobalFwdMatchingParam> + ;

// RS FIXME remove ?
//#pragma link C++ class o2::globaltracking::ABDebugLink + ;
//#pragma link C++ class o2::globaltracking::ABDebugTrack + ;

#pragma link C++ class std::pair < o2::dataformats::EvIndex < int, int>, o2::dataformats::MatchInfoTOF> + ;
#pragma link C++ class std::vector < std::pair < o2::dataformats::EvIndex < int, int>, o2::dataformats::MatchInfoTOF>> + ;

#pragma link C++ class std::vector < std::tuple < float, float, float>> + ;

#endif
