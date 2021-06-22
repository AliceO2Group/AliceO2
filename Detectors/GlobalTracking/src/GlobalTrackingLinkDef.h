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

#pragma link C++ class o2::globaltracking::MatchTOF + ;
#pragma link C++ class o2::globaltracking::TrackLocTPC + ;
#pragma link C++ class o2::globaltracking::TrackLocITS + ;
#pragma link C++ class o2::globaltracking::MatchTPCITSParams + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::globaltracking::MatchTPCITSParams> + ;

#pragma link C++ class o2::globaltracking::MatchCosmics + ;
#pragma link C++ class o2::globaltracking::MatchCosmicsParams + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::globaltracking::MatchCosmicsParams> + ;

#pragma link C++ class o2::globaltracking::ABDebugLink + ;
#pragma link C++ class o2::globaltracking::ABDebugTrack + ;

#pragma link C++ class std::pair < o2::dataformats::EvIndex < int, int>, o2::dataformats::MatchInfoTOF> + ;
#pragma link C++ class std::vector < std::pair < o2::dataformats::EvIndex < int, int>, o2::dataformats::MatchInfoTOF>> + ;

#endif
