// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::data::Stack + ;
#pragma link C++ class o2::sim::StackParam + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::sim::StackParam> + ;
#pragma link C++ class o2::MCTrackT < double> + ;
#pragma link C++ class o2::MCTrackT < float> + ;
#pragma link C++ class o2::MCTrack + ;
#pragma link C++ class std::vector < o2::MCTrack> + ;
#pragma link C++ class std::vector < o2::MCTrackT < double>> + ;
#pragma link C++ class std::vector < o2::MCTrackT < float>> + ;
#pragma link C++ class o2::MCCompLabel + ;

#pragma link C++ class o2::BaseHit + ;
#pragma link C++ class o2::BasicXYZVHit < float, float, float> + ;
#pragma link C++ class o2::BasicXYZVHit < double, double, double> + ;
#pragma link C++ class o2::BasicXYZVHit < float, float, int> + ;
#pragma link C++ class o2::BasicXYZVHit < double, double, int> + ;
#pragma link C++ class o2::BasicXYZEHit < float, float, float> + ;
#pragma link C++ class o2::BasicXYZEHit < double, double, double> + ;
#pragma link C++ class o2::BasicXYZQHit < float, float, int> + ;
#pragma link C++ class o2::BasicXYZQHit < double, double, int> + ;
#pragma link C++ struct o2::dataformats::MCTruthHeaderElement + ;
#pragma link C++ class o2::dataformats::MCTruthContainer < long> + ;
#pragma link C++ class o2::dataformats::MCTruthContainer < o2::MCCompLabel> + ;
#pragma link C++ class std::vector < o2::dataformats::MCTruthContainer < o2::MCCompLabel>> + ;
#pragma link C++ class std::vector < o2::MCCompLabel> + ;
#pragma link C++ class std::vector < o2::dataformats::MCTruthHeaderElement> + ;

#pragma link C++ class o2::SimTrackStatus + ;
#pragma link C++ class o2::TrackReference + ;
#pragma link C++ class std::vector < o2::TrackReference> + ;
#pragma link C++ class o2::dataformats::MCTruthContainer < o2::TrackReference> + ;

#pragma link C++ struct o2::data::SubEventInfo + ;
#pragma link C++ class std::vector < o2::data::SubEventInfo> + ;
#pragma link C++ struct o2::data::PrimaryChunk + ;

#pragma link C++ class o2::steer::RunContext + ;
#pragma link C++ class o2::steer::EventPart + ;
#pragma link C++ class vector < o2::steer::EventPart> + ;
#pragma link C++ class vector < vector < o2::steer::EventPart>> + ;

#pragma link C++ class o2::dataformats::MCEventStats + ;
#pragma link C++ class o2::dataformats::MCEventHeader + ;

#endif
