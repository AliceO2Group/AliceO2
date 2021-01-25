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

#pragma link C++ class o2::track::TrackPar + ;
#pragma link C++ class o2::track::TrackParametrization < float> + ;
#pragma link C++ class o2::track::TrackParametrization < double> + ;
#pragma link C++ class o2::track::TrackParCov + ;
#pragma link C++ class o2::track::TrackParametrizationWithError < float> + ;
#pragma link C++ class o2::track::TrackParametrizationWithError < double> + ;
#pragma link C++ class o2::track::TrackParFwd + ;
#pragma link C++ class o2::track::TrackParCovFwd + ;
#pragma link C++ class o2::track::PID + ;
#pragma link C++ class o2::track::TrackLTIntegral + ;

#pragma link C++ class o2::BaseCluster < float> + ;
#pragma link C++ class o2::dataformats::TrackTPCITS + ;
#pragma link C++ class std::vector < o2::dataformats::TrackTPCITS> + ;
#pragma link C++ class o2::dataformats::MatchInfoTOF + ;
#pragma link C++ class std::vector < o2::dataformats::MatchInfoTOF> + ;

#pragma link C++ class std::vector < std::pair < float, float>> + ;
#pragma link C++ class std::vector < std::pair < int, float>> + ;
#pragma link C++ class std::vector < int> + ;

#pragma link C++ class o2::dataformats::VertexBase + ;
#pragma link C++ class o2::dataformats::Vertex < int> + ;
#pragma link C++ class o2::dataformats::Vertex < o2::dataformats::TimeStamp < int>> + ;
#pragma link C++ class o2::dataformats::Vertex < o2::dataformats::TimeStampWithError < float, float>> + ;
#pragma link C++ class o2::dataformats::PrimaryVertex + ;

#pragma link C++ class std::vector < o2::dataformats::Vertex < o2::dataformats::TimeStamp < int>>> + ;
#pragma link C++ class std::vector < o2::dataformats::Vertex < o2::dataformats::TimeStampWithError < float, float>>> + ;
#pragma link C++ class std::vector < o2::dataformats::PrimaryVertex> + ;

#pragma link C++ class o2::dataformats::GlobalTrackID + ;
#pragma link C++ class std::vector < o2::dataformats::GlobalTrackID> + ;
#pragma link C++ class o2::dataformats::EvIndex < int, o2::dataformats::GlobalTrackID> + ;

#pragma link C++ class o2::dataformats::VtxTrackIndex + ;
#pragma link C++ class std::vector < o2::dataformats::VtxTrackIndex> + ;

#pragma link C++ class o2::dataformats::VtxTrackRef + ;
#pragma link C++ class std::vector < o2::dataformats::VtxTrackRef> + ;

#pragma link C++ class o2::dataformats::DCA + ;

#pragma link C++ class o2::dataformats::V0 + ;
#pragma link C++ class std::vector < o2::dataformats::V0> + ;

#endif
