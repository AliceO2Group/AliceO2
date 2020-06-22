// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_Analysis_ReducedInfoTables_H_
#define O2_Analysis_ReducedInfoTables_H_

#include "Framework/AnalysisDataModel.h"
#include "Analysis/Centrality.h"
//#include "Framework/ASoA.h"
#include "MathUtils/Utils.h"
#include <cmath>

namespace o2
{
namespace aod
{
// This is required to register SOA_TABLEs inside
// the o2::aod namespace.
//DECLARE_SOA_STORE();

namespace reducedevent {
  
  // basic event information
  DECLARE_SOA_COLUMN(Tag, tag, uint64_t);
  //DECLARE_SOA_COLUMN(RunNumber, runNumber, uint32_t);
  //DECLARE_SOA_COLUMN(VtxX, vtxX, float);
  //DECLARE_SOA_COLUMN(VtxY, vtxY, float);
  //DECLARE_SOA_COLUMN(VtxZ, vtxZ, float);
  //DECLARE_SOA_COLUMN(NContrib, nContrib, uint32_t);
  
  // extended event information
  //DECLARE_SOA_COLUMN(BC, bc, uint64_t);
  //DECLARE_SOA_COLUMN(TriggerMask, triggerMask, uint64_t);
  //DECLARE_SOA_COLUMN(CollisionTime, collisionTime, float);
  //DECLARE_SOA_COLUMN(CentVZERO, centVZERO, float);
  
  // event vertex covariance matrix
  /*DECLARE_SOA_COLUMN(CovXX, covXX, float);
  DECLARE_SOA_COLUMN(CovXY, covXY, float);
  DECLARE_SOA_COLUMN(CovXZ, covXZ, float);
  DECLARE_SOA_COLUMN(CovYY, covYY, float);
  DECLARE_SOA_COLUMN(CovYZ, covYZ, float);
  DECLARE_SOA_COLUMN(CovZZ, covZZ, float);
  DECLARE_SOA_COLUMN(Chi2, chi2, float);*/
  
  // TODO: implement additional separate tables for other types of information, e.g. multiplicity estimators, VZERO-TZERO/FIT, ZDC, etc.
  
}  // namespace reducedevent

DECLARE_SOA_TABLE(ReducedEvents, "AOD", "REDUCEDEVENT", o2::soa::Index<>,
                  reducedevent::Tag, bc::RunNumber, 
                  collision::PosX, collision::PosY, collision::PosZ, collision::NumContrib);

DECLARE_SOA_TABLE(ReducedEventsExtended, "AOD", "REEXTENDED",
                  bc::GlobalBC, bc::TriggerMask, collision::CollisionTime, cent::CentV0M);

DECLARE_SOA_TABLE(ReducedEventsVtxCov, "AOD", "REVTXCOV",
                  collision::CovXX, collision::CovXY, collision::CovXZ,
                  collision::CovYY, collision::CovYZ, collision::CovZZ, collision::Chi2);

using ReducedEvent = ReducedEvents::iterator;
using ReducedEventExtended = ReducedEventsExtended::iterator;
using ReducedEventVtxCov = ReducedEventsVtxCov::iterator;



namespace reducedtrack {
  // basic track information
  DECLARE_SOA_INDEX_COLUMN(ReducedEvent, reducedevent);
  DECLARE_SOA_COLUMN(Index, index, uint16_t);
  // ----  flags reserved for storing various information during filtering
  DECLARE_SOA_COLUMN(FilteringFlags, filteringFlags, uint64_t);   
  // BIT 0: track is from MUON arm      (if not toggled then this is a barrel track)
  // -----------------------------------------------------
  DECLARE_SOA_COLUMN(Pt, pt, float);
  DECLARE_SOA_COLUMN(Eta, eta, float);
  DECLARE_SOA_COLUMN(Phi, phi, float);
  DECLARE_SOA_COLUMN(Charge, charge, short);
  DECLARE_SOA_DYNAMIC_COLUMN(Px, px, [](float pt, float phi) -> float { return abs(pt)*cos(phi); });
  DECLARE_SOA_DYNAMIC_COLUMN(Py, py, [](float pt, float phi) -> float { return abs(pt)*sin(phi); });
  DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, [](float pt, float eta) -> float { return abs(pt)*sinh(eta); });
  DECLARE_SOA_DYNAMIC_COLUMN(Pmom, pmom, [](float pt, float eta) -> float { return abs(pt)*cosh(eta); });
  
  
  /*DECLARE_SOA_COLUMN(P1, p1, float);
  DECLARE_SOA_COLUMN(P2, p2, float);
  DECLARE_SOA_COLUMN(P3, p3, float);
  DECLARE_SOA_COLUMN(IsCartesian, isCartesian, uint8_t);
  DECLARE_SOA_COLUMN(Charge, charge, short);
  
  DECLARE_SOA_DYNAMIC_COLUMN(Px, px, [](float p1, float p2, float p3, uint8_t isCartesian) -> float { return (isCartesian ? p1 : abs(p1)*cos(p2)); });
  DECLARE_SOA_DYNAMIC_COLUMN(Py, py, [](float p1, float p2, float p3, uint8_t isCartesian) -> float { return (isCartesian ? p2 : abs(p1)*sin(p2)); });
  DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, [](float p1, float p2, float p3, uint8_t isCartesian) -> float { return (isCartesian ? p3 : abs(p1)*sinh(p3)); });
  DECLARE_SOA_DYNAMIC_COLUMN(Pmom, pmom, [](float p1, float p2, float p3, uint8_t isCartesian) -> float { return (isCartesian ? sqrt(p1*p1+p2*p2+p3*p3) : abs(p1)*cosh(p3)); });
  DECLARE_SOA_DYNAMIC_COLUMN(Pt, pt, [](float p1, float p2, float p3, uint8_t isCartesian) -> float { return (isCartesian ? sqrt(p1*p1+p2*p2) : p1); });
  DECLARE_SOA_DYNAMIC_COLUMN(Phi, phi, [](float p1, float p2, float p3, uint8_t isCartesian) -> float { 
    if(!isCartesian) return p2;
    auto phi=atan2(p2,p1); 
    if(phi>=0.0) 
      return phi;
    else 
      return phi+2.0*static_cast<float>(M_PI);
  });
  DECLARE_SOA_DYNAMIC_COLUMN(Eta, eta, [](float p1, float p2, float p3, uint8_t isCartesian) -> float { 
    if(!isCartesian) return p3;
    auto mom = sqrt(p1*p1+p2*p2+p3*p3);
    auto theta = (mom>1.0e-6 ? acos(p3/mom) : 0.0);
    auto eta = tan(0.5*theta);
    if(eta>1.0e-6)
      return -1.0*log(eta);
    else 
      return 0.0;
  });*/
  
  /*
  // Central barrel extra track information
  DECLARE_SOA_COLUMN(TPCInnerParam, tpcInnerParam, float);
  DECLARE_SOA_COLUMN(TrackingFlags, trackingFlags, uint64_t);
  DECLARE_SOA_COLUMN(ITSClusterMap, itsClusterMap, uint8_t);
  DECLARE_SOA_COLUMN(ITSChi2NCl, itsChi2NCl, float);
  DECLARE_SOA_COLUMN(TPCNCls, tpcNCls, uint8_t);
  DECLARE_SOA_COLUMN(TPCNClsFindable, tpcNClsFindable, int8_t);
  DECLARE_SOA_COLUMN(TPCNCrossedRows, tpcNCrossedRows, int8_t);
  DECLARE_SOA_COLUMN(TPCNClsShared, tpcNClsShared, uint8_t);
  DECLARE_SOA_COLUMN(TPCChi2NCl, tpcChi2NCl, float);
  DECLARE_SOA_COLUMN(TPCSignal, tpcSignal, float);
  DECLARE_SOA_COLUMN(TRDNTracklets, trdNTracklets, uint8_t);
  DECLARE_SOA_COLUMN(TRDChi2, trdChi2, float);
  DECLARE_SOA_COLUMN(TRDSignal, trdSignal, float);
  DECLARE_SOA_COLUMN(TOFSignal, tofSignal, float);
  DECLARE_SOA_COLUMN(TOFChi2, tofChi2, float);
  DECLARE_SOA_COLUMN(BarrelLength, barrelLength, float);
*/
  
  // Central barrel track covariance matrix
/*  DECLARE_SOA_COLUMN(CYY, cYY, float);
  DECLARE_SOA_COLUMN(CZZ, cZZ, float);
  DECLARE_SOA_COLUMN(CSnpSnp, cSnpSnp, float);
  DECLARE_SOA_COLUMN(CTglTgl, cTglTgl, float);
  DECLARE_SOA_COLUMN(C1Pt21Pt2, c1Pt21Pt2, float);
*/
  // TODO: central barrel covariance matrix
  

  // MUON tracks extra information 
  DECLARE_SOA_COLUMN(MuonChi2, muonChi2, float);
  DECLARE_SOA_COLUMN(MuonChi2MatchTrigger, muonChi2MatchTrigger, float);
  
}  //namespace reducedtrack

/*DECLARE_SOA_TABLE(ReducedTracks, "AOD", "REDUCEDTRACK",
                  o2::soa::Index<>, reducedtrack::ReducedEventId, reducedtrack::Index,
                  reducedtrack::P1, reducedtrack::P2, reducedtrack::P3, reducedtrack::IsCartesian, reducedtrack::Charge,
                  reducedtrack::Flags, 
                  reducedtrack::Px<reducedtrack::P1, reducedtrack::P2, reducedtrack::P3, reducedtrack::IsCartesian>,
                  reducedtrack::Py<reducedtrack::P1, reducedtrack::P2, reducedtrack::P3, reducedtrack::IsCartesian>,
                  reducedtrack::Pz<reducedtrack::P1, reducedtrack::P2, reducedtrack::P3, reducedtrack::IsCartesian>,
                  reducedtrack::Pmom<reducedtrack::P1, reducedtrack::P2, reducedtrack::P3, reducedtrack::IsCartesian>,
                  reducedtrack::Pt<reducedtrack::P1, reducedtrack::P2, reducedtrack::P3, reducedtrack::IsCartesian>,
                  reducedtrack::Phi<reducedtrack::P1, reducedtrack::P2, reducedtrack::P3, reducedtrack::IsCartesian>,
                  reducedtrack::Eta<reducedtrack::P1, reducedtrack::P2, reducedtrack::P3, reducedtrack::IsCartesian>);
*/
DECLARE_SOA_TABLE(ReducedTracks, "AOD", "REDUCEDTRACK",
                  o2::soa::Index<>, reducedtrack::ReducedEventId, reducedtrack::Index, reducedtrack::FilteringFlags, 
                  reducedtrack::Pt, reducedtrack::Eta, reducedtrack::Phi, reducedtrack::Charge,
                  reducedtrack::Px<reducedtrack::Pt, reducedtrack::Phi>,
                  reducedtrack::Py<reducedtrack::Pt, reducedtrack::Phi>,
                  reducedtrack::Pz<reducedtrack::Pt, reducedtrack::Eta>,
                  reducedtrack::Pmom<reducedtrack::Pt, reducedtrack::Eta>);


/*DECLARE_SOA_TABLE(ReducedTracksBarrel, "AOD", "RTBARREL",
                  reducedtrack::TPCInnerParam, 
                  reducedtrack::TrackingFlags,
                  reducedtrack::ITSClusterMap, 
                  reducedtrack::ITSChi2NCl, 
                  reducedtrack::TPCNCls, 
                  reducedtrack::TPCNClsFindable, 
                  reducedtrack::TPCNCrossedRows, 
                  reducedtrack::TPCNClsShared,
                  reducedtrack::TPCChi2NCl, 
                  reducedtrack::TPCSignal,
                  reducedtrack::TRDNTracklets, 
                  reducedtrack::TRDChi2, 
                  reducedtrack::TRDSignal,
                  reducedtrack::TOFSignal, 
                  reducedtrack::TOFChi2,
                  reducedtrack::BarrelLength);*/
DECLARE_SOA_TABLE(ReducedTracksBarrel, "AOD", "RTBARREL",
                  track::TPCInnerParam, track::Flags,                   // tracking status flags
                  track::ITSClusterMap, track::ITSChi2NCl, 
                  track::TPCNClsFindable, track::TPCNClsFindableMinusFound, track::TPCNClsFindableMinusCrossedRows,
                  track::TPCNClsShared, track::TPCChi2NCl,
                  track::TPCSignal, track::TRDSignal, track::TOFSignal,
                  track::TRDNTracklets, track::TRDChi2, track::TOFChi2, track::Length,
                  track::TPCNClsFound<track::TPCNClsFindable, track::TPCNClsFindableMinusFound>,
                  track::TPCNClsCrossedRows<track::TPCNClsFindable, track::TPCNClsFindableMinusCrossedRows>);


/*DECLARE_SOA_TABLE(ReducedTracksBarrelCov, "AOD", "RTBARRELCOV",
                  reducedtrack::CYY, reducedtrack::CZZ, reducedtrack::CSnpSnp,
                  reducedtrack::CTglTgl, reducedtrack::C1Pt21Pt2);*/

DECLARE_SOA_TABLE(ReducedTracksBarrelCov, "AOD", "RTBARRELCOV",
                  track::CYY, track::CZZ, track::CSnpSnp,
                  track::CTglTgl, track::C1Pt21Pt2);                 

/*DECLARE_SOA_TABLE(ReducedTracksMuon, "AOD", "RTMUON",
                  reducedtrack::MuonChi2, reducedtrack::MuonChi2MatchTrigger);*/

// TODO: change names of these columns in the AnalysisDataModel to identify the members as muon quantities
DECLARE_SOA_TABLE(ReducedTracksMuon, "AOD", "RTMUON",
                  reducedtrack::MuonChi2, reducedtrack::MuonChi2MatchTrigger);


using ReducedTrack = ReducedTracks::iterator;
using ReducedTrackBarrel = ReducedTracksBarrel::iterator;
using ReducedTrackBarrelCov = ReducedTracksBarrelCov::iterator;
using ReducedTrackMuon = ReducedTracksMuon::iterator;

} // namespace aod

} // namespace o2

#endif // O2_ANALYSIS_REDUCEDINFOTABLES_H_
