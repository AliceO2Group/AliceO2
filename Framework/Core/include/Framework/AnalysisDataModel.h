// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_ANALYSISDATAMODEL_H_
#define O2_FRAMEWORK_ANALYSISDATAMODEL_H_

#include "Framework/ASoA.h"
#include <cmath>

namespace o2
{
namespace aod
{
// This is required to register SOA_TABLEs inside
// the o2::aod namespace.
DECLARE_SOA_STORE();

namespace track
{
// TRACKPAR TABLE definition
DECLARE_SOA_COLUMN(CollisionId, collisionId, int, "fID4Collisions");
DECLARE_SOA_COLUMN(X, x, float, "fX");
DECLARE_SOA_COLUMN(Alpha, alpha, float, "fAlpha");
DECLARE_SOA_COLUMN(Y, y, float, "fY");
DECLARE_SOA_COLUMN(Z, z, float, "fZ");
DECLARE_SOA_COLUMN(Snp, snp, float, "fSnp");
DECLARE_SOA_COLUMN(Tgl, tgl, float, "fTgl");
DECLARE_SOA_COLUMN(Signed1Pt, signed1Pt, float, "fSigned1Pt");
DECLARE_SOA_DYNAMIC_COLUMN(Phi, phi, [](float snp, float alpha) { return asin(snp) + alpha + M_PI; });
DECLARE_SOA_DYNAMIC_COLUMN(Eta, eta, [](float tgl) { return log(tan(0.25 * M_PI - 0.5 * atan(tgl))); });
DECLARE_SOA_DYNAMIC_COLUMN(Pt, pt, [](float signed1Pt) { return fabs(1.0 / signed1Pt); });
// TRACKPARCOV TABLE definition
DECLARE_SOA_COLUMN(CYY, cZZ, float, "fCYY");
DECLARE_SOA_COLUMN(CZY, cZY, float, "fCZY");
DECLARE_SOA_COLUMN(CZZ, cXX, float, "fCZZ");
DECLARE_SOA_COLUMN(CSnpY, cSnpY, float, "fCSnpY");
DECLARE_SOA_COLUMN(CSnpZ, cSnpZ, float, "fCSnpZ");
DECLARE_SOA_COLUMN(CSnpSnp, cSnpSnp, float, "fCSnpSnp");
DECLARE_SOA_COLUMN(CTglY, cTglY, float, "fCTglY");
DECLARE_SOA_COLUMN(CTglZ, cTglZ, float, "fCTglZ");
DECLARE_SOA_COLUMN(CTglSnp, cTglSnp, float, "fCTglSnp");
DECLARE_SOA_COLUMN(CTglTgl, cTglTgl, float, "fCTglTgl");
DECLARE_SOA_COLUMN(C1PtY, c1PtY, float, "fC1PtY");
DECLARE_SOA_COLUMN(C1PtZ, c1PtZ, float, "fC1PtZ");
DECLARE_SOA_COLUMN(C1PtSnp, c1PtSnp, float, "fC1PtSnp");
DECLARE_SOA_COLUMN(C1PtTgl, c1PtTgl, float, "fC1PtTgl");
DECLARE_SOA_COLUMN(C1Pt21Pt2, c1Pt21Pt2, float, "fC1Pt21Pt2");

DECLARE_SOA_COLUMN(TPCInnerParam, tpcInnerParam, float, "fTPCinnerP");
DECLARE_SOA_COLUMN(Flags, flags, uint64_t, "fFlags");
DECLARE_SOA_COLUMN(ITSClusterMap, itsClusterMap, uint8_t, "fITSClusterMap");
DECLARE_SOA_COLUMN(TPCNCls, tpcNCls, uint16_t, "fTPCncls");
DECLARE_SOA_COLUMN(TRDNTracklets, trdNTracklets, uint8_t, "fTRDntracklets");
DECLARE_SOA_COLUMN(ITSChi2NCl, itsChi2NCl, float, "fITSchi2Ncl");
DECLARE_SOA_COLUMN(TPCchi2Ncl, tpcChi2Ncl, float, "fTPCchi2Ncl");
DECLARE_SOA_COLUMN(TRDchi2, trdChi2, float, "fTRDchi2");
DECLARE_SOA_COLUMN(TOFchi2, tofChi2, float, "fTOFchi2");
DECLARE_SOA_COLUMN(TPCsignal, tpcSignal, float, "fTPCsignal");
DECLARE_SOA_COLUMN(TRDsignal, trdSignal, float, "fTRDsignal");
DECLARE_SOA_COLUMN(TOFsignal, tofSignal, float, "fTOFsignal");
DECLARE_SOA_COLUMN(Lenght, lenght, float, "fLength");

} // namespace track

DECLARE_SOA_TABLE(Tracks, "RN2", "TRACKPAR",
                  track::CollisionId, track::X, track::Alpha,
                  track::Y, track::Z, track::Snp, track::Tgl,
                  track::Signed1Pt,
                  track::Phi<track::Snp, track::Alpha>,
                  track::Eta<track::Tgl>,
                  track::Pt<track::Signed1Pt>);

DECLARE_SOA_TABLE(TracksCov, "RN2", "TRACKPARCOV",
                  track::CYY, track::CZY, track::CZZ, track::CSnpY,
                  track::CSnpZ, track::CSnpSnp, track::CTglY,
                  track::CTglZ, track::CTglSnp, track::CTglTgl,
                  track::C1PtY, track::C1PtZ, track::C1PtSnp, track::C1PtTgl,
                  track::C1Pt21Pt2);
DECLARE_SOA_TABLE(TracksExtra, "RN2", "TRACKEXTRA",
                  track::TPCInnerParam, track::Flags, track::ITSClusterMap,
                  track::TPCNCls, track::TRDNTracklets, track::ITSChi2NCl,
                  track::TPCchi2Ncl, track::TRDchi2, track::TOFchi2,
                  track::TPCsignal, track::TRDsignal, track::TOFsignal, track::Lenght);

using Track = Tracks::iterator;
using TrackCov = TracksCov::iterator;
using TrackExtra = TracksExtra::iterator;

namespace calo
{
DECLARE_SOA_COLUMN(CollisionId, collisionId, int32_t, "fID4Collisions");
DECLARE_SOA_COLUMN(CellNumber, cellNumber, int64_t, "fCellNumber");
DECLARE_SOA_COLUMN(Amplitude, amplitude, float, "fAmplitude");
DECLARE_SOA_COLUMN(Time, time, float, "fTime");
DECLARE_SOA_COLUMN(CaloType, caloType, float, "fType");
} // namespace calo

DECLARE_SOA_TABLE(Calos, "RN2", "CALO",
                  calo::CollisionId, calo::CellNumber, calo::Amplitude, calo::Time, calo::CaloType);
using Calo = Calos::iterator;

namespace muon
{
DECLARE_SOA_COLUMN(CollisionId, collisionId, int, "fID4Collisions");
DECLARE_SOA_COLUMN(InverseBendingMomentum, inverseBendingMomentum, float, "fInverseBendingMomentum");
DECLARE_SOA_COLUMN(ThetaX, thetaX, float, "fThetaX");
DECLARE_SOA_COLUMN(ThetaY, thetaY, float, "fThetaY");
DECLARE_SOA_COLUMN(ZMu, zMu, float, "fZmu");
DECLARE_SOA_COLUMN(BendingCoor, bendingCoor, float, "fBendingCoor");
DECLARE_SOA_COLUMN(NonBendingCoor, nonBendingCoor, float, "fNonBendingCoor");
// FIXME: need to implement array columns...
// DECLARE_SOA_COLUMN(Covariances, covariances, float[], "fCovariances");
DECLARE_SOA_COLUMN(Chi2, chi2, float, "fChi2");
DECLARE_SOA_COLUMN(Chi2MatchTrigger, chi2MatchTrigger, float, "fChi2MatchTrigger");
} // namespace muon

DECLARE_SOA_TABLE(Muons, "RN2", "MUON",
                  muon::CollisionId, muon::InverseBendingMomentum,
                  muon::ThetaX, muon::ThetaY, muon::ZMu,
                  muon::BendingCoor, muon::NonBendingCoor,
                  muon::Chi2, muon::Chi2MatchTrigger);
using Muon = Muons::iterator;

namespace vzero
{
DECLARE_SOA_COLUMN(CollisionId, collisionId, int, "fIDvz");
// FIXME: add missing arrays...
} // namespace vzero

DECLARE_SOA_TABLE(VZeros, "RN2", "VZERO", vzero::CollisionId);
using VZero = VZeros::iterator;

namespace collision
{
DECLARE_SOA_COLUMN(TimeframeID, timeframeID, uint64_t, "timeframeID");
DECLARE_SOA_COLUMN(NumTracks, numtracks, uint32_t, "numtracks");
DECLARE_SOA_COLUMN(VtxID, vtxID, int, "vtxID");    // index of the vertex inside the timeframe, AliESDVertex, ushort_t
DECLARE_SOA_COLUMN(PosX, posX, double, "posX");    // vertex x position, AliVertex.h, Double32_t
DECLARE_SOA_COLUMN(PosY, posY, double, "posY");    // vertex y position, AliVertex.h, Double32_t
DECLARE_SOA_COLUMN(PosZ, posZ, double, "posZ");    // vertex z position, AliVertex.h, Double32_t
DECLARE_SOA_COLUMN(CovXX, covXX, double, "covXX"); // vertex covariance matrix element: XX, AliESDVertex.h, Double32_t
DECLARE_SOA_COLUMN(CovXY, covXY, double, "covXY"); // vertex covariance matrix element: XY, AliESDVertex.h, Double32_t
DECLARE_SOA_COLUMN(CovXZ, covXZ, double, "covXZ"); // vertex covariance matrix element: XZ, AliESDVertex.h, Double32_t
DECLARE_SOA_COLUMN(CovYX, covYX, double, "covYX"); // vertex covariance matrix element: YX (same as XY), AliESDVertex.h, Double32_t
DECLARE_SOA_COLUMN(CovYY, covYY, double, "covYY"); // vertex covariance matrix element: YY, AliESDVertex.h, Double32_t
DECLARE_SOA_COLUMN(CovYZ, covYZ, double, "covYZ"); // vertex covariance matrix element: YZ, AliESDVertex.h, Double32_t
DECLARE_SOA_COLUMN(CovZX, covZX, double, "covZX"); // vertex covariance matrix element: ZX (same as XZ), AliESDVertex.h, Double32_t
DECLARE_SOA_COLUMN(CovZY, covZY, double, "covZY"); // vertex covariance matrix element: ZY (same as YZ), AliESDVertex.h, Double32_t
DECLARE_SOA_COLUMN(CovZZ, covZZ, double, "covZZ"); // vertex covariance matrix element: ZZ, AliESDVertex.h, Double32_t
DECLARE_SOA_COLUMN(Chi2, chi2, double, "chi2");    // chi2 of vertex fit, AliESDVertex.h, Double32_t
//DECLARE_SOA_COLUMN(Indices, indices, int*, "indices");                // contributing track IDs to the vertex reconstruction, AliVertex.h, UShort_t*
DECLARE_SOA_COLUMN(BCNum, bcnum, int, "bcnum");          // LHC bunch crossing number, AliESDHeader.h, UShort_t
DECLARE_SOA_COLUMN(OrbitNum, orbitnum, int, "orbitnum"); // LHC orbit number, AliESDHeader, UInt_t
//DECLARE_SOA_COLUMN(PeriodNumber, periodNumber, int, "periodNumber");  // period number (No period number in run 3. Will be included in orbit number.), AliESDHeader, UInt_t
//DECLARE_SOA_COLUMN(V0mult, v0mult, int, "v0mult");                    // V0 multiplicity (redundant as it is also in the detector table)
//DECLARE_SOA_COLUMN(T0multA, t0multA, int, "t0multA");                 // T0 A multiplicity (redundant as it is also in the detector table)
//DECLARE_SOA_COLUMN(T0multC, t0multc, int, "t0multC");                 // T0 C multiplicity (redundant as it is also in the detector table)
//DECLARE_SOA_COLUMN(FITmult, fitmult, int, "fitmult");                 // FIT  multiplicity (redundant as it is also in the detector table)
DECLARE_SOA_COLUMN(TriggerMask, triggermask, int, "triggermask"); // Trigger mask, AliESDHeader.h, ULong64_t
} // namespace collision

DECLARE_SOA_TABLE(Collisions, "RN2", "COLLISION", collision::TimeframeID, collision::NumTracks, collision::VtxID, collision::PosX, collision::PosY, collision::PosZ, collision::CovXX, collision::CovXY, collision::CovXZ, collision::CovYX, collision::CovYY, collision::CovYZ, collision::CovZX, collision::CovZY, collision::CovZZ, collision::Chi2, collision::BCNum, collision::OrbitNum, collision::TriggerMask);

using Collision = Collisions::iterator;

namespace timeframe
{
DECLARE_SOA_COLUMN(Timestamp, timestamp, uint64_t, "timestamp");
} // namespace timeframe

DECLARE_SOA_TABLE(Timeframes, "RN2", "TIMEFRAME",
                  timeframe::Timestamp);
using Timeframe = Timeframes::iterator;

} // namespace aod

} // namespace o2
#endif // O2_FRAMEWORK_ANALYSISDATAMODEL_H_
