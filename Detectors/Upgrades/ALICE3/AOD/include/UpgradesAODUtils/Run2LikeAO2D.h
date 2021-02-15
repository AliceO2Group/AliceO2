// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Run2LikeAO2D.h

//******************************************************************
// AO2D helper information
//
// This header contains basic info needed to mimick a Run 2-like
// converted AO2D file. Its current use case is the ALICE 3
// G3/G4 simulation to AO2D conversion.
//
//******************************************************************
#ifndef RUN2LIKE_AOD_H
#define RUN2LIKE_AOD_H
#include <TString.h>

namespace o2
{
namespace upgrades_utils
{

enum TreeIndex {     // Index of the output trees
  kEvents = 0,       //ok
  kEventsExtra,      //ok
  kTracks,           //ok
  kCalo,             //N/A
  kCaloTrigger,      //N/A
  kMuon,             //N/A
  kMuonCls,          //N/A
  kZdc,              //N/A
  kFV0A,             //N/A
  kFV0C,             //N/A
  kFT0,              //N/A
  kFDD,              //N/A
  kV0s,              //may be ok (requires tuning)
  kCascades,         //may be ok (requires tuning)
  kTOF,              //N/A... for now
  kMcParticle,       //MC operation
  kMcCollision,      //MC operation
  kMcTrackLabel,     //MC operation
  kMcCaloLabel,      //N/A
  kMcCollisionLabel, //MC operation
  kBC,               //N/A
  kTrees             //N/A
};

enum TrackTypeEnum : uint8_t {
  GlobalTrack = 0,
  ITSStandalone,
  MFTStandalone,
  Run2GlobalTrack = 254,
  Run2Tracklet = 255
}; // corresponds to O2/Core/Framework/include/Framework/DataTypes.h
enum TrackFlagsRun2Enum {
  ITSrefit = 0x1,
  TPCrefit = 0x2,
  GoldenChi2 = 0x4
}; // corresponds to O2/Core/Framework/include/Framework/DataTypes.h
enum MCParticleFlags : uint8_t {
  ProducedInTransport = 1 // Bit 0: 0 = from generator; 1 = from transport
};

const TString gTreeName[kTrees] = {"O2collision", "DbgEventExtra", "O2track", "O2calo", "O2calotrigger", "O2muon", "O2muoncluster", "O2zdc", "O2fv0a", "O2fv0c", "O2ft0", "O2fdd", "O2v0", "O2cascade", "O2tof", "O2mcparticle", "O2mccollision", "O2mctracklabel", "O2mccalolabel", "O2mccollisionlabel", "O2bc"};
const TString gTreeTitle[kTrees] = {"Collision tree", "Collision extra", "Barrel tracks", "Calorimeter cells", "Calorimeter triggers", "MUON tracks", "MUON clusters", "ZDC", "FV0A", "FV0C", "FT0", "FDD", "V0s", "Cascades", "TOF hits", "Kinematics", "MC collisions", "MC track labels", "MC calo labels", "MC collision labels", "BC info"};

const Bool_t gSaveTree[kTrees] = {kTRUE, kFALSE, kTRUE, kFALSE, kFALSE, kFALSE, kFALSE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE,
                                  //V0 and cascade (not done for now)
                                  kFALSE, kFALSE,
                                  //TOF
                                  kFALSE,
                                  //MC information (not done for now)
                                  kTRUE, kTRUE, kTRUE, kFALSE, kTRUE, kTRUE};

//+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// structs for AO2D convenience
// straight from AO2D converter
struct {
  // Event data
  Int_t fBCsID = 0u; /// Index to BC table
  // Primary vertex position
  Float_t fPosX = -999.f; /// Primary vertex x coordinate
  Float_t fPosY = -999.f; /// Primary vertex y coordinate
  Float_t fPosZ = -999.f; /// Primary vertex z coordinate
  // Primary vertex covariance matrix
  Float_t fCovXX = 999.f; /// cov[0]
  Float_t fCovXY = 0.f;   /// cov[1]
  Float_t fCovXZ = 0.f;   /// cov[2]
  Float_t fCovYY = 999.f; /// cov[3]
  Float_t fCovYZ = 0.f;   /// cov[4]
  Float_t fCovZZ = 999.f; /// cov[5]
  // Quality parameters
  Float_t fChi2 = 999.f; /// Chi2 of the vertex
  UInt_t fN = 0u;        /// Number of contributors

  // The calculation of event time certainly will be modified in Run3
  // The prototype below can be switched on request
  Float_t fCollisionTime = 10;      /// Event time (t0) obtained with different methods (best, T0, T0-TOF, ...)
  Float_t fCollisionTimeRes = 1e-3; /// Resolution on the event time (t0) obtained with different methods (best, T0, T0-TOF, ...)
  UChar_t fCollisionTimeMask = 0u;  /// Mask with the method used to compute the event time (0x1=T0-TOF,0x2=T0A,0x3=TOC) for each momentum bins

} collision; //! structure to keep the primary vertex (avoid name conflicts)

struct {
  // Start indices and numbers of elements for data in the other trees matching this vertex.
  // Needed for random access of collision-related data, allowing skipping data discarded by the user
  Int_t fStart[kTrees] = {0};    /// Start entry indices for data in the other trees matching this vertex
  Int_t fNentries[kTrees] = {0}; /// Numbers of entries for data in the other trees matching this vertex
} eventextra;                    //! structure for benchmarking information

struct {
  int fRunNumber = -1;         /// Run number
  ULong64_t fGlobalBC = 0u;    /// Unique bunch crossing id. Contains period, orbit and bunch crossing numbers
  ULong64_t fTriggerMask = 0u; /// Trigger class mask
} bc;                          //! structure to keep trigger-related info

struct {
  // Track data

  Int_t fCollisionsID = -1; /// The index of the collision vertex in the TF, to which the track is attached

  uint8_t fTrackType = 0; // Type of track: global, ITS standalone, tracklet, ...

  // In case we need connection to TOF clusters, activate next lines
  // Int_t   fTOFclsIndex;     /// The index of the associated TOF cluster
  // Int_t   fNTOFcls;         /// The number of TOF clusters

  // Coordinate system parameters
  Float_t fX = -999.f;     /// X coordinate for the point of parametrisation
  Float_t fAlpha = -999.f; /// Local <--> global coor.system rotation angle

  // Track parameters
  Float_t fY = -999.f;         /// fP[0] local Y-coordinate of a track (cm)
  Float_t fZ = -999.f;         /// fP[1] local Z-coordinate of a track (cm)
  Float_t fSnp = -999.f;       /// fP[2] local sine of the track momentum azimuthal angle
  Float_t fTgl = -999.f;       /// fP[3] tangent of the track momentum dip angle
  Float_t fSigned1Pt = -999.f; /// fP[4] 1/pt (1/(GeV/c))

  // "Covariance matrix"
  // The diagonal elements represent the errors = Sqrt(C[i,i])
  // The off-diagonal elements are the correlations = C[i,j]/Sqrt(C[i,i])/Sqrt(C[j,j])
  // The off-diagonal elements are multiplied by 128 (7bits) and packed in Char_t
  Float_t fSigmaY = -999.f;   /// Sqrt(fC[0])
  Float_t fSigmaZ = -999.f;   /// Sqrt(fC[2])
  Float_t fSigmaSnp = -999.f; /// Sqrt(fC[5])
  Float_t fSigmaTgl = -999.f; /// Sqrt(fC[9])
  Float_t fSigma1Pt = -999.f; /// Sqrt(fC[14])
  Char_t fRhoZY = 0;          /// 128*fC[1]/SigmaZ/SigmaY
  Char_t fRhoSnpY = 0;        /// 128*fC[3]/SigmaSnp/SigmaY
  Char_t fRhoSnpZ = 0;        /// 128*fC[4]/SigmaSnp/SigmaZ
  Char_t fRhoTglY = 0;        /// 128*fC[6]/SigmaTgl/SigmaY
  Char_t fRhoTglZ = 0;        /// 128*fC[7]/SigmaTgl/SigmaZ
  Char_t fRhoTglSnp = 0;      /// 128*fC[8]/SigmaTgl/SigmaSnp
  Char_t fRho1PtY = 0;        /// 128*fC[10]/Sigma1Pt/SigmaY
  Char_t fRho1PtZ = 0;        /// 128*fC[11]/Sigma1Pt/SigmaZ
  Char_t fRho1PtSnp = 0;      /// 128*fC[12]/Sigma1Pt/SigmaSnp
  Char_t fRho1PtTgl = 0;      /// 128*fC[13]/Sigma1Pt/SigmaTgl

  // Additional track parameters
  Float_t fTPCinnerP = -999.f; /// Full momentum at the inner wall of TPC for dE/dx PID

  // Track quality parameters
  UInt_t fFlags = 0u; /// Reconstruction status flags

  // Clusters and tracklets
  UChar_t fITSClusterMap = 0u;                 /// ITS map of clusters, one bit per a layer
  UChar_t fTPCNClsFindable = 0u;               /// number of clusters that could be assigned in the TPC
  Char_t fTPCNClsFindableMinusFound = 0;       /// difference between foundable and found clusters
  Char_t fTPCNClsFindableMinusCrossedRows = 0; ///  difference between foundable clsuters and crossed rows
  UChar_t fTPCNClsShared = 0u;                 /// Number of shared clusters
  UChar_t fTRDPattern = 0u;                    /// Bit 0-5 if tracklet from TRD layer used for this track

  // Chi2
  Float_t fITSChi2NCl = -999.f; /// chi2/Ncl ITS
  Float_t fTPCChi2NCl = -999.f; /// chi2/Ncl TPC
  Float_t fTRDChi2 = -999.f;    /// chi2 TRD match (?)
  Float_t fTOFChi2 = -999.f;    /// chi2 TOF match (?)

  // PID
  Float_t fTPCSignal = -999.f; /// dE/dX TPC
  Float_t fTRDSignal = -999.f; /// dE/dX TRD
  Float_t fTOFSignal = -999.f; /// TOFsignal
  Float_t fLength = -999.f;    /// Int.Lenght @ TOF
  Float_t fTOFExpMom = -999.f; /// TOF Expected momentum based on the expected time of pions

  // Track extrapolation to EMCAL surface
  Float_t fTrackEtaEMCAL = -999.f; /// Track eta at the EMCAL surface
  Float_t fTrackPhiEMCAL = -999.f; /// Track phi at the EMCAL surface
} tracks;                          //! structure to keep track information

struct {
  // MC collision
  Int_t fBCsID = 0u;          /// Index to BC table
  Short_t fGeneratorsID = 0u; /// Generator ID used for the MC
  Float_t fPosX = -999.f;     /// Primary vertex x coordinate from MC
  Float_t fPosY = -999.f;     /// Primary vertex y coordinate from MC
  Float_t fPosZ = -999.f;     /// Primary vertex z coordinate from MC
  Float_t fT = -999.f;        /// Time of the collision from MC
  Float_t fWeight = -999.f;   /// Weight from MC
  // Generation details (HepMC3 in the future)
  Float_t fImpactParameter = -999.f; /// Impact parameter from MC
} mccollision;                       //! MC collisions = vertices

struct {
  // Track label to find the corresponding MC particle
  UInt_t fLabel = 0;       /// Track label
  UShort_t fLabelMask = 0; /// Bit mask to indicate detector mismatches (bit ON means mismatch)
                           /// Bit 0-6: mismatch at ITS layer
                           /// Bit 7-9: # of TPC mismatches in the ranges 0, 1, 2-3, 4-7, 8-15, 16-31, 32-63, >64
                           /// Bit 10: TRD, bit 11: TOF, bit 15: negative label sign
} mctracklabel;            //! Track labels

struct {
  // MC particle

  Int_t fMcCollisionsID = -1; /// The index of the MC collision vertex

  // MC information (modified version of TParticle
  Int_t fPdgCode = -99999;    /// PDG code of the particle
  Int_t fStatusCode = -99999; /// generation status code
  uint8_t fFlags = 0;         /// See enum MCParticleFlags
  Int_t fMother0 = 0;         /// Indices of the mother particles
  Int_t fMother1 = 0;
  Int_t fDaughter0 = 0; /// Indices of the daughter particles
  Int_t fDaughter1 = 0;
  Float_t fWeight = 1; /// particle weight from the generator or ML

  Float_t fPx = -999.f; /// x component of momentum
  Float_t fPy = -999.f; /// y component of momentum
  Float_t fPz = -999.f; /// z component of momentum
  Float_t fE = -999.f;  /// Energy (covers the case of resonances, no need for calculated mass)

  Float_t fVx = -999.f; /// x of production vertex
  Float_t fVy = -999.f; /// y of production vertex
  Float_t fVz = -999.f; /// z of production vertex
  Float_t fVt = -999.f; /// t of production vertex
  // We do not use the polarisation so far
} mcparticle; //! MC particles from the kinematics tree

struct {
  // MC collision label
  UInt_t fLabel = 0;       /// Collision label
  UShort_t fLabelMask = 0; /// Bit mask to indicate collision mismatches (bit ON means mismatch)
                           /// bit 15: negative label sign
} mccollisionlabel;        //! Collision labels

struct {
  /// FDD (AD)
  Int_t fBCsID = 0u;              /// Index to BC table
  Float_t fAmplitudeA[4] = {0.f}; /// Multiplicity for each A-side channel
  Float_t fAmplitudeC[4] = {0.f}; /// Multiplicity for each C-side channel
  Float_t fTimeA = 56.7f;         /// Average A-side time
  Float_t fTimeC = 65.3f;         /// Average C-side time
  uint8_t fTriggerMask = 0;       /// Trigger info
} fdd;

struct {
  /// V0A  (32 cells in Run2, 48 cells in Run3)
  Int_t fBCsID = 0u;              /// Index to BC table
  Float_t fAmplitude[48] = {0.f}; /// Multiplicity for each channel
  Float_t fTime = 11.f;           /// Average A-side time
  uint8_t fTriggerMask = 0;       /// Trigger info
} fv0a;                           //! structure to keep V0A information

struct {
  /// V0C  (32 cells in Run2)
  Int_t fBCsID = 0u;              /// Index to BC table
  Float_t fAmplitude[32] = {0.f}; /// Multiplicity for each channel
  Float_t fTime = 3.6f;           /// Average C-side time
} fv0c;                           //! structure to keep V0C information

struct {
  /// FT0 (12+12 channels in Run2, 96+112 channels in Run3)
  Int_t fBCsID = 0u;                /// Index to BC table
  Float_t fAmplitudeA[96] = {0.f};  /// Multiplicity for each A-side channel
  Float_t fAmplitudeC[112] = {0.f}; /// Multiplicity for each C-side channel
  Float_t fTimeA = 0.02f;           /// Average A-side time
  Float_t fTimeC = 0.03f;           /// Average C-side time
  uint8_t fTriggerMask = 0;         /// Trigger info
} ft0;                              //! structure to keep FT0 information

struct {
  Int_t fBCsID = 0u;                   /// Index to BC table
  Float_t fEnergyZEM1 = 0.f;           ///< E in ZEM1
  Float_t fEnergyZEM2 = 0.f;           ///< E in ZEM2
  Float_t fEnergyCommonZNA = 0.f;      ///< E in common ZNA PMT - high gain chain
  Float_t fEnergyCommonZNC = 0.f;      ///< E in common ZNC PMT - high gain chain
  Float_t fEnergyCommonZPA = 0.f;      ///< E in common ZPA PMT - high gain chain
  Float_t fEnergyCommonZPC = 0.f;      ///< E in common ZPC PMT - high gain chain
  Float_t fEnergySectorZNA[4] = {0.f}; ///< E in 4 ZNA sectors - high gain chain
  Float_t fEnergySectorZNC[4] = {0.f}; ///< E in 4 ZNC sectors - high gain chain
  Float_t fEnergySectorZPA[4] = {0.f}; ///< E in 4 ZPA sectors - high gain chain
  Float_t fEnergySectorZPC[4] = {0.f}; ///< E in 4 ZPC sectors - high gain chain
  Float_t fTimeZEM1 = 0.f;             ///< Corrected time in ZEM1
  Float_t fTimeZEM2 = 0.f;             ///< Corrected time in ZEM2
  Float_t fTimeZNA = 0.055f;           ///< Corrected time in ZNA
  Float_t fTimeZNC = -0.049f;          ///< Corrected time in ZNC
  Float_t fTimeZPA = 0.f;              ///< Corrected time in ZPA
  Float_t fTimeZPC = 0.f;              ///< Corrected time in ZPC
} zdc;
//! structure to keep ZDC information
} // namespace upgrades_utils
} // namespace o2

#endif