//******************************************************************
// ALICE 3 hits to AO2D converter tool
//
// This macro imports ALICE 3 tracker hits, runs the ITS tracker
// on them and then saves the tracks and primary vertex in a
// AO2D.root file that is compliant with the O2 framework.
//
// More specifically, it mimics Run 2-converted data so that
// any analysis geared towards that can run on the output of this
// conversion tool.
//
// To be run compiled, e.g. root.exe ALICE3toAO2D.C+
// Files to be converted: o2sim_HitsTRK.root, o2sim_Kine.root
// (need to be in the same directory)
//
// Output: AO2D.root (main file)
//         conversion-output.root (basic QA file)
//
// Comments, complaints, suggestions? Please write to:
// --- david.dobrigkeit.chinellato@cern.ch
//******************************************************************


#include <string>
#include <TFile.h>
#include <TChain.h>
#include <TH2D.h>
#include <TProfile.h>
#include <TBranch.h>
#include <TRandom3.h>
#include <TGeoGlobalMagField.h>
#include <vector>
#include <TTimeStamp.h>
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTSimulation/Hit.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/IOUtils.h"
#include "ITStracking/Tracker.h"
#include "ITStracking/TrackerTraitsCPU.h"
#include "ITStracking/Vertexer.h"
#include "ITStracking/VertexerTraits.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "SimulationDataFormat/MCTrack.h"
#include "MathUtils/Cartesian.h"
#include "ReconstructionDataFormats/TrackParametrization.h"
#include "ReconstructionDataFormats/TrackParametrizationWithError.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "DetectorsVertexing/DCAFitterN.h"
#include "ReconstructionDataFormats/DCA.h"
#include "ReconstructionDataFormats/Vertex.h"

using o2::its::MemoryParameters;
using o2::its::TrackingParameters;
using o2::itsmft::Hit;
using std::string;

using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

constexpr bool kUseSmearing{true};

enum TreeIndex { // Index of the output trees
  kEvents = 0,   //ok
  kEventsExtra,  //ok
  kTracks,       //ok
  kCalo,         //N/A
  kCaloTrigger,  //N/A
  kMuon,         //N/A
  kMuonCls,      //N/A
  kZdc,          //N/A
  kFV0A,         //N/A
  kFV0C,         //N/A
  kFT0,          //N/A
  kFDD,          //N/A
  kV0s,          //may be ok (requires tuning)
  kCascades,     //may be ok (requires tuning)
  kTOF,          //N/A... for now
  kMcParticle,   //MC operation
  kMcCollision,  //MC operation
  kMcTrackLabel, //MC operation
  kMcCaloLabel,  //N/A
  kMcCollisionLabel, //MC operation
  kBC,           //N/A
  kTrees         //N/A
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

const TString gTreeName[kTrees] = { "O2collision", "DbgEventExtra", "O2track", "O2calo",  "O2calotrigger", "O2muon", "O2muoncluster", "O2zdc", "O2fv0a", "O2fv0c", "O2ft0", "O2fdd", "O2v0", "O2cascade", "O2tof", "O2mcparticle", "O2mccollision", "O2mctracklabel", "O2mccalolabel", "O2mccollisionlabel", "O2bc" };
const TString gTreeTitle[kTrees] = { "Collision tree", "Collision extra", "Barrel tracks", "Calorimeter cells", "Calorimeter triggers", "MUON tracks", "MUON clusters", "ZDC", "FV0A", "FV0C", "FT0", "FDD", "V0s", "Cascades", "TOF hits", "Kinematics", "MC collisions", "MC track labels", "MC calo labels", "MC collision labels", "BC info" };

const Bool_t gSaveTree[kTrees] = { kTRUE, kFALSE, kTRUE, kFALSE,  kFALSE, kFALSE, kFALSE, kTRUE, kTRUE, kTRUE, kTRUE, kTRUE,
  //V0 and cascade (not done for now)
  kFALSE, kFALSE,
  //TOF
  kFALSE,
  //MC information (not done for now)
  kTRUE, kTRUE, kTRUE, kFALSE, kTRUE, kTRUE };

float getDetLengthFromEta(const float eta, const float radius)
{
  return 10. * (10. + radius * std::cos(2 * std::atan(std::exp(-eta))));
}

//+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// structs for AO2D convenience
// straight from AO2D converter
struct {
  // Event data
  Int_t fBCsID = 0u;       /// Index to BC table
  // Primary vertex position
  Float_t  fPosX = -999.f;       /// Primary vertex x coordinate
  Float_t  fPosY = -999.f;       /// Primary vertex y coordinate
  Float_t  fPosZ = -999.f;       /// Primary vertex z coordinate
  // Primary vertex covariance matrix
  Float_t  fCovXX = 999.f;    /// cov[0]
  Float_t  fCovXY = 0.f;      /// cov[1]
  Float_t  fCovXZ = 0.f;      /// cov[2]
  Float_t  fCovYY = 999.f;    /// cov[3]
  Float_t  fCovYZ = 0.f;      /// cov[4]
  Float_t  fCovZZ = 999.f;    /// cov[5]
  // Quality parameters
  Float_t  fChi2 = 999.f;             /// Chi2 of the vertex
  UInt_t   fN = 0u;                /// Number of contributors

  // The calculation of event time certainly will be modified in Run3
  // The prototype below can be switched on request
  Float_t fCollisionTime = 10;    /// Event time (t0) obtained with different methods (best, T0, T0-TOF, ...)
  Float_t fCollisionTimeRes = 1e-3; /// Resolution on the event time (t0) obtained with different methods (best, T0, T0-TOF, ...)
  UChar_t fCollisionTimeMask = 0u;    /// Mask with the method used to compute the event time (0x1=T0-TOF,0x2=T0A,0x3=TOC) for each momentum bins

} collision; //! structure to keep the primary vertex (avoid name conflicts)

struct {
  // Start indices and numbers of elements for data in the other trees matching this vertex.
  // Needed for random access of collision-related data, allowing skipping data discarded by the user
  Int_t     fStart[kTrees]    = {0}; /// Start entry indices for data in the other trees matching this vertex
  Int_t     fNentries[kTrees] = {0}; /// Numbers of entries for data in the other trees matching this vertex
} eventextra; //! structure for benchmarking information

struct {
  int fRunNumber = -1;         /// Run number
  ULong64_t fGlobalBC = 0u;    /// Unique bunch crossing id. Contains period, orbit and bunch crossing numbers
  ULong64_t fTriggerMask = 0u; /// Trigger class mask
} bc; //! structure to keep trigger-related info

struct {
  // Track data

  Int_t   fCollisionsID = -1;    /// The index of the collision vertex in the TF, to which the track is attached
  
  uint8_t fTrackType = 0;       // Type of track: global, ITS standalone, tracklet, ...
  
  // In case we need connection to TOF clusters, activate next lines
  // Int_t   fTOFclsIndex;     /// The index of the associated TOF cluster
  // Int_t   fNTOFcls;         /// The number of TOF clusters
  
  

  // Coordinate system parameters
  Float_t fX = -999.f;     /// X coordinate for the point of parametrisation
  Float_t fAlpha = -999.f; /// Local <--> global coor.system rotation angle

  // Track parameters
  Float_t fY = -999.f;          /// fP[0] local Y-coordinate of a track (cm)
  Float_t fZ = -999.f;          /// fP[1] local Z-coordinate of a track (cm)
  Float_t fSnp = -999.f;        /// fP[2] local sine of the track momentum azimuthal angle
  Float_t fTgl = -999.f;        /// fP[3] tangent of the track momentum dip angle
  Float_t fSigned1Pt = -999.f;  /// fP[4] 1/pt (1/(GeV/c))

  // "Covariance matrix"
  // The diagonal elements represent the errors = Sqrt(C[i,i])
  // The off-diagonal elements are the correlations = C[i,j]/Sqrt(C[i,i])/Sqrt(C[j,j])
  // The off-diagonal elements are multiplied by 128 (7bits) and packed in Char_t
  Float_t fSigmaY      = -999.f; /// Sqrt(fC[0])
  Float_t fSigmaZ      = -999.f; /// Sqrt(fC[2])
  Float_t fSigmaSnp    = -999.f; /// Sqrt(fC[5])
  Float_t fSigmaTgl    = -999.f; /// Sqrt(fC[9])
  Float_t fSigma1Pt    = -999.f; /// Sqrt(fC[14])
  Char_t fRhoZY        = 0;      /// 128*fC[1]/SigmaZ/SigmaY
  Char_t fRhoSnpY      = 0;      /// 128*fC[3]/SigmaSnp/SigmaY
  Char_t fRhoSnpZ      = 0;      /// 128*fC[4]/SigmaSnp/SigmaZ
  Char_t fRhoTglY      = 0;      /// 128*fC[6]/SigmaTgl/SigmaY
  Char_t fRhoTglZ      = 0;      /// 128*fC[7]/SigmaTgl/SigmaZ
  Char_t fRhoTglSnp    = 0;      /// 128*fC[8]/SigmaTgl/SigmaSnp
  Char_t fRho1PtY      = 0;      /// 128*fC[10]/Sigma1Pt/SigmaY
  Char_t fRho1PtZ      = 0;      /// 128*fC[11]/Sigma1Pt/SigmaZ
  Char_t fRho1PtSnp    = 0;      /// 128*fC[12]/Sigma1Pt/SigmaSnp
  Char_t fRho1PtTgl    = 0;      /// 128*fC[13]/Sigma1Pt/SigmaTgl

  // Additional track parameters
  Float_t fTPCinnerP = -999.f; /// Full momentum at the inner wall of TPC for dE/dx PID

  // Track quality parameters
  UInt_t fFlags = 0u;       /// Reconstruction status flags

  // Clusters and tracklets
  UChar_t fITSClusterMap = 0u;   /// ITS map of clusters, one bit per a layer
  UChar_t fTPCNClsFindable = 0u; /// number of clusters that could be assigned in the TPC
  Char_t fTPCNClsFindableMinusFound = 0;       /// difference between foundable and found clusters
  Char_t fTPCNClsFindableMinusCrossedRows = 0; ///  difference between foundable clsuters and crossed rows
  UChar_t fTPCNClsShared = 0u;   /// Number of shared clusters
  UChar_t fTRDPattern = 0u;   /// Bit 0-5 if tracklet from TRD layer used for this track

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
} tracks;                      //! structure to keep track information

struct {
  // MC collision
  Int_t fBCsID = 0u;       /// Index to BC table
  Short_t fGeneratorsID = 0u; /// Generator ID used for the MC
  Float_t fPosX = -999.f;  /// Primary vertex x coordinate from MC
  Float_t fPosY = -999.f;  /// Primary vertex y coordinate from MC
  Float_t fPosZ = -999.f;  /// Primary vertex z coordinate from MC
  Float_t fT = -999.f;  /// Time of the collision from MC
  Float_t fWeight = -999.f;  /// Weight from MC
  // Generation details (HepMC3 in the future)
  Float_t fImpactParameter = -999.f; /// Impact parameter from MC
} mccollision;  //! MC collisions = vertices

struct {
  // Track label to find the corresponding MC particle
  UInt_t fLabel = 0;       /// Track label
  UShort_t fLabelMask = 0; /// Bit mask to indicate detector mismatches (bit ON means mismatch)
                         /// Bit 0-6: mismatch at ITS layer
                         /// Bit 7-9: # of TPC mismatches in the ranges 0, 1, 2-3, 4-7, 8-15, 16-31, 32-63, >64
                         /// Bit 10: TRD, bit 11: TOF, bit 15: negative label sign
} mctracklabel; //! Track labels

struct {
  // MC particle

  Int_t   fMcCollisionsID = -1;    /// The index of the MC collision vertex

  // MC information (modified version of TParticle
  Int_t fPdgCode    = -99999; /// PDG code of the particle
  Int_t fStatusCode = -99999; /// generation status code
  uint8_t fFlags    = 0;     /// See enum MCParticleFlags
  Int_t fMother0    = 0; /// Indices of the mother particles
  Int_t fMother1    = 0;
  Int_t fDaughter0  = 0; /// Indices of the daughter particles
  Int_t fDaughter1  = 0;
  Float_t fWeight   = 1;     /// particle weight from the generator or ML

  Float_t fPx = -999.f; /// x component of momentum
  Float_t fPy = -999.f; /// y component of momentum
  Float_t fPz = -999.f; /// z component of momentum
  Float_t fE  = -999.f; /// Energy (covers the case of resonances, no need for calculated mass)

  Float_t fVx = -999.f; /// x of production vertex
  Float_t fVy = -999.f; /// y of production vertex
  Float_t fVz = -999.f; /// z of production vertex
  Float_t fVt = -999.f; /// t of production vertex
  // We do not use the polarisation so far
} mcparticle;  //! MC particles from the kinematics tree

struct {
  // MC collision label
  UInt_t fLabel = 0;       /// Collision label
  UShort_t fLabelMask = 0; /// Bit mask to indicate collision mismatches (bit ON means mismatch)
                           /// bit 15: negative label sign
} mccollisionlabel; //! Collision labels

struct {
  /// FDD (AD)
  Int_t fBCsID = 0u;                /// Index to BC table
  Float_t fAmplitudeA[4] = {0.f};   /// Multiplicity for each A-side channel
  Float_t fAmplitudeC[4] = {0.f};   /// Multiplicity for each C-side channel
  Float_t fTimeA = 56.7f;             /// Average A-side time
  Float_t fTimeC = 65.3f;             /// Average C-side time
  uint8_t fTriggerMask = 0;         /// Trigger info
} fdd;

struct {
  /// V0A  (32 cells in Run2, 48 cells in Run3)
  Int_t fBCsID = 0u;                /// Index to BC table
  Float_t fAmplitude[48] = {0.f};   /// Multiplicity for each channel
  Float_t fTime = 11.f;              /// Average A-side time
  uint8_t fTriggerMask = 0;         /// Trigger info
} fv0a;                             //! structure to keep V0A information

struct {
  /// V0C  (32 cells in Run2)
  Int_t fBCsID = 0u;                /// Index to BC table
  Float_t fAmplitude[32] = {0.f};   /// Multiplicity for each channel
  Float_t fTime = 3.6f;              /// Average C-side time
} fv0c;                             //! structure to keep V0C information

struct {
  /// FT0 (12+12 channels in Run2, 96+112 channels in Run3)
  Int_t fBCsID = 0u;                /// Index to BC table
  Float_t fAmplitudeA[96] = {0.f};  /// Multiplicity for each A-side channel
  Float_t fAmplitudeC[112] = {0.f}; /// Multiplicity for each C-side channel
  Float_t fTimeA = 0.02f;             /// Average A-side time
  Float_t fTimeC = 0.03f;             /// Average C-side time
  uint8_t fTriggerMask = 0;         /// Trigger info
} ft0;                              //! structure to keep FT0 information

struct {
  Int_t   fBCsID = 0u;                 /// Index to BC table
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
  Float_t fTimeZNA = 0.055f;              ///< Corrected time in ZNA
  Float_t fTimeZNC = -0.049f;              ///< Corrected time in ZNC
  Float_t fTimeZPA = 0.f;              ///< Corrected time in ZPA
  Float_t fTimeZPC = 0.f;              ///< Corrected time in ZPC
} zdc;                                 //! structure to keep ZDC information

//+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

void ALICE3toAO2D()
{
  std::cout<<"\e[1;31m***********************************************\e[0;00m"<<std::endl;
  std::cout<<"\e[1;31m      ALICE 3 hits to AO2D converter tool \e[0;00m"<<std::endl;
  std::cout<<"\e[1;31m***********************************************\e[0;00m"<<std::endl;
  
  std::cout << "*- Starting..."<<std::endl;
  const string hitsFileName = "o2sim_HitsTRK.root";
  TChain mcTree("o2sim");
  mcTree.AddFile("o2sim_Kine.root");
  mcTree.SetBranchStatus("*", 0); //disable all branches
  mcTree.SetBranchStatus("MCTrack*", 1);
  mcTree.SetBranchStatus("MCEventHeader.", 1);

  std::vector<o2::MCTrack>* mcArr = nullptr;
  mcTree.SetBranchAddress("MCTrack", &mcArr);
  
  //o2::dataformats::MCEventHeader* mcHead;
  //FairMCEventHeader *mcHead;
  auto mcHead = new o2::dataformats::MCEventHeader;
  mcTree.SetBranchAddress("MCEventHeader.", &mcHead);
  
  o2::its::Vertexer vertexer(new o2::its::VertexerTraits());
  
  TChain itsHits("o2sim");

  itsHits.AddFile(hitsFileName.data());

  o2::its::Tracker tracker(new o2::its::TrackerTraitsCPU());
  tracker.setBz(5.f);
  
  std::uint32_t roFrame;
  std::vector<Hit>* hits = nullptr;
  itsHits.SetBranchAddress("TRKHit", &hits);

  std::vector<TrackingParameters> trackParams(4);
  
  trackParams[0].NLayers = 10;
  trackParams[0].MinTrackLength = 10; //this is the one with fixed params
  std::vector<float> LayerRadii = {1.8f, 2.8f, 3.8f, 8.0f, 20.0f, 25.0f, 40.0f, 55.0f, 80.0f, 100.0f};
  std::vector<float> LayerZ(10);
  for (int i{0}; i < 10; ++i)
    LayerZ[i] = getDetLengthFromEta(1.44, LayerRadii[i]) + 1.;
  std::vector<float> TrackletMaxDeltaZ = {0.1f, 0.1f, 0.3f, 0.3f, 0.3f, 0.3f, 0.5f, 0.5f, 0.5f};
  std::vector<float> CellMaxDCA = {0.05f, 0.04f, 0.05f, 0.2f, 0.4f, 0.5f, 0.5f, 0.5f};
  std::vector<float> CellMaxDeltaZ = {0.2f, 0.4f, 0.5f, 0.6f, 3.0f, 3.0f, 3.0f, 3.0f};
  std::vector<float> NeighbourMaxDeltaCurvature = {0.008f, 0.0025f, 0.003f, 0.0035f, 0.004f, 0.004f, 0.005f};
  std::vector<float> NeighbourMaxDeltaN = {0.002f, 0.0090f, 0.002f, 0.005f, 0.005f, 0.005f, 0.005f};

  trackParams[0].LayerRadii = LayerRadii;
  trackParams[0].LayerZ = LayerZ;
  trackParams[0].TrackletMaxDeltaPhi = 0.3;
  trackParams[0].CellMaxDeltaPhi = 0.15;
  trackParams[0].CellMaxDeltaTanLambda = 0.03;
  trackParams[0].TrackletMaxDeltaZ = TrackletMaxDeltaZ;
  trackParams[0].CellMaxDCA = CellMaxDCA;
  trackParams[0].CellMaxDeltaZ = CellMaxDeltaZ;
  trackParams[0].NeighbourMaxDeltaCurvature = NeighbourMaxDeltaCurvature;
  trackParams[0].NeighbourMaxDeltaN = NeighbourMaxDeltaN;
    
  std::vector<MemoryParameters> memParams(4);
  std::vector<float> CellsMemoryCoefficients = {2.3208e-08f * 20, 2.104e-08f * 20, 1.6432e-08f * 20, 1.2412e-08f * 20, 1.3543e-08f * 20, 1.5e-08f * 20, 1.6e-08f * 20, 1.7e-08f * 20};
  std::vector<float> TrackletsMemoryCoefficients = {0.0016353f * 1000, 0.0013627f * 1000, 0.000984f * 1000, 0.00078135f * 1000, 0.00057934f * 1000, 0.00052217f * 1000, 0.00052217f * 1000, 0.00052217f * 1000, 0.00052217f * 1000};
  memParams[0].CellsMemoryCoefficients = CellsMemoryCoefficients;
  memParams[0].TrackletsMemoryCoefficients = TrackletsMemoryCoefficients;
  memParams[0].MemoryOffset = 8000;

  float loosening = 3.;
  for (int i = 1; i < 4; ++i) {
    memParams[i] = memParams[i - 1];
    trackParams[i] = trackParams[i - 1];
    // trackParams[i].MinTrackLength -= 2;
    trackParams[i].TrackletMaxDeltaPhi = trackParams[i].TrackletMaxDeltaPhi * 3 > TMath::Pi() ? TMath::Pi() : trackParams[i].TrackletMaxDeltaPhi * 3;
    trackParams[i].CellMaxDeltaPhi = trackParams[i].CellMaxDeltaPhi * 3 > TMath::Pi() ? TMath::Pi() : trackParams[i].CellMaxDeltaPhi * 3;
    trackParams[i].CellMaxDeltaTanLambda *= loosening;
    for (auto& val : trackParams[i].TrackletMaxDeltaZ)
      val *= loosening;
    for (auto& val : trackParams[i].CellMaxDCA)
      val *= loosening;
    for (auto& val : trackParams[i].CellMaxDeltaZ)
      val *= loosening;
    for (auto& val : trackParams[i].NeighbourMaxDeltaCurvature)
      val *= loosening;
    for (auto& val : trackParams[i].NeighbourMaxDeltaN)
      val *= loosening;
  }

  tracker.setParameters(memParams, trackParams);
  
  constexpr int nBins = 100;
  constexpr float minPt = 0.01;
  constexpr float maxPt = 10;
  double newBins[nBins + 1];
  newBins[0] = minPt;
  double factor = pow(maxPt / minPt, 1. / nBins);
  for (int i = 1; i <= nBins; i++) {
    newBins[i] = factor * newBins[i - 1];
  }
  
  Double_t ptbinlimits[] ={ 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8, 0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.,2.2,2.4,2.6,2.8,3.0,
  3.3,3.6,3.9,4.2,4.6,5,5.4,5.9, 6.5,7,7.5,8,8.5,9.2,10,11,12,13.5,15,17,20};
  Long_t ptbinnumb = sizeof(ptbinlimits)/sizeof(Double_t) - 1;
  
  //Debug output
  TH1D *hNVertices = new TH1D("hNVertices", "", 10,0,10);
  TH1D *hNTracks = new TH1D("hNTracks", "", 100,0,100);
  
  TH1D *hPtSpectra = new TH1D("hPtSpectra", "", ptbinnumb,ptbinlimits);
  TH1D *hPtSpectraFake = new TH1D("hPtSpectraFake", "", ptbinnumb,ptbinlimits);

  //Define o2 fitter, 2-prong
  o2::vertexing::DCAFitterN<2> fitterV0, fitterCasc, fitterCascC;
  fitterV0.setBz(5);
  fitterV0.setPropagateToPCA(true);
  fitterV0.setMaxR(200.);
  fitterV0.setMinParamChange(1e-5);
  fitterV0.setMinRelChi2Change(0.9);
  fitterV0.setMaxDZIni(1e9);
  fitterV0.setMaxChi2(1e9);
  fitterV0.setUseAbsDCA(true);
  
  //+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  std::cout << "*- Setting up output file..."<<std::endl;
  // Setup output
  UInt_t fCompress = 101;
  int fBasketSizeEvents = 1000000;   // Maximum basket size of the trees for events
  int fBasketSizeTracks = 10000000;   // Maximum basket size of the trees for tracks
  TFile *fOutputFile = TFile::Open("AO2D.root","RECREATE", "O2 AOD", fCompress); // File to store the trees of time frames
  
  //setup timestamp for output
  TTimeStamp ts0(2020,11,1,0,0,0);
  TTimeStamp ts1;
  UInt_t tfId = ts1.GetSec() - ts0.GetSec();
  
  // Create the output directory for the current time frame
  TDirectory * fOutputDir = 0x0; ///! Pointer to the output Root subdirectory
  fOutputDir = fOutputFile->mkdir(Form("TF_%d", tfId));
  fOutputDir->cd();
  
  //Create output trees in file
  TTree *fTree[kTrees];
  for(Int_t ii=0; ii<kTrees; ii++){
    if(gSaveTree[ii]){
      std::cout << "*- Creating tree "<<gTreeName[ii]<<"..."<<std::endl;
      fTree[ii] = new TTree(gTreeName[ii], gTreeTitle[ii]);
      fTree[ii]->SetAutoFlush(0);
    }
  }
  if(gSaveTree[kEvents]){
    fTree[kEvents]->Branch("fBCsID", &collision.fBCsID, "fBCsID/I");
    fTree[kEvents]->Branch("fPosX", &collision.fPosX, "fPosX/F");
    fTree[kEvents]->Branch("fPosY", &collision.fPosY, "fPosY/F");
    fTree[kEvents]->Branch("fPosZ", &collision.fPosZ, "fPosZ/F");
    fTree[kEvents]->Branch("fCovXX", &collision.fCovXX, "fCovXX/F");
    fTree[kEvents]->Branch("fCovXY", &collision.fCovXY, "fCovXY/F");
    fTree[kEvents]->Branch("fCovXZ", &collision.fCovXZ, "fCovXZ/F");
    fTree[kEvents]->Branch("fCovYY", &collision.fCovYY, "fCovYY/F");
    fTree[kEvents]->Branch("fCovYZ", &collision.fCovYZ, "fCovYZ/F");
    fTree[kEvents]->Branch("fCovZZ", &collision.fCovZZ, "fCovZZ/F");
    fTree[kEvents]->Branch("fChi2", &collision.fChi2, "fChi2/F");
    fTree[kEvents]->Branch("fNumContrib", &collision.fN, "fNumContrib/i");
    fTree[kEvents]->Branch("fCollisionTime", &collision.fCollisionTime, "fCollisionTime/F");
    fTree[kEvents]->Branch("fCollisionTimeRes", &collision.fCollisionTimeRes, "fCollisionTimeRes/F");
    fTree[kEvents]->Branch("fCollisionTimeMask", &collision.fCollisionTimeMask, "fCollisionTimeMask/b");
    fTree[kEvents]->SetBasketSize("*", fBasketSizeEvents);
  }
  
  if(gSaveTree[kTracks]){
    fTree[kTracks]->Branch("fCollisionsID", &tracks.fCollisionsID, "fCollisionsID/I");
    fTree[kTracks]->Branch("fTrackType", &tracks.fTrackType, "fTrackType/b");
    //    fTree[kTracks]->Branch("fTOFclsIndex", &tracks.fTOFclsIndex, "fTOFclsIndex/I");
    //    fTree[kTracks]->Branch("fNTOFcls", &tracks.fNTOFcls, "fNTOFcls/I");
    fTree[kTracks]->Branch("fX", &tracks.fX, "fX/F");
    fTree[kTracks]->Branch("fAlpha", &tracks.fAlpha, "fAlpha/F");
    fTree[kTracks]->Branch("fY", &tracks.fY, "fY/F");
    fTree[kTracks]->Branch("fZ", &tracks.fZ, "fZ/F");
    fTree[kTracks]->Branch("fSnp", &tracks.fSnp, "fSnp/F");
    fTree[kTracks]->Branch("fTgl", &tracks.fTgl, "fTgl/F");
    fTree[kTracks]->Branch("fSigned1Pt", &tracks.fSigned1Pt, "fSigned1Pt/F");
    // Modified covariance matrix
    fTree[kTracks]->Branch("fSigmaY", &tracks.fSigmaY, "fSigmaY/F");
    fTree[kTracks]->Branch("fSigmaZ", &tracks.fSigmaZ, "fSigmaZ/F");
    fTree[kTracks]->Branch("fSigmaSnp", &tracks.fSigmaSnp, "fSigmaSnp/F");
    fTree[kTracks]->Branch("fSigmaTgl", &tracks.fSigmaTgl, "fSigmaTgl/F");
    fTree[kTracks]->Branch("fSigma1Pt", &tracks.fSigma1Pt, "fSigma1Pt/F");
    fTree[kTracks]->Branch("fRhoZY", &tracks.fRhoZY, "fRhoZY/B");
    fTree[kTracks]->Branch("fRhoSnpY", &tracks.fRhoSnpY, "fRhoSnpY/B");
    fTree[kTracks]->Branch("fRhoSnpZ", &tracks.fRhoSnpZ, "fRhoSnpZ/B");
    fTree[kTracks]->Branch("fRhoTglY", &tracks.fRhoTglY, "fRhoTglY/B");
    fTree[kTracks]->Branch("fRhoTglZ", &tracks.fRhoTglZ, "fRhoTglZ/B");
    fTree[kTracks]->Branch("fRhoTglSnp", &tracks.fRhoTglSnp, "fRhoTglSnp/B");
    fTree[kTracks]->Branch("fRho1PtY", &tracks.fRho1PtY, "fRho1PtY/B");
    fTree[kTracks]->Branch("fRho1PtZ", &tracks.fRho1PtZ, "fRho1PtZ/B");
    fTree[kTracks]->Branch("fRho1PtSnp", &tracks.fRho1PtSnp, "fRho1PtSnp/B");
    fTree[kTracks]->Branch("fRho1PtTgl", &tracks.fRho1PtTgl, "fRho1PtTgl/B");
    //
    fTree[kTracks]->Branch("fTPCInnerParam", &tracks.fTPCinnerP, "fTPCInnerParam/F");
    fTree[kTracks]->Branch("fFlags", &tracks.fFlags, "fFlags/i");
    fTree[kTracks]->Branch("fITSClusterMap", &tracks.fITSClusterMap, "fITSClusterMap/b");
    fTree[kTracks]->Branch("fTPCNClsFindable", &tracks.fTPCNClsFindable, "fTPCNClsFindable/b");
    fTree[kTracks]->Branch("fTPCNClsFindableMinusFound",&tracks.fTPCNClsFindableMinusFound, "fTPCNClsFindableMinusFound/B");
    fTree[kTracks]->Branch("fTPCNClsFindableMinusCrossedRows", &tracks.fTPCNClsFindableMinusCrossedRows, "fTPCNClsFindableMinusCrossedRows/B");
    fTree[kTracks]->Branch("fTPCNClsShared", &tracks.fTPCNClsShared, "fTPCNClsShared/b");
    fTree[kTracks]->Branch("fTRDPattern", &tracks.fTRDPattern, "fTRDPattern/b");
    fTree[kTracks]->Branch("fITSChi2NCl", &tracks.fITSChi2NCl, "fITSChi2NCl/F");
    fTree[kTracks]->Branch("fTPCChi2NCl", &tracks.fTPCChi2NCl, "fTPCChi2NCl/F");
    fTree[kTracks]->Branch("fTRDChi2", &tracks.fTRDChi2, "fTRDChi2/F");
    fTree[kTracks]->Branch("fTOFChi2", &tracks.fTOFChi2, "fTOFChi2/F");
    fTree[kTracks]->Branch("fTPCSignal", &tracks.fTPCSignal, "fTPCSignal/F");
    fTree[kTracks]->Branch("fTRDSignal", &tracks.fTRDSignal, "fTRDSignal/F");
    fTree[kTracks]->Branch("fTOFSignal", &tracks.fTOFSignal, "fTOFSignal/F");
    fTree[kTracks]->Branch("fLength", &tracks.fLength, "fLength/F");
    fTree[kTracks]->Branch("fTOFExpMom", &tracks.fTOFExpMom, "fTOFExpMom/F");
    fTree[kTracks]->Branch("fTrackEtaEMCAL", &tracks.fTrackEtaEMCAL, "fTrackEtaEMCAL/F");
    fTree[kTracks]->Branch("fTrackPhiEMCAL", &tracks.fTrackPhiEMCAL, "fTrackPhiEMCAL/F");
    fTree[kTracks]->SetBasketSize("*", fBasketSizeTracks);
  }
  
  if (gSaveTree[kMcTrackLabel]) {
    fTree[kMcTrackLabel]->Branch("fLabel", &mctracklabel.fLabel, "fLabel/i");
    fTree[kMcTrackLabel]->Branch("fLabelMask", &mctracklabel.fLabelMask, "fLabelMask/s");
    fTree[kMcTrackLabel]->SetBasketSize("*", fBasketSizeTracks);
  }
  
  if(gSaveTree[kMcCollision]) {
    fTree[kMcCollision]->Branch("fBCsID", &mccollision.fBCsID, "fBCsID/I");
    fTree[kMcCollision]->Branch("fGeneratorsID", &mccollision.fGeneratorsID, "fGeneratorsID/S");
    fTree[kMcCollision]->Branch("fPosX", &mccollision.fPosX, "fPosX/F");
    fTree[kMcCollision]->Branch("fPosY", &mccollision.fPosY, "fPosY/F");
    fTree[kMcCollision]->Branch("fPosZ", &mccollision.fPosZ, "fPosZ/F");
    fTree[kMcCollision]->Branch("fT", &mccollision.fT, "fT/F");
    fTree[kMcCollision]->Branch("fWeight", &mccollision.fWeight, "fWeight/F");
    fTree[kMcCollision]->Branch("fImpactParameter", &mccollision.fImpactParameter, "fImpactParameter/F");
    fTree[kMcCollision]->SetBasketSize("*", fBasketSizeEvents);
  }
  
  if (gSaveTree[kMcParticle]) {
    fTree[kMcParticle]->Branch("fMcCollisionsID", &mcparticle.fMcCollisionsID, "fMcCollisionsID/I");
    fTree[kMcParticle]->Branch("fPdgCode", &mcparticle.fPdgCode, "fPdgCode/I");
    fTree[kMcParticle]->Branch("fStatusCode", &mcparticle.fStatusCode, "fStatusCode/I");
    fTree[kMcParticle]->Branch("fFlags", &mcparticle.fFlags, "fFlags/b");
    fTree[kMcParticle]->Branch("fMother0", &mcparticle.fMother0, "fMother0/I");
    fTree[kMcParticle]->Branch("fMother1", &mcparticle.fMother1, "fMother1/I");
    fTree[kMcParticle]->Branch("fDaughter0", &mcparticle.fDaughter0, "fDaughter0/I");
    fTree[kMcParticle]->Branch("fDaughter1", &mcparticle.fDaughter1, "fDaughter1/I");
    fTree[kMcParticle]->Branch("fWeight", &mcparticle.fWeight, "fWeight/F");
    
    fTree[kMcParticle]->Branch("fPx", &mcparticle.fPx, "fPx/F");
    fTree[kMcParticle]->Branch("fPy", &mcparticle.fPy, "fPy/F");
    fTree[kMcParticle]->Branch("fPz", &mcparticle.fPz, "fPz/F");
    fTree[kMcParticle]->Branch("fE", &mcparticle.fE, "fE/F");
    
    fTree[kMcParticle]->Branch("fVx", &mcparticle.fVx, "fVx/F");
    fTree[kMcParticle]->Branch("fVy", &mcparticle.fVy, "fVy/F");
    fTree[kMcParticle]->Branch("fVz", &mcparticle.fVz, "fVz/F");
    fTree[kMcParticle]->Branch("fVt", &mcparticle.fVt, "fVt/F");
    fTree[kMcParticle]->SetBasketSize("*", fBasketSizeTracks);
  }
  
  if (gSaveTree[kMcCollisionLabel]) {
    fTree[kMcCollisionLabel]->Branch("fLabel", &mccollisionlabel.fLabel, "fLabel/i");
    fTree[kMcCollisionLabel]->Branch("fLabelMask", &mccollisionlabel.fLabelMask, "fLabelMask/s");
    fTree[kMcCollisionLabel]->SetBasketSize("*", fBasketSizeEvents);
  }
  
  if (gSaveTree[kBC]) {
    fTree[kBC]->Branch("fRunNumber", &bc.fRunNumber, "fRunNumber/I");
    fTree[kBC]->Branch("fGlobalBC", &bc.fGlobalBC, "fGlobalBC/l");
    fTree[kBC]->Branch("fTriggerMask", &bc.fTriggerMask, "fTriggerMask/l");
    fTree[kBC]->SetBasketSize("*", fBasketSizeEvents);
  }
  
  if (gSaveTree[kFDD]) {
    fTree[kFDD]->Branch("fBCsID", &fdd.fBCsID, "fBCsID/I");
    fTree[kFDD]->Branch("fAmplitudeA", fdd.fAmplitudeA, "fAmplitudeA[4]/F");
    fTree[kFDD]->Branch("fAmplitudeC", fdd.fAmplitudeC, "fAmplitudeC[4]/F");
    fTree[kFDD]->Branch("fTimeA", &fdd.fTimeA, "fTimeA/F");
    fTree[kFDD]->Branch("fTimeC", &fdd.fTimeC, "fTimeC/F");
    fTree[kFDD]->Branch("fTriggerMask", &fdd.fTriggerMask, "fTriggerMask/b");
    fTree[kFDD]->SetBasketSize("*", fBasketSizeEvents);
  }
  
  // Associate branches for V0A
  if (gSaveTree[kFV0A]) {
    fTree[kFV0A]->Branch("fBCsID", &fv0a.fBCsID, "fBCsID/I");
    fTree[kFV0A]->Branch("fAmplitude", fv0a.fAmplitude, "fAmplitude[48]/F");
    fTree[kFV0A]->Branch("fTime", &fv0a.fTime, "fTime/F");
    fTree[kFV0A]->Branch("fTriggerMask", &fv0a.fTriggerMask, "fTriggerMask/b");
    fTree[kFV0A]->SetBasketSize("*", fBasketSizeEvents);
  }

  // Associate branches for V0C
  if (gSaveTree[kFV0C]) {
    fTree[kFV0C]->Branch("fBCsID", &fv0c.fBCsID, "fBCsID/I");
    fTree[kFV0C]->Branch("fAmplitude", fv0c.fAmplitude, "fAmplitude[32]/F");
    fTree[kFV0C]->Branch("fTime", &fv0c.fTime, "fTime/F");
    fTree[kFV0C]->SetBasketSize("*", fBasketSizeEvents);
  }

  // Associate branches for FT0
  if (gSaveTree[kFT0]) {
    fTree[kFT0]->Branch("fBCsID", &ft0.fBCsID, "fBCsID/I");
    fTree[kFT0]->Branch("fAmplitudeA", ft0.fAmplitudeA, "fAmplitudeA[96]/F");
    fTree[kFT0]->Branch("fAmplitudeC", ft0.fAmplitudeC, "fAmplitudeC[112]/F");
    fTree[kFT0]->Branch("fTimeA", &ft0.fTimeA, "fTimeA/F");
    fTree[kFT0]->Branch("fTimeC", &ft0.fTimeC, "fTimeC/F");
    fTree[kFT0]->Branch("fTriggerMask", &ft0.fTriggerMask, "fTriggerMask/b");
    fTree[kFT0]->SetBasketSize("*", fBasketSizeEvents);
  }
  
  if (gSaveTree[kZdc]) {
    fTree[kZdc]->Branch("fBCsID",           &zdc.fBCsID          , "fBCsID/I");
    fTree[kZdc]->Branch("fEnergyZEM1",      &zdc.fEnergyZEM1     , "fEnergyZEM1/F");
    fTree[kZdc]->Branch("fEnergyZEM2",      &zdc.fEnergyZEM2     , "fEnergyZEM2/F");
    fTree[kZdc]->Branch("fEnergyCommonZNA", &zdc.fEnergyCommonZNA, "fEnergyCommonZNA/F");
    fTree[kZdc]->Branch("fEnergyCommonZNC", &zdc.fEnergyCommonZNC, "fEnergyCommonZNC/F");
    fTree[kZdc]->Branch("fEnergyCommonZPA", &zdc.fEnergyCommonZPA, "fEnergyCommonZPA/F");
    fTree[kZdc]->Branch("fEnergyCommonZPC", &zdc.fEnergyCommonZPC, "fEnergyCommonZPC/F");
    fTree[kZdc]->Branch("fEnergySectorZNA", &zdc.fEnergySectorZNA, "fEnergySectorZNA[4]/F");
    fTree[kZdc]->Branch("fEnergySectorZNC", &zdc.fEnergySectorZNC, "fEnergySectorZNC[4]/F");
    fTree[kZdc]->Branch("fEnergySectorZPA", &zdc.fEnergySectorZPA, "fEnergySectorZPA[4]/F");
    fTree[kZdc]->Branch("fEnergySectorZPC", &zdc.fEnergySectorZPC, "fEnergySectorZPC[4]/F");
    fTree[kZdc]->Branch("fTimeZEM1",        &zdc.fTimeZEM1       , "fTimeZEM1/F");
    fTree[kZdc]->Branch("fTimeZEM2",        &zdc.fTimeZEM2       , "fTimeZEM2/F");
    fTree[kZdc]->Branch("fTimeZNA",         &zdc.fTimeZNA        , "fTimeZNA/F");
    fTree[kZdc]->Branch("fTimeZNC",         &zdc.fTimeZNC        , "fTimeZNC/F");
    fTree[kZdc]->Branch("fTimeZPA",         &zdc.fTimeZPA        , "fTimeZPA/F");
    fTree[kZdc]->Branch("fTimeZPC",         &zdc.fTimeZPC        , "fTimeZPC/F");
    fTree[kZdc]->SetBasketSize("*", fBasketSizeEvents);
  }

  //+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  Long_t lGoodEvents=0;
  Long_t fOffsetLabel=0;
  std::cout << "*- Number of events detected: " <<   itsHits.GetEntries() << std::endl;
  for (int iEvent{0}; iEvent < itsHits.GetEntriesFast(); ++iEvent) {
    itsHits.GetEntry(iEvent);
    mcTree.GetEvent(iEvent);
    o2::its::ROframe event{iEvent, 10};
    std::cout << "*- Processing event " << iEvent << "..." << std::endl;
    
    int id{0};
    
    for (auto& hit : *hits) {
      const int layer{hit.GetDetectorID()};
      float xyz[3]{hit.GetX(), hit.GetY(), hit.GetZ()};
      float r{std::hypot(xyz[0], xyz[1])};
      float phi{std::atan2(-xyz[1], -xyz[0]) + o2::its::constants::math::Pi};

      if (kUseSmearing) {
        phi = gRandom->Gaus(phi, std::asin(0.0005f / r));
        xyz[0] = r * std::cos(phi);
        xyz[1] = r * std::sin(phi);
        xyz[2] = gRandom->Gaus(xyz[2], 0.0005f);
      }
    
      event.addTrackingFrameInfoToLayer(layer, xyz[0], xyz[1], xyz[2], r, phi, std::array<float, 2>{0.f, xyz[2]},
                                        std::array<float, 3>{0.0005f * 0.0005f, 0.f, 0.0005f * 0.0005f});
      event.addClusterToLayer(layer, xyz[0], xyz[1], xyz[2], event.getClustersOnLayer(layer).size());
      event.addClusterLabelToLayer(layer, o2::MCCompLabel(hit.GetTrackID(), iEvent, iEvent, false));
      event.addClusterExternalIndexToLayer(layer, id++);
    }
    roFrame = iEvent;
    std::cout << "*- Event " << iEvent << " finished adding hits." << std::endl;
        
    vertexer.clustersToVertices(event);
    
    std::vector<Vertex> vertices = vertexer.exportVertices();
    std::cout<<"*- Number of vertices found: "<<vertices.size()<<endl;
    hNVertices->Fill(vertices.size());
    //Skip events with no vertex
    if(vertices.size()==0){
      std::cout <<"*- No primary vertex found, skipping event"<<std::endl;
      continue;
    }
    
    o2::math_utils::Point3D<float> pos{vertices[0].getX(),vertices[0].getY(),vertices[0].getZ()};
    std::array<float, 6> cov;
    for(Int_t jj=0; jj<6; jj++) cov[jj]=vertices[0].getCov()[jj];
    o2::dataformats::VertexBase vtx(pos, cov);
    o2::dataformats::DCA dca;
    
    //+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    std::cout <<"*- Acquiring collision information..."<<std::endl;
    //---> Collision data
    Long_t lEventNumber = lGoodEvents;
    collision.fBCsID = fTree[kEvents]->GetEntries();
    fdd.fBCsID = collision.fBCsID;
    ft0.fBCsID = collision.fBCsID;
    fv0a.fBCsID = collision.fBCsID;
    fv0c.fBCsID = collision.fBCsID;
    zdc.fBCsID = collision.fBCsID;
    collision.fPosX = vertices[0].getX();
    collision.fPosY = vertices[0].getY();
    collision.fPosZ = vertices[0].getZ();
    collision.fCovXX = cov[0];
    collision.fCovXY = cov[1];
    collision.fCovXZ = cov[2];
    collision.fCovYY = cov[3];
    collision.fCovYZ = cov[4];
    collision.fCovZZ = cov[5];
    collision.fChi2 = vertices[0].getChi2();
    collision.fN = vertices[0].getNContributors();
    collision.fCollisionTime = 10;
    collision.fCollisionTimeRes = 1e-6;
    ft0.fTimeA = 10;
    ft0.fTimeC = 10;
    
    //---> MC collision data
    mccollision.fBCsID = lGoodEvents;
    if(!mcHead){
      std::cout <<"\e[1;31m*- Problem with MC header! \e[0;00m"<<std::endl;
      return;
    }
    mccollision.fPosX = mcHead->GetX();
    mccollision.fPosY = mcHead->GetY();
    mccollision.fPosZ = mcHead->GetZ();
    mccollision.fT = mcHead->GetT();
    mccollision.fWeight = 1;
    mccollision.fImpactParameter = mcHead->GetB();
    
    mccollisionlabel.fLabel = lGoodEvents;
    mccollisionlabel.fLabelMask = 0;
    //---> Save fake hits on i-th layer for track
    
    
    //---> Dummy trigger mask to ensure nobody rejects this
    bc.fTriggerMask = 0;
    for( Int_t iii=0; iii<60; iii++) bc.fTriggerMask |= 1ull << iii;
    bc.fRunNumber = 246087; //ah, the good old days
    
    std::cout << "*- Event " << iEvent << " tracking" << std::endl;
    tracker.clustersToTracks(event);
    auto& lTracks = tracker.getTracks();
    auto& lTracksLabels = tracker.getTrackLabels();
    std::cout << "*- Event " << iEvent << " done tracking!" << std::endl;
    hNTracks -> Fill(lTracks.size());
    
    //+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    //---> Track data
    Long_t lNTracks = lTracks.size();
    for (Int_t i = 0; i < lNTracks; i++)
    {
      //get result from tracker
      auto& lab = lTracksLabels[i];
      auto& track = lTracks[i];
      int trackID = std::abs(lab.getTrackID());
     
      //Propagate to primary vertex as usual
      o2::dataformats::DCA dca1;
      if (!track.propagateToDCA(vtx, tracker.getBz(), &dca1)) {
        std::cout << "Track propagation to primary vertex failed." << std::endl;
      }
      
      //Fill QA histograms
      hPtSpectra->Fill(track.getPt());
      if(lab.isFake()) hPtSpectraFake->Fill(track.getPt());
      
      tracks.fCollisionsID = lEventNumber;
      tracks.fTrackType = TrackTypeEnum::Run2GlobalTrack; //Make track selection happy, please
      tracks.fFlags = 0x0;
      //Assume it all worked, fool regular selections
      tracks.fFlags |= TrackFlagsRun2Enum::ITSrefit;
      tracks.fFlags |= TrackFlagsRun2Enum::TPCrefit;
      tracks.fFlags |= TrackFlagsRun2Enum::GoldenChi2;
      
      //Main: X, alpha, track params
      tracks.fX = track.getX();
      tracks.fY = track.getY();
      tracks.fZ = track.getZ();
      tracks.fAlpha = track.getAlpha();
      tracks.fSnp = track.getSnp();
      tracks.fTgl = track.getTgl();
      tracks.fSigned1Pt = track.getQ2Pt();
            
      // diagonal elements of covariance matrix
      tracks.fSigmaY = TMath::Sqrt(track.getSigmaY2());
      tracks.fSigmaZ = TMath::Sqrt(track.getSigmaZ2());
      tracks.fSigmaSnp = TMath::Sqrt(track.getSigmaSnp2());
      tracks.fSigmaTgl = TMath::Sqrt(track.getSigmaTgl2());
      tracks.fSigma1Pt = TMath::Sqrt(track.getSigma1Pt2());
      // off-diagonal elements of covariance matrix
      tracks.fRhoZY = (Char_t)(128.*track.getSigmaZY()/tracks.fSigmaZ/tracks.fSigmaY);
      tracks.fRhoSnpY = (Char_t)(128.*track.getSigmaSnpY()/tracks.fSigmaSnp/tracks.fSigmaY);
      tracks.fRhoSnpZ = (Char_t)(128.*track.getSigmaSnpZ()/tracks.fSigmaSnp/tracks.fSigmaZ);
      tracks.fRhoTglY = (Char_t)(128.*track.getSigmaTglY()/tracks.fSigmaTgl/tracks.fSigmaY);
      tracks.fRhoTglZ = (Char_t)(128.*track.getSigmaTglZ()/tracks.fSigmaTgl/tracks.fSigmaZ);
      tracks.fRhoTglSnp = (Char_t)(128.*track.getSigmaTglSnp()/tracks.fSigmaTgl/tracks.fSigmaSnp);
      tracks.fRho1PtY = (Char_t)(128.*track.getSigma1PtY()/tracks.fSigma1Pt/tracks.fSigmaY);
      tracks.fRho1PtZ = (Char_t)(128.*track.getSigma1PtZ()/tracks.fSigma1Pt/tracks.fSigmaZ);
      tracks.fRho1PtSnp = (Char_t)(128.*track.getSigma1PtSnp()/tracks.fSigma1Pt/tracks.fSigmaSnp);
      tracks.fRho1PtTgl = (Char_t)(128.*track.getSigma1PtTgl()/tracks.fSigma1Pt/tracks.fSigmaTgl);

      //insist it's good
      tracks.fITSChi2NCl = 1.0;
      tracks.fTPCChi2NCl = 1.0;
      tracks.fTPCNClsFindable = (UChar_t)(120);
      tracks.fTPCNClsFindableMinusFound = (Char_t)(0);
      tracks.fTPCNClsFindableMinusCrossedRows = (Char_t)(0);
      UChar_t fITSClusterMap = 0u;
      fITSClusterMap |= 0x1<<0; // flag manually
      fITSClusterMap |= 0x1<<1; // flag manually
      tracks.fITSClusterMap = fITSClusterMap; 
      
      //MC labels for MC use - negative, yes, but negative with offset
      mctracklabel.fLabel = TMath::Abs(lab.getTrackID()) + fOffsetLabel;
      mctracklabel.fLabelMask = 0;
      //Tag as fake. Note: used first bit only.
      if(lab.isFake()) mctracklabel.fLabelMask = 1;
            
      fTree[kTracks]->Fill();
      fTree[kMcTrackLabel]->Fill();
    }
    //+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    //---> MC stack information for de-referencing
    for (Long_t iii=0; iii< (Long_t) mcArr->size(); iii++ ){
      auto part = mcArr->at(iii);

      mcparticle.fMcCollisionsID = lGoodEvents;
      
      //Get the kinematic values of the particles
      mcparticle.fPdgCode = part.GetPdgCode();
      mcparticle.fStatusCode = part.isPrimary();
      
      mcparticle.fFlags = 0;
      if (part.isSecondary())
        mcparticle.fFlags |= MCParticleFlags::ProducedInTransport;
      
      mcparticle.fMother0 = part.getMotherTrackId();
      if (mcparticle.fMother0 > -1) mcparticle.fMother0+=fOffsetLabel;
      mcparticle.fMother1 = -1;
      mcparticle.fDaughter0 = part.getFirstDaughterTrackId();
      if (mcparticle.fDaughter0 > -1) mcparticle.fDaughter0+=fOffsetLabel;
      mcparticle.fDaughter1 = part.getLastDaughterTrackId();
      if (mcparticle.fDaughter1 > -1) mcparticle.fDaughter1+=fOffsetLabel;
      mcparticle.fWeight = 1;

      mcparticle.fPx = part.Px();
      mcparticle.fPy = part.Py();
      mcparticle.fPz = part.Pz();
      mcparticle.fE  = part.GetEnergy();

      mcparticle.fVx = part.Vx();
      mcparticle.fVy = part.Vy();
      mcparticle.fVz = part.Vz();
      mcparticle.fVt = part.T();

      fTree[kMcParticle]->Fill();
    }
    //+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    // Go for conversion: save info
    fTree[kEvents]->Fill();
    fTree[kMcCollision]->Fill();
    fTree[kMcCollisionLabel]->Fill();
    fTree[kBC]->Fill();
    fTree[kFDD]->Fill();
    fTree[kFV0A]->Fill();
    fTree[kFV0C]->Fill();
    fTree[kFT0]->Fill();
    fTree[kZdc]->Fill();
    //+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    fOffsetLabel = fTree[kMcParticle] -> GetEntries(); //processed total
    lGoodEvents++;
  }
    
  TFile output("conversion-output.root", "recreate");
    
  //QA of conversion process: the basics
  hNVertices->Write();
  hNTracks->Write();
  hPtSpectra->Write();
  hPtSpectraFake->Write();
  
  fOutputDir->cd();
  fTree[kEvents] -> Write();
  fTree[kBC] -> Write();
  fTree[kFDD]->Write();
  fTree[kFV0A]->Write();
  fTree[kFV0C]->Write();
  fTree[kFT0]->Write();
  fTree[kZdc]->Write();
  fTree[kTracks] -> Write();
  fTree[kMcTrackLabel] -> Write();
  fTree[kMcParticle]->Write();
  fTree[kMcCollision]->Write();
  fTree[kMcCollisionLabel]->Write();
  
  std::cout<<"*- Saved "<<lGoodEvents<<" events with a PV (total processed: "<<itsHits.GetEntries()<<"). Enjoy! \U0001F596"<<std::endl;
  
}

