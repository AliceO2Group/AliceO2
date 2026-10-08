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

/// \file MatchCosmics.h
/// \brief Class to perform matching/refit of cosmic tracks legs
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_MATCH_COSMICS
#define ALICEO2_MATCH_COSMICS

#include <Rtypes.h>
#include <array>
#include <cstdint>
#include <MathUtils/Primitive2D.h>
#include "ReconstructionDataFormats/TrackCosmics.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/GlobalTrackAccessor.h"
#include "ReconstructionDataFormats/BaseCluster.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsITSMFT/ClustersPerLayer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "GlobalTracking/MatchCosmicsParams.h"
#include "CommonUtils/TreeStreamRedirector.h"

#define _ALLOW_DEBUG_TREES_COSM // to allow debug and control tree output

namespace o2
{
namespace tpc
{
class VDriftCorrFact;
}
namespace gpu
{
class TPCFastTransformPOD;
class GPUO2InterfaceRefit;
} // namespace gpu
namespace globaltracking
{

namespace o2d = o2::dataformats;

using GTrackID = o2d::GlobalTrackID;
using TBracket = o2::math_utils::Bracketf_t;
class RecoContainer;

class MatchCosmics
{
 public:
  static constexpr int Zero = 0;
  static constexpr int MinusOne = -1;
  static constexpr int MinusTen = -10;
  static constexpr int Validated = -2;
  static constexpr int Reject = MinusTen;
  enum RejFlag {
    Accept = 0,
    RejY,
    RejZ,
    RejSnp,
    RejTgl,
    RejQ2Pt,
    RejTime,
    RejProp,
    RejChi2,
    RejOther,
    RejSameHalf
  };

  using InfoAccessor = o2d::AbstractRefAccessor<int, GTrackID::NSources>; // there is no unique <Info> structure, so the default return type is dummy (int)

  ///< record about matchig of 2 tracks
  struct MatchRecord {
    int id0 = MinusOne;  ///< id of 1st parnter
    int id1 = MinusOne;  ///< id of 2nd parnter
    float chi2 = -1.f;   ///< matching chi2
    int next = MinusOne; ///< index of eventual next record
    float tCommon = 0.f; ///< common time [mus] fixed by z continuity of TPC-only legs on opposite TPC sides
    float tCommonErr = -1.f;      ///< its 1 sigma error [mus]; < 0: not fixed, the refit uses the centre of the time-bracket overlap and the cosmic's time error is the overlap's half-width
    float tofScore = -1.f;        ///< score of the top / bottom TOF hit pair matching the muon's flight (tofFlightSelection; < 0: none); a pair with one wins against pairs without
    float tCommonNoTOF = 0.f;     ///< tCommon before a TOF flight pair replaced it: the refit falls back to it if the refit at the TOF time fails
    float tCommonErrNoTOF = -1.f; ///< tCommonErr before a TOF flight pair replaced it
  };

  struct TOFCandidate { ///< TOF cluster along the outward continuation of a TPC-only seed
    int index = -1;     ///< index of the TOF cluster
    double timeNS = 0.; ///< its time since the start of the TF [ns]
    float dy = 0.f;     ///< cluster - predicted y in the frame of the cluster's sector [cm]
    float dz = 0.f;     ///< cluster - predicted z, the leg's z taken at its own reference time tRef [cm]
    float gx = 0.f;     ///< global position of the cluster [cm]
    float gy = 0.f;
    float gz = 0.f;
  };

  struct TrackSeed : public o2::track::TrackParCov {
    TBracket tBracket;      ///< bracketing time-bins
    GTrackID origID;        ///< track origin id
    int matchID = MinusOne; ///< entry (none if MinusOne) of its match in the vector of matches
    short vtIDMin = -1;     ///< id of the 1st compatible vertex
    short vtIDMax = -1;     ///< id of the last compatible vertex
    float tRef = 0.f;       ///< time [mus] the z of the parameters refers to (TPC-only: the TrackTPC time0; others: bracket centre)
    int8_t tpcSide = 0;     ///< TPC-only seed with clusters on one side: +1 A, -1 C (z = z(t) - side*vD*(t-tRef)); 0: z absolute
    std::array<float, 3> xyzRef{}; ///< global position of the reference point before the propagation to the DCA (same-half veto)
  };
  void setTPCCorrMaps(const o2::gpu::TPCFastTransformPOD* maph);
  void setTPCVDrift(const o2::tpc::VDriftCorrFact& v);
  void setITSROFrameLengthMUS(float fums) { mITSROFrameLengthMUS = fums; }
  void setITSDict(const o2::itsmft::TopologyDictionary* dict) { mITSDict = dict; }
  void process(const o2::globaltracking::RecoContainer& data);
  void setUseMC(bool mc) { mUseMC = mc; }
  void setUsePVInfo(bool v) { mUsePVInfo = v; }
  void init();
  void end();

  auto getCosmicTracks() const { return mCosmicTracks; }
  auto getCosmicTracksLbl() const { return mCosmicTracksLbl; }

#ifdef _ALLOW_DEBUG_TREES_COSM
  enum DebugFlagTypes : UInt_t {
    MatchTreeAll = 0x1,         ///< produce matching candidates tree for all candidates
    MatchTreeAccOnly = 0x1 << 1 ///< fill the matching candidates tree only once the cut is passed
  };
  ///< check if partucular flags are set
  bool isDebugFlag(UInt_t flags) const { return mDBGFlags & flags; }

  ///< get debug trees flags
  UInt_t getDebugFlags() const { return mDBGFlags; }

  ///< set or unset debug stream flag
  void setDebugFlag(UInt_t flag, bool on = true);

  ///< set the name of output debug file
  void setDebugTreeFileName(std::string name)
  {
    if (!name.empty()) {
      mDebugTreeFileName = name;
    }
  }

  ///< get the name of output debug file
  const std::string& getDebugTreeFileName() const { return mDebugTreeFileName; }
#endif

 private:
  void updateTimeDependentParams();
  RejFlag checkPair(int i, int j);
  bool refitSeedAtTime(const TrackSeed& seed, float timeMUS, TrackSeed& out);
  void registerMatch(int i, int j, float chi2, float tCommon = 0.f, float tCommonErr = -1.f, float tofScore = -1.f, float tCommonNoTOF = 0.f, float tCommonErrNoTOF = -1.f);
  void prepareTOFClusters(const o2::globaltracking::RecoContainer& data);
  const std::vector<TOFCandidate>& getTOFCandidates(int iseed);
  float findTOFFlightPair(int i, int j, float tMinMUS, float tMaxMUS, float& tofTimeMUS);
  void suppressMatch(int partner0, int partner1);
  void createSeeds(const o2::globaltracking::RecoContainer& data);
  bool validateMatch(int partner0);
  void selectWinners();
  void refitWinners(const o2::globaltracking::RecoContainer& data);
  o2::itsmft::ClustersPerLayer<o2::BaseCluster<float>> prepareITSClusters(const o2::globaltracking::RecoContainer& data) const;

  std::vector<TrackSeed> mSeeds;
  std::vector<MatchRecord> mRecords;
  std::vector<int> mWinners;
  const o2::itsmft::TopologyDictionary* mITSDict = nullptr; // cluster patterns dictionary
  const o2::gpu::TPCFastTransformPOD* mTPCCorrMaps = nullptr;
  int mTFCount = 0;
  float mTPCVDriftRef = -1.;      ///< TPC nominal drift speed in cm/microseconds
  float mTPCVDriftCorrFact = 1.;  ///< TPC nominal correction factort (wrt ref)
  float mTPCVDrift = -1.;         ///< TPC drift speed in cm/microseconds
  float mTPCDriftTimeOffset = 0.; ///< drift time offset in mus
  float mTPCTBinMUS = 0.;         ///< TPC time bin duration in microseconds
  float mBz = 0;                  ///< nominal Bz
  bool mFieldON = true;
  bool mUsePVInfo = false;
  bool mUseMC = true;
  float mITSROFrameLengthMUS = 0.;
  float mQ2PtCutoff = 1e9;
  float mQ2PtCutoffOppositeSides = 1e9;
  const MatchCosmicsParams* mMatchParams = nullptr;
  const o2::globaltracking::RecoContainer* mRecoData = nullptr; ///< inputs of the TF being processed
  o2::gpu::GPUO2InterfaceRefit* mTPCRefitter = nullptr;         ///< TPC refitter of the TF being processed (owned by process())
  size_t mNRefitsCommonTime = 0;                                ///< seeds refitted at the common time of a same-side pair in this TF
  std::vector<int> mTOFClusterOrder;                            ///< TOF clusters of the TF sorted in time (tofFlightSelection)
  std::vector<float> mTOFClusterTimeMUS;                        ///< their times since the start of the TF [mus], same order
  std::vector<std::vector<TOFCandidate>> mSeedTOFCandidates;    ///< TOF candidates per seed, filled on first use
  std::vector<bool> mSeedTOFDone;                               ///< the TOF candidates of the seed are filled
  size_t mNTOFConfirmed = 0;                                    ///< accepted pairs with a TOF flight pair in this TF
  size_t mNTOFFallbacks = 0;                                    ///< TOF-confirmed winners refitted at their time without TOF in this TF

  std::vector<o2d::TrackCosmics> mCosmicTracks;
  std::vector<o2::MCCompLabel> mCosmicTracksLbl;

#ifdef _ALLOW_DEBUG_TREES_COSM
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  UInt_t mDBGFlags = 0;
  std::string mDebugTreeFileName = "dbg_cosmics_match.root"; ///< name for the debug tree file
#endif

  ClassDefNV(MatchCosmics, 1);
};

} // namespace globaltracking
} // namespace o2

#endif
