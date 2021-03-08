// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <MathUtils/Primitive2D.h>
#include "ReconstructionDataFormats/TrackCosmics.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/GlobalTrackAccessor.h"
#include "ReconstructionDataFormats/BaseCluster.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "GlobalTracking/MatchCosmicsParams.h"
#include "TPCFastTransform.h"

namespace o2
{
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

  using InfoAccessor = o2d::AbstractRefAccessor<int, GTrackID::NSources>; // there is no unique <Info> structure, so the default return type is dummy (int)

  ///< record about matchig of 2 tracks
  struct MatchRecord {
    int id0 = MinusOne;  ///< id of 1st parnter
    int id1 = MinusOne;  ///< id of 2nd parnter
    float chi2 = -1.f;   ///< matching chi2
    int next = MinusOne; ///< index of eventual next record
  };

  struct TrackSeed : public o2::track::TrackParCov {
    TBracket tBracket;      ///< bracketing time-bins
    GTrackID origID;        ///< track origin id
    int matchID = MinusOne; ///< entry (none if MinusOne) of its match in the vector of matches
  };
  void setITSROFrameLengthMUS(float fums) { mITSROFrameLengthMUS = fums; }
  void setITSDict(std::unique_ptr<o2::itsmft::TopologyDictionary>& dict) { mITSDict = std::move(dict); }
  void process(const o2::globaltracking::RecoContainer& data);
  void setUseMC(bool mc) { mUseMC = mc; }
  void init();

  auto getCosmicTracks() const { return mCosmicTracks; }
  auto getCosmicTracksLbl() const { return mCosmicTracksLbl; }

 private:
  void updateTimeDependentParams();
  bool checkPair(int i, int j);
  void registerMatch(int i, int j, float chi2);
  void suppressMatch(int partner0, int partner1);
  void createSeeds(const o2::globaltracking::RecoContainer& data);
  bool validateMatch(int partner0);
  void selectWinners();
  void refitWinners(const o2::globaltracking::RecoContainer& data);
  std::vector<o2::BaseCluster<float>> prepareITSClusters(const o2::globaltracking::RecoContainer& data) const;

  std::vector<TrackSeed> mSeeds;
  std::vector<MatchRecord> mRecords;
  std::vector<int> mWinners;
  std::unique_ptr<o2::gpu::TPCFastTransform> mTPCTransform; ///< TPC cluster transformation
  std::unique_ptr<o2::itsmft::TopologyDictionary> mITSDict; // cluster patterns dictionary

  float mTPCTBinMUS = 0.; ///< TPC time bin duration in microseconds
  float mBz = 0;          ///< nominal Bz
  bool mFieldON = true;
  bool mUseMC = true;
  float mITSROFrameLengthMUS = 0.;
  float mQ2PtCutoff = 1e9;
  const MatchCosmicsParams* mMatchParams = nullptr;

  std::vector<o2d::TrackCosmics> mCosmicTracks;
  std::vector<o2::MCCompLabel> mCosmicTracksLbl;

  ClassDefNV(MatchCosmics, 1);
};

} // namespace globaltracking
} // namespace o2

#endif
