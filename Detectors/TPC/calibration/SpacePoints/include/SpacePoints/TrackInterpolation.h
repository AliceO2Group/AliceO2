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

/// \file TrackInterpolation.h
/// \brief Definition of the TrackInterpolation class
///
/// \author Ole Schmidt, ole.schmidt@cern.ch

#ifndef ALICEO2_TPC_TRACKINTERPOLATION_H_
#define ALICEO2_TPC_TRACKINTERPOLATION_H_

#include <gsl/span>
#include "CommonDataFormat/EvIndex.h"
#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/TrkClusRef.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTOF/Cluster.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/CalibratedTracklet.h"
#include "SpacePoints/SpacePointsCalibParam.h"
#include "SpacePoints/SpacePointsCalibConfParam.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "DetectorsBase/Propagator.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "TRDBase/RecoParam.h"
#include "TRDBase/Geometry.h"

class TTree;

namespace o2
{

namespace tpc
{
class VDriftCorrFact;

/// Internal struct used to store the unbinned TPC cluster residuals with float values
struct TPCClusterResiduals {
  float dy{};           ///< residual in Y
  float dz{};           ///< residual in Z
  float y{};            ///< Y position of track
  float z{};            ///< Z position of track
  float snp{};          ///< sin of the phi angle between padrow and track
  unsigned char sec{};  ///< sector number 0..35
  unsigned char dRow{}; ///< distance to previous row in units of pad rows
  void setDY(float val) { dy = fabs(val) < param::MaxResid ? val : std::copysign(param::MaxResid, val); }
  void setDZ(float val) { dz = fabs(val) < param::MaxResid ? val : std::copysign(param::MaxResid, val); }
  void setY(float val) { y = fabs(val) < param::MaxY ? val : std::copysign(param::MaxY, val); }
  void setZ(float val) { z = fabs(val) < param::MaxZ ? val : std::copysign(param::MaxZ, val); }
  void setSnp(float val) { snp = fabs(val) < param::MaxTgSlp ? val : std::copysign(param::MaxTgSlp, val); }
  ClassDefNV(TPCClusterResiduals, 4);
};

/// This struct is used to store the unbinned TPC cluster residuals in a compact way
/// (this is the data type which will be sent from the EPNs to the aggregator)
struct UnbinnedResid {
  UnbinnedResid() = default;
  UnbinnedResid(float dyIn, float dzIn, float tgSlpIn, float yIn, float zIn, unsigned char rowIn, unsigned char secIn) : dy(static_cast<short>(dyIn * 0x7fff / param::MaxResid)),
                                                                                                                         dz(static_cast<short>(dzIn * 0x7fff / param::MaxResid)),
                                                                                                                         tgSlp(static_cast<short>(tgSlpIn * 0x7fff / param::MaxTgSlp)),
                                                                                                                         y(static_cast<short>(yIn * 0x7fff / param::MaxY)),
                                                                                                                         z(static_cast<short>(zIn * 0x7fff / param::MaxZ)),
                                                                                                                         row(rowIn),
                                                                                                                         sec(secIn) {}
  short dy;          ///< residual in y
  short dz;          ///< residual in z
  short tgSlp;       ///< tan of the phi angle between padrow and track
  short y;           ///< y position of the track, needed for binning
  short z;           ///< z position of the track, needed for binning
  unsigned char row; ///< TPC pad row
  unsigned char sec; ///< TPC sector (0..35)
  ClassDefNV(UnbinnedResid, 1);
};

/// Structure for the information required to associate each residual with a given track type (ITS-TPC-TRD-TOF, etc)
struct TrackDataCompact {
  TrackDataCompact() = default;
  TrackDataCompact(uint32_t idx, uint8_t nRes, uint8_t source) : idxFirstResidual(idx), nResiduals(nRes), sourceId(source) {}
  uint32_t idxFirstResidual; ///< the index of the first residual from this track
  uint8_t nResiduals;        ///< total number of residuals associated to this track
  uint8_t sourceId;          ///< source ID obtained from the global track ID
  ClassDefNV(TrackDataCompact, 1);
};

// TODO add to UnbinnedResid::sec flag if cluster was used or not
// TODO add to TrackDataCompact::sourceId flag if track was used or not (if possible)

/// Heavy structure with track parameterizations + track points for debugging
struct TrackDataExtended {
  o2::dataformats::GlobalTrackID gid{};              ///< GID of the most global barrel track
  o2::its::TrackITS trkITS{};                        ///< ITS seeding track
  o2::tpc::TrackTPC trkTPC{};                        ///< TPC track (including dEdx information)
  o2::trd::TrackTRD trkTRD{};                        ///< TRD seeding track
  o2::track::TrackPar trkOuter{};                    ///< refit of TRD and/or TOF points
  o2::dataformats::MatchInfoTOF matchTOF{};          ///< TOF matching information
  std::vector<o2::BaseCluster<float>> clsITS{};      ///< attached ITS clusters
  std::vector<o2::trd::Tracklet64> trkltTRD{};       ///< attached TRD tracklets (if available)
  std::vector<o2::trd::CalibratedTracklet> clsTRD{}; ///< the TRD space points (if available)
  o2::tof::Cluster clsTOF{};                         ///< the TOF cluster (if available)
  o2::dataformats::RangeReference<> clIdx{};         ///< index of first cluster residual and total number of cluster residuals of this track
  ClassDefNV(TrackDataExtended, 2);
};

/// Structure filled for each track with track quality information and a vector with TPCClusterResiduals
struct TrackData {
  o2::dataformats::GlobalTrackID gid{}; ///< global track ID for seeding track
  o2::track::TrackPar par{};            ///< ITS track at inner TPC radius
  float dEdxTPC{};                      ///< TPC dEdx information
  float chi2TPC{};             ///< chi2 of TPC track
  float chi2ITS{};             ///< chi2 of ITS track
  float chi2TRD{};             ///< chi2 of TRD track
  unsigned short nClsTPC{};    ///< number of attached TPC clusters
  unsigned short nClsITS{};    ///< number of attached ITS clusters
  unsigned short nTrkltsTRD{}; ///< number of attached TRD tracklets
  unsigned short clAvailTOF{}; ///< whether or not track seed has a matched TOF cluster
  o2::dataformats::RangeReference<> clIdx{}; ///< index of first cluster residual and total number of cluster residuals of this track
  ClassDefNV(TrackData, 6);
};

/// \class TrackInterpolation
/// This class is retrieving the TPC space point residuals by interpolating ITS/TRD/TOF tracks.
/// The residuals are stored in the specified vectors of TPCClusterResiduals
/// It has been ported from the AliTPCcalibAlignInterpolation class from AliRoot.
class TrackInterpolation
{
 public:
  using MatCorrType = o2::base::Propagator::MatCorrType;

  /// Default constructor
  TrackInterpolation() = default;

  // since this class has pointer members, we should explicitly delete copy and assignment operators
  TrackInterpolation(const TrackInterpolation&) = delete;
  TrackInterpolation& operator=(const TrackInterpolation&) = delete;

  /// Enumeration for indexing the arrays of the CacheStruct
  enum {
    ExtOut = 0, ///< extrapolation outwards of ITS track
    ExtIn,      ///< extrapolation inwards of TRD/TOF track
    Int,        ///< interpolation (mean positions of both extrapolations)
    NIndices    ///< total number of indices (3)
  };

  /// Structure for caching positions, covariances and angles for extrapolations from ITS and TRD/TOF and for interpolation
  struct CacheStruct {
    std::array<float, NIndices> y{};
    std::array<float, NIndices> z{};
    std::array<float, NIndices> sy2{};
    std::array<float, NIndices> szy{};
    std::array<float, NIndices> sz2{};
    std::array<float, NIndices> snp{};
    float clY{0.f};
    float clZ{0.f};
    float clAngle{0.f};
    unsigned short clAvailable{0};
    unsigned char clSec{0};
  };

  /// Structure for on-the-fly re-calculated track parameters at the validation stage
  struct TrackParams {
    TrackParams() = default;
    float qpt{0.f};
    float tgl{0.f};
    std::array<float, param::NPadRows> zTrk{};
    std::array<float, param::NPadRows> xTrk{};
    std::array<float, param::NPadRows> dy{};
    std::array<float, param::NPadRows> dz{};
    std::array<float, param::NPadRows> tglArr{};
    std::bitset<param::NPadRows> flagRej{};
  };

  // -------------------------------------- processing functions --------------------------------------------------

  /// Initialize everything, set the requested track sources
  void init(o2::dataformats::GlobalTrackID::mask_t src);

  /// Check if input track passes configured cuts
  bool isInputTrackAccepted(const o2::dataformats::GlobalTrackID& gid, const o2::globaltracking::RecoContainer::GlobalIDSet& gidTable, const o2::dataformats::PrimaryVertex& pv) const;

  /// For given vertex track source which is not in mSourcesConfigured find the seeding source which is enabled
  o2::dataformats::GlobalTrackID::Source findValidSource(o2::dataformats::GlobalTrackID::Source src) const;

  /// Prepare input track sample (not relying on CreateTracksVariadic functionality)
  void prepareInputTrackSample(const o2::globaltracking::RecoContainer& inp);

  /// Given the defined downsampling factor tsalisThreshold check if track is selected
  bool isTrackSelected(const o2::track::TrackParCov& trk) const;

  /// Main processing function
  void process();

  /// Extrapolate ITS-only track through TPC and store residuals to TPC clusters along the way
  /// \param seed index
  void extrapolateTrack(int iSeed);

  /// Interpolate ITS-TRD-TOF track inside TPC and store residuals to TPC clusters along the way
  /// \param seed index
  void interpolateTrack(int iSeed);

  /// Reset cache and output vectors
  void reset();

  // -------------------------------------- outlier rejection --------------------------------------------------

  /// Validates the given input track and its residuals
  /// \param trk The track parameters, e.g. q/pT, eta, ...
  /// \param params Structure with per pad information recalculated on the fly
  /// \return true if the track could be validated, false otherwise
  bool validateTrack(const TrackData& trk, TrackParams& params, const std::vector<TPCClusterResiduals>& clsRes) const;

  /// Filter out individual outliers from all cluster residuals of given track
  /// \return true for tracks which pass the cuts on e.g. max. masked clusters and false for rejected tracks
  bool outlierFiltering(const TrackData& trk, TrackParams& params, const std::vector<TPCClusterResiduals>& clsRes) const;

  /// Is called from outlierFiltering() and does the actual calculations (moving average filter etc.)
  /// \return The RMS of the long range moving average
  float checkResiduals(const TrackData& trk, TrackParams& params, const std::vector<TPCClusterResiduals>& clsRes) const;

  /// Calculates the differences in Y and Z for a given set of clusters to a fitted helix.
  /// First a circular fit in the azimuthal plane is performed and subsequently a linear fit in the transversal plane
  bool compareToHelix(const TrackData& trk, TrackParams& params, const std::vector<TPCClusterResiduals>& clsRes) const;

  /// For a given set of points, calculate the differences from each point to the fitted lines from all other points in their neighbourhoods (+- nMAShort points)
  void diffToLocLine(const int np, int idxOffset, const std::array<float, param::NPadRows>& x, const std::array<float, param::NPadRows>& y, std::array<float, param::NPadRows>& diffY) const;

  /// For a given set of points, calculate their deviation from the moving average (build from the neighbourhood +- nMALong points)
  void diffToMA(const int np, const std::array<float, param::NPadRows>& y, std::array<float, param::NPadRows>& diffMA) const;

  // -------------------------------------- settings --------------------------------------------------
  void setTPCVDrift(const o2::tpc::VDriftCorrFact& v);

  /// Sets the flag if material correction should be applied when extrapolating the tracks
  void setMatCorr(MatCorrType matCorr) { mMatCorr = matCorr; }

  /// Sets the maximum number of tracks to be processed (successfully) per TF
  void setMaxTracksPerTF(int n) { mMaxTracksPerTF = n; }

  /// In addition to mMaxTracksPerTF up to the set number of ITS-TPC only tracks can be processed
  void setAddITSTPCTracksPerTF(int n) { mAddTracksITSTPC = n; }

  /// Enable full output
  void setDumpTrackPoints() { mDumpTrackPoints = true; }

  /// Allow setting the ITS cluster dictionary from outside
  void setITSClusterDictionary(const o2::itsmft::TopologyDictionary* dict) { mITSDict = dict; }

  /// Enable processing of seeds
  void setProcessSeeds() { mProcessSeeds = true; }

  /// Enable ITS-TPC only processing
  void setProcessITSTPConly() { mProcessITSTPConly = true; }

  /// Set the centre of mass energy required for pT downsampling Tsalis function
  void setSqrtS(float s) { mSqrtS = s; }

  // --------------------------------- output ---------------------------------------------
  std::vector<UnbinnedResid>& getClusterResiduals() { return mClRes; }
  std::vector<TrackDataCompact>& getTrackDataCompact() { return mTrackDataCompact; }
  std::vector<TrackDataExtended>& getTrackDataExtended() { return mTrackDataExtended; }
  std::vector<TrackData>& getReferenceTracks() { return mTrackData; }
  std::vector<TPCClusterResiduals>& getClusterResidualsUnfiltered() { return mClResUnfiltered; }
  std::vector<TrackData>& getReferenceTracksUnfiltered() { return mTrackDataUnfiltered; }

 private:
  static constexpr float sFloatEps{1.e-7f}; ///< float epsilon for robust linear fitting
  // parameters + settings
  const SpacePointsCalibConfParam* mParams = nullptr;
  float mTPCTimeBinMUS{.2f};    ///< TPC time bin duration in us
  float mTPCVDriftRef = -1.;    ///< TPC nominal drift speed in cm/microseconds
  float mTPCDriftTimeOffsetRef = 0.;                 ///< TPC nominal (e.g. at the start of run) drift time bias in cm/mus
  float mSqrtS{13600.f};                             ///< centre of mass energy set from LHC IF
  MatCorrType mMatCorr{MatCorrType::USEMatCorrNONE}; ///< if material correction should be done
  int mMaxTracksPerTF{-1};                           ///< max number of tracks to be processed per TF (-1 means there is no limit)
  int mAddTracksITSTPC{0};                           ///< number of ITS-TPC tracks which can be processed in addition to mMaxTracksPerTF
  bool mDumpTrackPoints{false};                      ///< dump also track points in ITS, TRD and TOF
  bool mProcessSeeds{false};                         ///< in case for global tracks also their shorter parts are processed separately
  bool mProcessITSTPConly{false};                    ///< flag, whether or not to extrapolate ITS-only through TPC
  o2::dataformats::GlobalTrackID::mask_t mSourcesConfigured; ///< keep only the matches here, not the individual detector contributors

  // input
  const o2::globaltracking::RecoContainer* mRecoCont = nullptr;                            ///< input reco container
  std::vector<o2::dataformats::GlobalTrackID> mGIDs{};                                     ///< GIDs of input tracks
  std::vector<o2::globaltracking::RecoContainer::GlobalIDSet> mGIDtables{};                ///< GIDs of contributors from single detectors for each seed
  std::vector<float> mTrackTimes{};                                                        ///< time estimates for all input tracks in micro seconds
  std::vector<o2::track::TrackParCov> mSeeds{};                                            ///< seeding track parameters (ITS tracks)
  std::map<int, int> mTrackTypes;                                                          ///< mapping of track source to array index in mTrackIndices
  std::array<std::vector<o2::dataformats::GlobalTrackID>, 4> mTrackIndices;                ///< keep GIDs of input tracks separately for each track type
  gsl::span<const TPCClRefElem> mTPCTracksClusIdx;                                         ///< input TPC cluster indices from span
  const ClusterNativeAccess* mTPCClusterIdxStruct = nullptr; ///< struct holding the TPC cluster indices
  // ITS specific input only needed for debugging
  gsl::span<const int> mITSTrackClusIdx;                    ///< input ITS track cluster indices span
  std::vector<o2::BaseCluster<float>> mITSClustersArray;    ///< ITS clusters created in run() method from compact clusters
  const o2::itsmft::TopologyDictionary* mITSDict = nullptr; ///< cluster patterns dictionary

  // output
  std::vector<TrackData> mTrackData{};                 ///< this vector is used to store the track quality information on a per track basis
  std::vector<TrackDataCompact> mTrackDataCompact{};   ///< required to connect each residual to a global track
  std::vector<TrackDataExtended> mTrackDataExtended{}; ///< full tracking information for debugging
  std::vector<UnbinnedResid> mClRes{};                 ///< residuals for each available TPC cluster of all tracks
  std::vector<TrackData> mTrackDataUnfiltered{};       ///< same as mTrackData, but for all tracks before outlier filtering
  std::vector<TPCClusterResiduals> mClResUnfiltered{}; ///< same as mClRes, but for all residuals before outlier filtering

  // cache
  std::array<CacheStruct, constants::MAXGLOBALPADROW> mCache{{}}; ///< caching positions, covariances and angles for track extrapolations and interpolation
  std::vector<o2::dataformats::GlobalTrackID> mGIDsSuccess;       ///< keep track of the GIDs which could be processed successfully

  // helpers
  o2::trd::RecoParam mRecoParam;                      ///< parameters required for TRD refit
  o2::trd::Geometry* mGeoTRD;                         ///< TRD geometry instance (needed for tilted pad correction)
  std::unique_ptr<TPCFastTransform> mFastTransform{}; ///< TPC cluster transformation
  float mBz;                                          ///< required for helix approximation
  bool mInitDone{false};                              ///< initialization done flag

  ClassDefNV(TrackInterpolation, 1);
};

} // namespace tpc

} // namespace o2

#endif
