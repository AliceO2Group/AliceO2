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

///
/// @file   CalibPadGainTracks.h
/// @author Matthias Kleiner, matthias.kleiner@cern.ch
///

#ifndef AliceO2_TPC_CalibPadGainTracks_H
#define AliceO2_TPC_CalibPadGainTracks_H

//o2 includes
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/Mapper.h"
#include "TPCCalibration/FastHisto.h"

#include <vector>
#include <gsl/span>
#include <tuple>

class TCanvas;

namespace o2
{
namespace tpc
{

/// \brief Gain calibration class
///
/// This class is used to produce pad wise gain calibration information with reconstructed tracks.
/// The idea is to use the self calibrated probe qMax/dEdx and store the information for each pad in an histogram.
/// The dEdx can be used from the track itself or from some bethe-bloch parametrization.
/// Using the dEdx information from the bethe bloch avoids biases in the dEdx of the track itself.
/// However the use of a bethe bloch parametrization is not yet implemented and shouldnt be used yet.
/// When enough thata is collected, the truncated mean of each histogram delivers the relative gain of each pad.
/// This method can be used to study the pad-by-pad gain as a function of time (i.e. performing this method n times with n different consecutive data samples)
///
/// origin: TPC
/// \author Matthias Kleiner, matthias.kleiner@cern.ch
///
///
/// how to use:
/// example:
/// CalibPadGainTracks cGain{};
/// cGain.init(20, 0, 3, 1, 1); // set the binning which will be used: 20 bins, minimum x=0, maximum x=10, use underflow and overflow bin
/// start loop over the data
/// cGain.setMembers(tpcTracks, tpcTrackClIdxVecInput, clusterIndex); // set the member variables: TrackTPC, TPCClRefElem, o2::tpc::ClusterNativeAccess
/// cGain.setMomentumRange(.1, 3);
/// cGain.processTracks();
/// after looping of the data (filling the histograms) is done
/// cGain.fillgainMap(); // fill the gainmap with the truncated mean from each histogram
/// cGain.dumpGainMap(); // write the gainmap to file
///
/// see also: extractGainMap.C macro

class CalibPadGainTracks
{

 public:
  /// mode of normalizing qmax
  enum dEdxType : unsigned char {
    DedxTrack, ///< normalize qMax using the truncated mean from the track
    DedxBB     ///< normalize qMax by evaluating a Bethe Bloch fit. THIS is yet not implemented and shouldnt be used.
  };

  /// default constructor
  /// the member variables have to be set manually with setMembers()
  CalibPadGainTracks() = default;

  /// constructor
  /// \param vTPCTracksArrayInp vector of tpc tracks
  /// \param tpcTrackClIdxVecInput the TPCClRefElem of the track
  /// \param clIndex clusternative access object
  CalibPadGainTracks(gsl::span<const o2::tpc::TrackTPC>* vTPCTracksArrayInp, gsl::span<const o2::tpc::TPCClRefElem>* tpcTrackClIdxVecInput, const o2::tpc::ClusterNativeAccess& clIndex)
    : mTracks(vTPCTracksArrayInp), mTPCTrackClIdxVecInput(tpcTrackClIdxVecInput), mClusterIndex(&clIndex)
  {
    initDefault();
  };

  /// constructor
  /// \param vTPCTracksArrayInp vector of tpc tracks
  /// \param tpcTrackClIdxVecInput the TPCClRefElem of the track
  /// \param clIndex clusternative access object
  /// \param nBins number of bins used in the histograms
  /// \param xmin minimum value in histogram
  /// \param xmax maximum value in histogram
  /// \param useUnderflow set usage of underflow bin
  /// \param useOverflow set usage of overflow bin
  CalibPadGainTracks(gsl::span<const o2::tpc::TrackTPC>* vTPCTracksArrayInp, gsl::span<const o2::tpc::TPCClRefElem>* tpcTrackClIdxVecInput, const o2::tpc::ClusterNativeAccess& clIndex,
                     const unsigned int nBins, const float xmin, const float xmax, const bool useUnderflow, const bool useOverflow)
    : mTracks(vTPCTracksArrayInp), mTPCTrackClIdxVecInput(tpcTrackClIdxVecInput), mClusterIndex(&clIndex)
  {
    init(nBins, xmin, xmax, useUnderflow, useOverflow);
  };

  /// default destructor
  ~CalibPadGainTracks() = default;

  /// processes input tracks and filling the histograms with self calibrated probe qMax/dEdx
  void processTracks();

  /// set the member variables
  /// \param vTPCTracksArrayInp vector of tpc tracks
  /// \param tpcTrackClIdxVecInput set the TPCClRefElem member variable
  /// \param clIndex set the ClusterNativeAccess member variable
  void setMembers(gsl::span<const o2::tpc::TrackTPC>* vTPCTracksArrayInp, gsl::span<const o2::tpc::TPCClRefElem>* tpcTrackClIdxVecInput, const o2::tpc::ClusterNativeAccess& clIndex);

  /// this function sets the mode of the class.
  /// e.g. mode=0 -> use the truncated mean from the track for normalizing the dedx
  ///      mode=1 -> use the value from the BB-fit for normalizing the dedx. NOT implemented yet
  void setMode(dEdxType iMode) { mMode = iMode; }

  /// \param momMin minimum accpeted momentum of the tracks
  /// \param momMax maximum accpeted momentum of the tracks
  void setMomentumRange(const float momMin, const float momMax);

  /// \param eta maximum accpeted eta of the tracks
  void setMaxEta(const float eta) { mEtaMax = eta; }

  /// \param eta maximum accpeted eta of the tracks
  void setMinNClusters(const float nCl) { mMinClusters = nCl; }

  /// \param field magnetic field in kG, used for track propagation
  void setField(const float field) { mField = field; }

  /// \return returns minimum momentum of accepted tracks
  float getMomMin() const { return mMomMin; }

  /// \return returns maximum momentum of accepted tracks
  float getMomMax() const { return mMomMax; }

  /// \return returns maximum eta of accepted tracks
  float getEtaMax() const { return mEtaMax; }

  /// \return returns maximum eta of accepted tracks
  float getMinNClusters() const { return mMinClusters; }

  /// \return returns magnetic field which is used for propagation of track parameters
  float getField() const { return mField; };

  /// initialize the histograms with default parameters
  void initDefault();

  /// \param low lower truncation range for calculating the rel gain
  /// \param high upper truncation range
  void setTruncationRange(const float low = 0.05f, const float high = 0.6f);

  /// \param minEntries minimum number of entries which are needed to extract the mean value
  void setMinEntriesHis(const int minEntries) { mMinEntries = minEntries; }

  /// initialize the histograms with custom parameters
  /// \param nBins number of bins used in the histograms
  /// \param xmin minimum value in histogram
  /// \param xmax maximum value in histogram
  /// \param useUnderflow set usage of underflow bin
  /// \param useOverflow set usage of overflow bin
  void init(const unsigned int nBins, const float xmin, const float xmax, const bool useUnderflow, const bool useOverflow);

  /// dump the gain map to disk
  /// \param fileName name of the output file
  void dumpGainMap(const char* fileName = "GainMap.root") const;

  /// dump object to disc
  /// \param outFileName name of the output file
  /// \param outName name of the object in the output file
  void dumpToFile(const char* outFileName = "calPadGainTracks.root", const char* outName = "calPadGain") const;

  /// dump histograms to tree for debugging
  /// \param outFileName name of the output file
  void dumpToTree(const char* outFileName = "GainMapTree.root") const;

  /// get the truncated mean for each histogram and fill the extracted gainvalues in a CalPad object
  void fillgainMap();

  /// \return returns the gainmap object
  const CalPad& getPadGainMap() const { return mGainMap; }

  /// draw gain map sector
  /// \param sector sector which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn
  /// \param minZ min z value for drawing (if minZ > maxZ automatic z axis)
  /// \param maxZ max z value for drawing (if minZ > maxZ automatic z axis)
  void drawExtractedGainMapSector(const int sector, const std::string filename = "GainMapSector.pdf", const float minZ = 0, const float maxZ = -1) const { drawExtractedGainMapHelper(false, sector, filename, minZ, maxZ); }

  /// draw gain map side
  /// \param side side of the TPC which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn
  /// \param minZ min z value for drawing (if minZ > maxZ automatic z axis)
  /// \param maxZ max z value for drawing (if minZ > maxZ automatic z axis)
  void drawExtractedGainMapSide(const o2::tpc::Side side, const std::string filename = "GainMapSide.pdf", const float minZ = 0, const float maxZ = -1) const { drawExtractedGainMapHelper(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), filename, minZ, maxZ); }

  /// draw gain map using painter functionality
  TCanvas* drawExtractedGainMapPainter() const;

  /// \return return histogram which is used to extract the gain
  /// \param sector sector of the TPC
  /// \param region region of the TPC
  /// \param lrow local row in region
  /// \param pad pad in row
  auto getHistogram(const int sector, const int region, const int lrow, const int pad) const { return mPadHistosDet->getValue(sector, Mapper::getGlobalPadNumber(lrow, pad, region)); }

  /// \return return histogram which is used to extract the gain
  /// \param sector sector of the TPC
  /// \param grow global row in sector
  /// \param pad pad in row
  auto getHistogram(const int sector, const int grow, const int pad) const { return mPadHistosDet->getValue(sector, Mapper::GLOBALPADOFFSET[Mapper::REGION[grow]] + Mapper::OFFSETCRUGLOBAL[grow] + pad); }

  /// divide the extracted gain map with a CalDet (can be usefull for comparing two gainmaps)
  /// \param inpFile input file containing some caldet
  /// \param mapName name of the caldet
  void divideGainMap(const char* inpFile, const char* mapName);

  /// setting a gain map from a file
  /// \param inpFile input file containing some caldet
  /// \param mapName name of the caldet
  void setGainMap(const char* inpFile, const char* mapName);

  /// resetting the histograms which are used for extraction of the gain map
  void resetHistos();

 private:
  gsl::span<const TrackTPC>* mTracks{nullptr};                                        ///<! vector containing the tpc tracks which will be processed. Cant be const due to the propagate function
  gsl::span<const TPCClRefElem>* mTPCTrackClIdxVecInput{nullptr};                     ///<! input vector with TPC tracks cluster indicies
  const o2::tpc::ClusterNativeAccess* mClusterIndex{nullptr};                         ///<! needed to access clusternative with tpctracks
  dEdxType mMode = DedxTrack;                                                         ///< normalization type: type=DedxTrack use truncated mean, type=DedxBB use value from BB fit
  inline static auto& mapper = Mapper::instance();                                    ///< initialize mapper object
  float mField{-5};                                                                   ///< Magnetic field in kG, used for track propagation
  float mMomMin{0.1f};                                                                ///< minimum momentum which is required by tracks
  float mMomMax{5.f};                                                                 ///< maximum momentum which is required by tracks
  float mEtaMax{1.f};                                                                 ///< maximum accpeted eta of tracks
  int mMinClusters{50};                                                               ///< minimum number of clusters the tracks require
  std::vector<float> mDEdxIROC{};                                                     ///<! memory for dE/dx in IROC
  std::vector<float> mDEdxOROC{};                                                     ///<! memory for dE/dx in OROC
  std::vector<o2::tpc::ClusterNative> mCLNat;                                         ///<! memory for clusters
  std::vector<std::tuple<unsigned char, unsigned char, unsigned char, float>> mClTrk; ///<! memory for cluster informations
  std::unique_ptr<CalDet<FastHisto<float>>> mPadHistosDet;                            ///< Calibration object containing for each pad a histogram with normalized charge
  CalPad mGainMap{"GainMap"};                                                         ///< Gain map object
  float mLowTruncation{0.05f};                                                        ///< lower truncation range for calculating mean of the histograms
  float mUpTruncation{0.6f};                                                          ///< upper truncation range for calculating mean of the histograms
  int mMinEntries{20};                                                                ///< minimum number of entries which are needed to extract the mean value

  /// calculate truncated mean for track
  /// \param track input track which will be processed
  void processTrack(TrackTPC track);

  /// get the index for given pad which is needed for the filling of the CalDet object
  /// \param padSub pad subset type
  /// \param padSubsetNumber index of the pad subset
  /// \param row corresponding pad row
  /// \param pad pad in row
  static int getIndex(o2::tpc::PadSubset padSub, int padSubsetNumber, const int row, const int pad) { return mapper.getPadNumber(padSub, padSubsetNumber, row, pad); }

  float getTrackTopologyCorrection(o2::tpc::TrackTPC& track, int iCl);

  ///get the truncated mean for input vector and the truncation range low*nCl<nCl<high*nCl
  /// \param vCharge vector containing all qmax values of the track
  /// \param low lower cluster cut of  0.05*nCluster
  /// \param high higher cluster cut of  0.6*nCluster
  float getTruncMean(std::vector<float>& vCharge, float low = 0.05f, float high = 0.6f) const;

  /// Helper function for drawing the extracted gain map
  void drawExtractedGainMapHelper(const bool type, const Sector sector, const std::string filename, const float minZ, const float maxZ) const;
};

} // namespace tpc
} // namespace o2

#endif
