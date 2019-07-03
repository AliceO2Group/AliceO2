// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "TPCBase/ROC.h"
#include "TPCBase/PadPos.h"
#include "TPCBase/Mapper.h"
#include "TPCCalibration/FastHisto.h"

//root includes
#include "TFile.h"
#include "TTree.h"

#include <vector>
#include <tuple>

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
/// cGain.processTracks(false, 3, 8); // dont write the histograms to TTree, set minimum and maximum momentum range of the tracks 3<p<8
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
  CalibPadGainTracks(std::vector<o2::tpc::TrackTPC>* vTPCTracksArrayInp, std::vector<o2::tpc::TPCClRefElem>* tpcTrackClIdxVecInput, const o2::tpc::ClusterNativeAccess& clIndex)
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
  CalibPadGainTracks(std::vector<o2::tpc::TrackTPC>* vTPCTracksArrayInp, std::vector<o2::tpc::TPCClRefElem>* tpcTrackClIdxVecInput, const o2::tpc::ClusterNativeAccess& clIndex,
                     const unsigned int nBins, const float xmin, const float xmax, const bool useUnderflow, const bool useOverflow)
    : mTracks(vTPCTracksArrayInp), mTPCTrackClIdxVecInput(tpcTrackClIdxVecInput), mClusterIndex(&clIndex)
  {
    init(nBins, xmin, xmax, useUnderflow, useOverflow);
  };

  /// default destructor
  ~CalibPadGainTracks() = default;

  /// processes input tracks and filling the histograms with self calibrated probe qMax/dEdx
  /// \param writeTree write tree for debugging
  /// \param momMin minimum momentum which is required by tracks
  /// \param momMax maximum momentum which is required by tracks
  void processTracks(const bool writeTree = false, const float momMin = 0, const float momMax = 100);

  /// set the member variables
  /// \param vTPCTracksArrayInp vector of tpc tracks
  /// \param tpcTrackClIdxVecInput set the TPCClRefElem member variable
  /// \param clIndex set the ClusterNativeAccess member variable
  void setMembers(std::vector<o2::tpc::TrackTPC>* vTPCTracksArrayInp, std::vector<o2::tpc::TPCClRefElem>* tpcTrackClIdxVecInput, const o2::tpc::ClusterNativeAccess& clIndex)
  {
    mTracks = vTPCTracksArrayInp;
    mTPCTrackClIdxVecInput = tpcTrackClIdxVecInput;
    mClusterIndex = &clIndex;
  }

  /// this function sets the mode of the class.
  /// e.g. mode=0 -> use the truncated mean from the track for normalizing the dedx
  ///      mode=1 -> use the value from the BB-fit for normalizing the dedx. NOT implemented yet
  void setMode(dEdxType iMode)
  {
    mMode = iMode;
  }

  /// initialize the histograms with default parameters
  void initDefault()
  {
    mPadHistosDet = std::make_unique<o2::tpc::CalDet<o2::tpc::FastHisto<float>>>("Histo");
  }

  /// initialize the histograms with custom parameters
  /// \param nBins number of bins used in the histograms
  /// \param xmin minimum value in histogram
  /// \param xmax maximum value in histogram
  /// \param useUnderflow set usage of underflow bin
  /// \param useOverflow set usage of overflow bin
  void init(const unsigned int nBins, const float xmin, const float xmax, const bool useUnderflow, const bool useOverflow)
  {
    o2::tpc::FastHisto<float> hist(nBins, xmin, xmax, useUnderflow, useOverflow);
    initDefault();
    for (auto& calArray : mPadHistosDet->getData()) {
      for (auto& tHist : calArray.getData()) {
        tHist = hist;
      }
    }
  }

  /// dump the gain map to disk
  void dumpGainMap();

  /// get the truncated mean for each histogram and fill the extracted gainvalues in a CalPad object
  void fillgainMap();

  /// \return returns the gainmap object
  CalPad getPadGainMap() const
  {
    return mGainMap;
  }

 private:
  std::vector<o2::tpc::TrackTPC>* mTracks{nullptr};           ///< vector containing the tpc tracks which will be processed. Cant be const due to the propagate function
  std::vector<TPCClRefElem>* mTPCTrackClIdxVecInput{nullptr}; ///< input vector with TPC tracks cluster indicies
  const o2::tpc::ClusterNativeAccess* mClusterIndex{nullptr}; ///< needed to access clusternative with tpctracks
  dEdxType mMode = DedxTrack;                                 ///< normalization type: type=DedxTrack use truncated mean, type=DedxBB use value from BB fit

  inline static auto& mapper = Mapper::instance();     ///< initialize mapper object
  static constexpr unsigned int NROWS = 152;           ///< number of padrows used TODO change to mapper
  static constexpr unsigned int NROWSIROC = 63;        ///< number of padrows used TODO change to mapper
  static constexpr unsigned int NROWSOROC = 89;        ///< number of padrows used TODO change to mapper
  static constexpr unsigned int NSECTORS = 36;         ///< number of sectors TODO change to mapper
  static constexpr unsigned int NPADSINSECTOR = 14560; ///< number of total pads in sector TODO change to mapper

  std::unique_ptr<CalDet<o2::tpc::FastHisto<float>>> mPadHistosDet; ///< Calibration object containing for each pad a histogram with normalized charge
  CalPad mGainMap{"GainMap"};                                       ///< Gain map object

  /// calculate truncated mean for track
  /// \param track input track which will be processed
  /// \param momMin minimum momentum required by the track
  /// \param momMax maximum momentum required by the track
  void processTrack(o2::tpc::TrackTPC track, float momMin, float momMax);

  /// get the index for given pad which is needed for the filling of the CalDet object
  /// \param padSub pad subset type
  /// \param padSubsetNumber index of the pad subset
  /// \param row corresponding pad row
  /// \param pad pad in row
  static int getIndex(o2::tpc::PadSubset padSub, int padSubsetNumber, const int row, const int pad)
  {
    return mapper.getPadNumber(padSub, padSubsetNumber, row, pad);
  }

  float getTrackTopologyCorrection(o2::tpc::TrackTPC& track, int iCl);

  ///get the truncated mean for input vector and the truncation range low*nCl<nCl<high*nCl
  /// \param vCharge vector containing all qmax values of the track
  /// \param low lower cluster cut of  0.05*nCluster
  /// \param high higher cluster cut of  0.6*nCluster
  float getTruncMean(std::vector<float> vCharge, float low = 0.05f, float high = 0.6f) const;

  /// write the relevant variables used by this class to file
  void writeTree() const;
};

} // namespace tpc
} // namespace o2

#endif
