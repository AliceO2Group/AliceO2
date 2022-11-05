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
/// @file   CalibPadGainTracksBase.h
/// @author Matthias Kleiner, mkleiner@ikf.uni-frankfurt.de
///

#ifndef ALICEO2_TPC_CALIBPADGAINTRACKSBASE_H
#define ALICEO2_TPC_CALIBPADGAINTRACKSBASE_H

// o2 includes
#include "TPCBase/CalDet.h"
#include "TPCCalibration/FastHisto.h"

#include <gsl/span>

class TCanvas;

namespace o2
{
namespace tpc
{

/// \brief
/// base class for the pad-by-pad residual gain calibration
/// this class can be used to store for each pad in the TPC a histogram.
/// After all pad-by-pad histograms are filled one can calculate and store for each histogram the mean value in a CalPad object.
class CalibPadGainTracksBase
{
 public:
  /// normalization type of the extracted gain map
  enum NormType : unsigned char {
    none,  ///< no normalization
    stack, ///< normalization per GEM stack
    region ///< normalization per region
  };

  using DataTHisto = FastHisto<unsigned int>;
  using DataTHistos = CalDet<DataTHisto>;

  /// constructor
  /// \param initCalPad initialisation of the calpad for the gain map (if the gainmap is not extracted it can be false to save some memory)
  CalibPadGainTracksBase(const bool initCalPad = true);

  /// initializing CalPad object for gainmap
  void initCalPadMemory() { mGainMap = std::make_unique<CalPad>("GainMap"); }

  /// initializing CalPad object for std dev map
  void initCalPadStdDevMemory() { mSigmaMap = std::make_unique<CalPad>("SigmaMap"); }

  /// initializing CalPad object for std dev map
  void initCalPadStat() { mNClMap = std::make_unique<CalPad>("NClustersMap"); }

  /// copy constructor
  CalibPadGainTracksBase(const CalibPadGainTracksBase& other) : mPadHistosDet(std::make_unique<DataTHistos>(*other.mPadHistosDet)), mGainMap(std::make_unique<CalPad>(*other.mGainMap)) {}

  /// filling the pad-by-pad histograms
  /// \param caldets span of caldets containing pad-by-pad histograms
  void fill(const gsl::span<const DataTHistos>& caldets);

  /// filling the pad-by-pad histograms
  /// \param caldets span of caldets containing pad-by-pad histograms
  void fill(const DataTHistos& caldet) { *mPadHistosDet.get() += caldet; }

  /// Print the total number of entries and minimum number of entries (ToDo add some more informations which will be printed)
  void print() const;

  /// Add histograms from other container
  void merge(const CalibPadGainTracksBase* other) { *mPadHistosDet.get() += *(other->mPadHistosDet).get(); }

  /// check if the pad-by-pad histograms has enough data
  /// \param minEntries minimum number of entries in each histogram
  bool hasEnoughData(const int minEntries) const;

  /// get the truncated mean and the sigma for each histogram and fill the extracted values in a CalPad object
  /// \param minEntries minimum number of entries in pad-by-pad histogram to calculate the mean
  /// \param minRelgain minimum accpeted relative gain (if the gain is below this value it will be set to 1)
  /// \param maxRelgain maximum accpeted relative gain (if the gain is above this value it will be set to 1)
  /// \param low lower truncation range for calculating the rel gain
  /// \param high upper truncation range
  /// \param minStDev to exlude outliers (histograms with a very narrow distributions) a min std dev cut is used
  void finalize(const int minEntries = 10, const float minRelgain = 0.1f, const float maxRelgain = 2.f, const float low = 0.05f, const float high = 0.6f, const float minStDev = 0.01);

  /// returns calpad containing pad-by-pad histograms
  const auto& getHistos() const { return mPadHistosDet; }

  /// \return returns the gainmap object as const reference
  const CalPad& getPadGainMap() const { return *mGainMap; }

  /// \return returns the gainmap object as reference
  CalPad& getPadGainMap() { return *mGainMap; }

  /// \return returns the gainmap object
  const CalPad& getSigmaMap() const { return *mSigmaMap; }

  /// \return returns the number of tracks per pad map
  const CalPad& getNTracksMap() const { return *mNClMap; }

  /// \return return histogram which is used to extract the gain
  /// \param sector sector of the TPC
  /// \param region region of the TPC
  /// \param lrow local row in region
  /// \param pad pad in row
  DataTHisto getHistogram(const int sector, const int region, const int lrow, const int pad) const;

  /// \return return histogram which is used to extract the gain
  /// \param sector sector of the TPC
  /// \param grow global row in sector
  /// \param pad pad in row
  DataTHisto getHistogram(const int sector, const int grow, const int pad) const;

  /// draw gain map sector
  /// \param sector sector which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn
  /// \param minZ min z value for drawing (if minZ > maxZ automatic z axis)
  /// \param maxZ max z value for drawing (if minZ > maxZ automatic z axis)
  void drawExtractedGainMapSector(const int sector, const std::string filename = "GainMapSector.pdf", const float minZ = 0, const float maxZ = -1) const { drawExtractedGainMapHelper(false, 0, sector, filename, minZ, maxZ); }

  /// draw gain map side
  /// \param side side of the TPC which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn
  /// \param minZ min z value for drawing (if minZ > maxZ automatic z axis)
  /// \param maxZ max z value for drawing (if minZ > maxZ automatic z axis)
  void drawExtractedGainMapSide(const o2::tpc::Side side, const std::string filename = "GainMapSide.pdf", const float minZ = 0, const float maxZ = -1) const { drawExtractedGainMapHelper(true, 0, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), filename, minZ, maxZ); }

  /// draw number of clusters side
  /// \param side side of the TPC which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn
  /// \param minZ min z value for drawing (if minZ > maxZ automatic z axis)
  /// \param maxZ max z value for drawing (if minZ > maxZ automatic z axis)
  void drawNClustersMapSide(const o2::tpc::Side side, const std::string filename = "NClustersMapSide.pdf", const float minZ = 0, const float maxZ = -1) const { drawExtractedGainMapHelper(true, 2, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), filename, minZ, maxZ); }

  /// draw sigma map sector
  /// \param sector sector which will be drawn
  /// \param norm normalizing the sigma to the extracted gain map
  /// \param filename name of the output file. If empty the canvas is drawn
  /// \param minZ min z value for drawing (if minZ > maxZ automatic z axis)
  /// \param maxZ max z value for drawing (if minZ > maxZ automatic z axis)
  void drawExtractedSigmaMapSector(const int sector, const bool norm = false, const std::string filename = "StdDevMapSector.pdf", const float minZ = 0, const float maxZ = -1) const { drawExtractedGainMapHelper(false, 1, sector, filename, minZ, maxZ, norm); }

  /// draw sigma map side
  /// \param side side of the TPC which will be drawn
  /// \param norm normalizing the sigma to the extracted gain map
  /// \param filename name of the output file. If empty the canvas is drawn
  /// \param minZ min z value for drawing (if minZ > maxZ automatic z axis)
  /// \param maxZ max z value for drawing (if minZ > maxZ automatic z axis)
  void drawExtractedSigmaMapSide(const o2::tpc::Side side, const bool norm = false, const std::string filename = "StdDevMapSide.pdf", const float minZ = 0, const float maxZ = -1) const { drawExtractedGainMapHelper(true, 1, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), filename, minZ, maxZ, norm); }

  /// draw number of clusters sector
  /// \param sector sector which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn
  /// \param minZ min z value for drawing (if minZ > maxZ automatic z axis)
  /// \param maxZ max z value for drawing (if minZ > maxZ automatic z axis)
  void drawNClustersMapSector(const int sector, const std::string filename = "NClustersMapSector.pdf", const float minZ = 0, const float maxZ = -1) const { drawExtractedGainMapHelper(false, 2, sector, filename, minZ, maxZ, false); }

  /// draw gain map using painter functionality
  TCanvas* drawExtractedGainMapPainter() const;

  /// divide the extracted gain map with a CalDet (can be usefull for comparing two gainmaps)
  /// \param inpFile input file containing some caldet
  /// \param mapName name of the caldet
  void divideGainMap(const char* inpFile, const char* mapName);

  /// dump the gain map to disk
  /// \param fileName name of the output file
  void dumpGainMap(const char* fileName = "GainMap.root") const;

  /// dump object to disc
  /// \param outFileName name of the output file
  /// \param outName name of the object in the output file
  void dumpToFile(const char* outFileName = "calPadGainTracksBase.root", const char* outName = "calPadGain") const;

  /// dump extracted gain and sigma map to tree
  /// \filename name of the output file
  void dumpToTree(const std::string filename = "map_debug.root") const;

  /// setting a gain map from a file
  /// \param inpFile input file containing some caldet
  /// \param mapName name of the caldet
  void setGainMap(const char* inpFile, const char* mapName);

  /// setting the gain map
  void setGainMap(const CalPad& gainmap) { mGainMap = std::make_unique<CalPad>(gainmap); }

  /// setting the RMS map
  void setRMSMap(const CalPad& rmsMap) { mSigmaMap = std::make_unique<CalPad>(rmsMap); }

  /// setting number of clusters map
  void setNClMap(const CalPad& nclMap) { mNClMap = std::make_unique<CalPad>(nclMap); }

  /// set how the extracted gain map is normalized
  void setNormalizationType(const NormType type) { mNormType = type; }

  /// \return return how the extracted gain map is normalized
  auto getNormalizationType() const { return mNormType; }

  /// resetting the histograms which are used for extraction of the gain map
  void resetHistos();

  /// initialize the histograms with custom parameters
  /// \param nBins number of bins used in the histograms
  /// \param xmin minimum value in histogram
  /// \param xmax maximum value in histogram
  /// \param useUnderflow set usage of underflow bin
  /// \param useOverflow set usage of overflow bin
  void init(const unsigned int nBins, const float xmin, const float xmax, const bool useUnderflow, const bool useOverflow);

  /// \param roc numerical ROC value
  /// \param padInROC pad number in ROC
  /// \param val value which is filled in the pad-by-pad histogram
  void fillPadByPadHistogram(const size_t roc, const size_t padInROC, const float val) { mPadHistosDet->getCalArray(roc).getData()[padInROC].fill(val); }

 private:
  std::unique_ptr<DataTHistos> mPadHistosDet; ///< Calibration object containing for each pad a histogram with normalized charge
  std::unique_ptr<CalPad> mGainMap;           ///< Extracted gain map from tracks
  std::unique_ptr<CalPad> mSigmaMap;          ///< standard deviation map
  std::unique_ptr<CalPad> mNClMap;            ///< statistics (number of entries per pad)
  NormType mNormType = region;                ///< Normalization type for the extracted gain map

  /// Helper function for drawing the extracted gain map
  void drawExtractedGainMapHelper(const bool type, const int typeMap, const Sector sector, const std::string filename, const float minZ, const float maxZ, const bool norm = false) const;

  /// normalizing an input vector
  /// \param data vector which will be normalized
  /// \param nPads intervals which will be used to calculate the median of the data vector for normalization
  void normalize(std::vector<float>& data, const std::vector<int>& nPads);

  /// normalize calpad per stack or per region
  /// \calPad Calpad which will be normalized
  void normalizeGain(CalPad& calPad);

  /// helper function for normalizing the gain
  std::vector<int> getNPadsForNormalization(const bool iroc) const;

  /// normalize calpad per stack or per region
  void normalizeGainMap() { normalizeGain(*mGainMap.get()); }
};

} // namespace tpc
} // namespace o2

#endif
