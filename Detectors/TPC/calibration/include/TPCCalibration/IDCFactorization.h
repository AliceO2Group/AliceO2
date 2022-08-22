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

/// \file IDCFactorization.h
/// \brief class for aggregating IDCs for the full TPC (all sectors) and factorization of aggregated IDCs
///
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Apr 30, 2021

#ifndef ALICEO2_IDCFACTORIZATION_H_
#define ALICEO2_IDCFACTORIZATION_H_

#include <vector>
#include "Rtypes.h"
#include "TPCBase/Mapper.h"
#include "TPCCalibration/IDCContainer.h"
#include "TPCCalibration/IDCGroupHelperSector.h"
#include "DataFormatsTPC/Defs.h"

namespace o2::tpc
{

template <class T>
class CalDet;

class IDCFactorization : public IDCGroupHelperSector
{
 public:
  /// constructor
  /// \param groupPads number of pads in pad direction which will be grouped for all regions
  /// \param groupRows number of pads in row direction which will be grouped for all regions
  /// \param groupLastRowsThreshold minimum number of pads in row direction for the last group in row direction for all regions
  /// \param groupLastPadsThreshold minimum number of pads in pad direction for the last group in pad direction for all regions
  /// \param timeFrames number of time frames which will be stored
  /// \param timeframesDeltaIDC number of time frames stored for each DeltaIDC object
  IDCFactorization(const std::array<unsigned char, Mapper::NREGIONS>& groupPads, const std::array<unsigned char, Mapper::NREGIONS>& groupRows, const std::array<unsigned char, Mapper::NREGIONS>& groupLastRowsThreshold, const std::array<unsigned char, Mapper::NREGIONS>& groupLastPadsThreshold, const unsigned int groupNotnPadsSectorEdges, const unsigned int timeFrames, const unsigned int timeframesDeltaIDC, const std::vector<uint32_t>& crus);

  /// constructor for creating and object without grouped input IDCs
  /// \param timeFrames number of time frames which will be stored
  /// \param timeframesDeltaIDC number of time frames stored for each DeltaIDC object
  IDCFactorization(const unsigned int timeFrames, const unsigned int timeframesDeltaIDC, const std::vector<uint32_t>& crus) : IDCFactorization(std::array<unsigned char, Mapper::NREGIONS>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, std::array<unsigned char, Mapper::NREGIONS>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, std::array<unsigned char, Mapper::NREGIONS>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, std::array<unsigned char, Mapper::NREGIONS>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 0, timeFrames, timeframesDeltaIDC, crus){};

  /// default constructor for ROOT I/O
  IDCFactorization() = default;

  /// destructor
  ~IDCFactorization();

  /// returns sides for CRUs
  /// \param crus crus which will be checked for their side
  static std::vector<o2::tpc::Side> getSides(const std::vector<uint32_t>& crus);

  /// calculate I_0(r,\phi) = <I(r,\phi,t)>_t
  /// calculate I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  /// calculate \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  /// \param norm normalize IDCs to pad size
  void factorizeIDCs(const bool norm);

  /// calculate I_0(r,\phi) = <I(r,\phi,t)>_t
  void calcIDCZero(const bool norm);

  /// fill I_0 values in case of dead pads,FECs etc.
  void fillIDCZeroDeadPads();

  /// create status map for pads which are dead or delivering extremly high values (static outliers will be mapped)
  void createStatusMap();

  /// calculate I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  void calcIDCOne();

  /// calculate \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  void calcIDCDelta();

  /// calculate I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  template <typename DataVec>
  static void calcIDCOne(const DataVec& idcsData, const int idcsPerCRU, const int integrationIntervalOffset, const unsigned int indexOffset, const CRU cru, std::vector<float>& idcOneTmp, std::vector<unsigned int>& weights, const IDCZero* idcZero, const CalDet<PadFlags>* flagMap = nullptr, const bool usePadStatusMap = false);

  /// \return returns the stored grouped and integrated IDC
  /// \param sector sector
  /// \param region region
  /// \param grow row in the region of the grouped IDCs
  /// \param gpad pad number of the grouped IDCs
  /// \param integrationInterval integration interval
  float getIDCValGrouped(const unsigned int sector, const unsigned int region, const unsigned int grow, unsigned int gpad, unsigned int integrationInterval) const { return mIDCs[sector * Mapper::NREGIONS + region][integrationInterval][mOffsRow[region][grow] + gpad]; }

  /// \return returns the stored value for local ungrouped pad row and ungrouped pad
  /// \param sector sector
  /// \param region region
  /// \param urow row of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  /// \param integrationInterval integration interval
  float getIDCValUngrouped(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad, unsigned int integrationInterval) const;

  /// \return returns the stored IDC0 value for local ungrouped pad row and ungrouped pad
  /// \param sector sector
  /// \param region region
  /// \param urow row of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  float getIDCZeroVal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad) const { return mIDCZero[mSideIndex[Sector(sector).side()]].getValueIDCZero(getIndexUngrouped(sector, region, urow, upad, 0)); }

  /// \return returns the stored DeltaIDC value for local ungrouped pad row and ungrouped pad
  /// \param sector sector
  /// \param region region
  /// \param urow row of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  /// \param chunk chunk of the Delta IDC (can be obtained with getLocalIntegrationInterval())
  /// \param localintegrationInterval local integration interval for chunk (can be obtained with getLocalIntegrationInterval())
  float getIDCDeltaVal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad, unsigned int chunk, unsigned int localintegrationInterval) const { return mIDCDelta[mSideIndex[Sector(sector).side()]][chunk].getValue(getIndexUngrouped(sector, region, urow, upad, localintegrationInterval)); }

  /// \return returns index of integration interval in the chunk from global integration interval
  /// \param integrationInterval integration interval
  /// \param chunk which will be set in the function
  /// \param localintegrationInterval local integration interval for chunk which will be set in the function
  void getLocalIntegrationInterval(const unsigned int integrationInterval, unsigned int& chunk, unsigned int& localintegrationInterval) const;

  /// \return returns number of timeframes for which the IDCs are stored
  unsigned int getNTimeframes() const { return mTimeFrames; }

  /// \return returns the number of stored integration intervals for given Delta IDC chunk (dropped TFs taken into account)
  /// \param chunk chunk of Delta IDC
  unsigned long getNIntegrationIntervalsInChunk(const unsigned int chunk) const;

  /// \return returns the number of stored integration intervals up to a given Delta IDC chunk (dropped TFs taken into account)
  /// \param chunk chunk of Delta IDC
  unsigned long getNIntegrationIntervalsToChunk(const unsigned int chunk) const;

  /// \return returns the total number of stored integration intervals (dropped TFs not taken into account)
  unsigned long getNIntegrationIntervals(const int cru) const;

  /// \return returns the total number of stored integration intervals (dropped TFs are taken into account)
  unsigned long getNIntegrationIntervals() const;

  /// \return returns stored IDC0 I_0(r,\phi) = <I(r,\phi,t)>_t
  /// \param side TPC side
  const std::vector<float>& getIDCZeroVec(const o2::tpc::Side side) const { return mIDCZero[mSideIndex[side]].mIDCZero; }

  /// \return returns stored IDC0 I_0(r,\phi) = <I(r,\phi,t)>_t
  /// \param side TPC side
  const IDCZero& getIDCZero(const o2::tpc::Side side) const { return mIDCZero[mSideIndex[side]]; }

  /// \return returns stored IDC1 I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  /// \param side TPC side
  const std::vector<float>& getIDCOneVec(const o2::tpc::Side side) const { return mIDCOne[mSideIndex[side]].mIDCOne; }

  /// \return returns stored IDC1 I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  /// \param side TPC side
  const IDCOne& getIDCOne(const o2::tpc::Side side) const { return mIDCOne[mSideIndex[side]]; }

  /// \return returns stored IDCDelta \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  /// \param side TPC side
  /// \param chunk chunk of Delta IDC
  const std::vector<float>& getIDCDeltaValuesUncompressed(const unsigned int chunk, const o2::tpc::Side side) const { return mIDCDelta[mSideIndex[side]][chunk].getIDCDelta(); }

  /// \return returns stored IDCDelta \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  /// \param chunk chunk of Delta IDC
  const auto& getIDCDeltaUncompressed(const unsigned int chunk, const Side side) const& { return mIDCDelta[mSideIndex[side]][chunk]; }

  /// \return returns returns stored IDCDelta \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) ) using move semantics
  auto getIDCDeltaUncompressed(const unsigned int chunk, const Side side) && { return std::move(mIDCDelta[mSideIndex[side]][chunk]); }

  /// \return creates and returns medium compressed IDCDelta \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  /// \param chunk chunk of Delta IDC
  auto getIDCDeltaMediumCompressed(const unsigned int chunk, const Side side) const { return IDCDeltaCompressionHelper<unsigned short>::getCompressedIDCs(mIDCDelta[mSideIndex[side]][chunk]); }

  /// \return creates and returns high compressed IDCDelta \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  /// \param chunk chunk of Delta IDC
  auto getIDCDeltaHighCompressed(const unsigned int chunk, const Side side) const { return IDCDeltaCompressionHelper<unsigned char>::getCompressedIDCs(mIDCDelta[mSideIndex[side]][chunk]); }

  /// \return returns number of chunks for Delta IDCs
  unsigned int getNChunks(const Side side) const { return mIDCDelta[mSideIndex[side]].size(); }

  /// \return returns vector of structs containing IDC0
  const auto& getIDCZero() const { return mIDCZero; }

  /// \return returns vector of structs containing IDC1
  const auto& getIDCOne() const& { return mIDCOne; }

  /// \return returns vector of structs containing IDC1 using move semantics
  auto getIDCOne() && { return std::move(mIDCOne); }

  /// \return returns grouped IDCs
  const auto& getIDCs() const { return mIDCs; }

  /// \return returns vector of processed CRUs
  auto getCRUs() const { return mCRUs; }

  // get number of TFs in which the DeltaIDCs are split/stored
  unsigned int getTimeFramesDeltaIDC() const { return mTimeFramesDeltaIDC; }

  /// \return returns the number of threads used for some of the calculations
  static int getNThreads() { return sNThreads; }

  /// set the IDC data
  /// \param idcs vector containing the IDCs
  /// \param cru CRU
  /// \param timeframe time frame of the IDCs
  void setIDCs(std::vector<float>&& idcs, const unsigned int cru, const unsigned int timeframe) { mIDCs[cru][timeframe] = std::move(idcs); }

  /// set the number of threads used for some of the calculations
  /// \param nThreads number of threads
  static void setNThreads(const int nThreads) { sNThreads = nThreads; }

  /// \param minIDCDeltaValue minimum IDC delta value for compressed IDC delta
  static void setMinCompressedIDCDelta(const float minIDCDeltaValue) { o2::conf::ConfigurableParam::setValue<float>("TPCIDCCompressionParam", "minIDCDeltaValue", minIDCDeltaValue); }

  /// \param maxIDCDeltaValue maximum IDC delta value for compressed IDC delta
  static void setMaxCompressedIDCDelta(const float maxIDCDeltaValue) { o2::conf::ConfigurableParam::setValue<float>("TPCIDCCompressionParam", "maxIDCDeltaValue", maxIDCDeltaValue); }

  /// draw IDCs for one sector for one integration interval
  /// \param sector sector which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCsSector(const unsigned int sector, const unsigned int integrationInterval, const float minZ = 0, const float maxZ = -1, const std::string filename = "IDCsSector.pdf") const { drawIDCHelper(false, Sector(sector), integrationInterval, filename, minZ, maxZ); }

  /// draw IDC zero I_0(r,\phi) = <I(r,\phi,t)>_t
  /// \param sector sector which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCZeroSector(const unsigned int sector, const float minZ = 0, const float maxZ = -1, const std::string filename = "IDCZeroSector.pdf") const { drawIDCZeroHelper(false, Sector(sector), filename, minZ, maxZ); }

  /// draw IDCDelta for one sector for one integration interval
  /// \param sector sector which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param compression compression of Delta IDCs. (setMaxCompressedIDCDelta() should be called first in case of non standard compression parameter)
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCDeltaSector(const unsigned int sector, const unsigned int integrationInterval, const float minZ = 0, const float maxZ = -1, const IDCDeltaCompression compression = IDCDeltaCompression::NO, const std::string filename = "IDCDeltaSector.pdf") const { drawIDCDeltaHelper(false, Sector(sector), integrationInterval, compression, filename, minZ, maxZ); }

  /// draw IDCs for one side for one integration interval
  /// \param side side which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCsSide(const o2::tpc::Side side, const unsigned int integrationInterval, const float minZ = 0, const float maxZ = -1, const std::string filename = "IDCsSide.pdf") const { drawIDCHelper(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), integrationInterval, filename, minZ, maxZ); }

  /// draw IDC zero I_0(r,\phi) = <I(r,\phi,t)>_t
  /// \param side side which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCZeroSide(const o2::tpc::Side side, const float minZ = 0, const float maxZ = -1, const std::string filename = "IDCZeroSide.pdf") const { drawIDCZeroHelper(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), filename, minZ, maxZ); }

  /// draw IDCDelta for one side for one integration interval
  /// \param side side which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param compression compression of Delta IDCs. (setMaxCompressedIDCDelta() should be called first in case of non standard compression parameter)
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCDeltaSide(const o2::tpc::Side side, const unsigned int integrationInterval, const float minZ = 0, const float maxZ = -1, const IDCDeltaCompression compression = IDCDeltaCompression::NO, const std::string filename = "IDCDeltaSide.pdf") const { drawIDCDeltaHelper(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), integrationInterval, compression, filename, minZ, maxZ); }

  /// draw the status map for the flags (for debugging) for a sector
  /// \param sector sector which will be drawn
  /// \flag flag which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawPadStatusFlagsMapSector(const unsigned int sector, const PadFlags flag = PadFlags::flagSkip, const std::string filename = "PadStatusFlags_Sector.pdf") const { drawPadFlagMap(false, Sector(sector), filename, flag); }

  /// draw the status map for the flags (for debugging) for a full side
  /// \param side side which will be drawn
  /// \flag flag which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawPadStatusFlagsMapSide(const o2::tpc::Side side, const PadFlags flag = PadFlags::flagSkip, const std::string filename = "PadStatusFlags_Side.pdf") const { drawPadFlagMap(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), filename, flag); }

  /// dump object to disc
  /// \param outFileName name of the output file
  /// \param outName name of the object in the output file
  void dumpToFile(const char* outFileName = "IDCFactorized.root", const char* outName = "IDCFactorized") const;

  /// dump the IDC0 to file
  void dumpIDCZeroToFile(const Side side, const char* outFileName = "IDCZero.root", const char* outName = "IDC0") const;

  /// dump the IDC1 to file
  void dumpIDCOneToFile(const Side side, const char* outFileName = "IDCOne.root", const char* outName = "IDC1") const;

  /// \param integrationIntervals number of integration intervals which will be dumped to the tree (-1: all integration intervalls)
  /// \param outFileName name of the output file
  void dumpToTree(int integrationIntervals = -1, const char* outFileName = "IDCTree.root") const;

  /// \returns vector containing the number of integration intervals for each stored TF (dropped TFs not taken into account)
  /// \param cru cru which is used for the lookup (cru=-1: automatic cru lookup)
  std::vector<unsigned int> getIntegrationIntervalsPerTF(const int cru = -1) const;

  /// \returns vector containing the number of integration intervals for each stored TF (dropped TFs taken into account)
  std::vector<unsigned int> getAllIntegrationIntervalsPerTF() const;

  /// resetting aggregated IDCs
  void reset();

  /// setting a gain map from a file
  /// \param inpFile input file containing some caldet
  /// \param mapName name of the caldet
  void setGainMap(const char* inpFile, const char* mapName);

  /// setting a gain map from a file
  void setGainMap(const CalDet<float>& gainmap);

  /// setting a map containing status flags for each pad
  /// \param inpFile input file containing the object
  /// \param mapName name of the caldet
  void setPadFlagMap(const char* inpFile, const char* mapName);

  /// setting a map containing status flags for each pad
  void setPadFlagMap(const CalDet<PadFlags>& flagmap);

  /// setting the usage of the pad-by-pad status map during the factorization of the IDCs
  void setUsePadStatusMap(const bool usePadStatusMap) { mUsePadStatusMap = usePadStatusMap; }

  /// \return returns whether the pad-by-pad status map will be used during the factorization of the IDCs
  bool getUsePadStatusMap() const { return mUsePadStatusMap; }

  /// writing the pad status map to file
  /// \param outFile output file name
  /// \param mapName output name of the object
  void dumpPadFlagMap(const char* outFile, const char* mapName);

  /// \return returns pointer to pad status map
  CalDet<PadFlags>* getPadStatusMapPtr() const { return mPadFlagsMap.get(); }

  /// \return returns unique_ptr to pad status map
  std::unique_ptr<CalDet<PadFlags>> getPadStatusMap() { return std::move(mPadFlagsMap); }

  /// \return returns TPC sides for which the factorization is performed
  const std::vector<Side>& getSides() const { return mSides; }

 private:
  const unsigned int mTimeFrames{};                                 ///< number of timeframes which are stored
  const unsigned int mTimeFramesDeltaIDC{};                         ///< number of timeframes of which Delta IDCs are stored
  std::array<std::vector<std::vector<float>>, CRU::MaxCRU> mIDCs{}; ///< grouped and integrated IDCs for the whole TPC. CRU -> time frame -> IDCs
  std::vector<IDCZero> mIDCZero{};                                  ///< sides -> I_0(r,\phi) = <I(r,\phi,t)>_t
  std::vector<IDCOne> mIDCOne{};                                    ///< I_1(t) = sides -> <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  std::vector<std::vector<IDCDelta<float>>> mIDCDelta{};            ///< uncompressed: sides -> chunk -> Delta IDC: \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  inline static int sNThreads{1};                                   ///< number of threads which are used during the calculations
  std::unique_ptr<CalDet<float>> mGainMap;                          ///<! static Gain map object used for filling missing IDC_0 values
  std::unique_ptr<CalDet<PadFlags>> mPadFlagsMap;                   ///< status flag for each pad (i.e. if the pad is dead)
  bool mInputGrouped{false};                                        ///< flag which is set to true if the input IDCs are grouped (checked via the grouping parameters from the constructor)
  bool mUsePadStatusMap{false};                                     ///< flag for using the pad-by-pad status map during the factorization of the IDCs
  const std::vector<uint32_t> mCRUs{};                              ///< CRUs to process in this instance
  std::array<unsigned int, SIDES> mSideIndex{0, 1};                 ///< index to mIDCZero, mIDCOne and mIDCDelta for TPC side
  std::vector<Side> mSides{};                                       ///< processed TPC sides
  std::vector<unsigned int> mIntegrationIntervalsPerTF{};           ///< storage of integration intervals per TF (taken dropped TFs into account)

  /// helper function for drawing IDCDelta
  void drawIDCDeltaHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const IDCDeltaCompression compression, const std::string filename, const float minZ, const float maxZ) const;

  /// helper function for drawing IDCs
  void drawIDCHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const std::string filename, const float minZ, const float maxZ) const;

  /// helper function for drawing IDCZero
  void drawIDCZeroHelper(const bool type, const Sector sector, const std::string filename, const float minZ, const float maxZ) const;

  /// get time frame and index of integrationInterval in the TF
  void getTF(const unsigned int region, unsigned int integrationInterval, unsigned int& timeFrame, unsigned int& interval) const;

  /// \returns chunk from timeframe
  unsigned int getChunk(const unsigned int timeframe) const;

  /// \return returns number of TFs for given chunk
  unsigned int getNTFsPerChunk(const unsigned int chunk) const;

  /// helper function for drawing
  void drawPadFlagMap(const bool type, const Sector sector, const std::string filename, const PadFlags flag) const;

  ClassDefNV(IDCFactorization, 1)
};

} // namespace o2::tpc

#endif
