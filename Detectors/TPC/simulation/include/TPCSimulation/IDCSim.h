// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file IDCSim.h
/// \brief class for integration of IDCs
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Apr 16, 2021

#ifndef ALICEO2_TPC_IDCSIM_H_
#define ALICEO2_TPC_IDCSIM_H_

#include <vector>
#include <array>
#include "DataFormatsTPC/Constants.h"
#include "CommonConstants/LHCConstants.h"
#include "TPCBase/Mapper.h"
#include <gsl/span>
#include "Rtypes.h"

namespace o2::utils
{
class TreeStreamRedirector;
}

namespace o2::tpc
{

class Digit;

/// \class IDCSim
/// This class is for the integration of IDCs for one sector.
/// The input has to be provided for one TF.
/// The IDCs are stored per CRU for all integration intervals.

class IDCSim
{
 public:
  /// constructor
  /// \param sector sector for which the data is processed
  /// \param nOrbits length of integration intervals
  IDCSim(const unsigned int sector = 0, const unsigned int nOrbits = 12) : mSector{sector}, mNOrbits{nOrbits} {}

  // integrate IDCs for one TF
  /// \param digits digits for one sector for one Time Frame
  void integrateDigitsForOneTF(const gsl::span<const o2::tpc::Digit>& digits);

  /// set number of orbits per TF which is used to determine the size of the vectors etc.
  /// \param nOrbitsPerTF number of orbits per TF
  static void setNOrbitsPerTF(const unsigned int nOrbitsPerTF) { mOrbitsPerTF = nOrbitsPerTF; }

  /// for debugging: dumping IDCs to ROOT file
  /// \param filename name of the output file
  void dumpIDCs(const char* outFileName, const char* outName = "IDCSim") const;

  /// for debugging: creating debug tree for integrated IDCs
  /// \param nameTree name of the output file
  void createDebugTree(const char* nameTree) const;

  /// for debugging: creating debug tree for integrated IDCs for all objects which are in the same file
  /// \param nameTree name of the output file
  /// \param filename name of the input file containing the objects
  static void createDebugTreeForAllSectors(const char* nameTree, const char* filename);

  /// \return returns the IDCs for all regions
  const auto& get() const { return mIDCs[!mBufferIndex]; }

  /// \return returns the sector for which the IDCs are integrated
  unsigned int getSector() const { return mSector; }

  /// \return returns the number of orbits for one TF
  static unsigned int getNOrbitsPerTF() { return mOrbitsPerTF; }

  /// \return returns number of orbits used for each integration interval
  unsigned int getNOrbitsPerIntegrationInterval() const { return mNOrbits; }

  /// \return returns number of time stamps per integration interval (should be 5346)
  unsigned int getNTimeStampsPerIntegrationInterval() const { return mTimeStampsPerIntegrationInterval; }

  /// \return returns the check if an additional interval is used (if the last integration interval can also be empty)
  bool additionalInterval() const { return mAddInterval; }

  /// \return returns maximum number of integration intervals used for one TF
  unsigned int getNIntegrationIntervalsPerTF() const { return mIntegrationIntervalsPerTF; }

  /// \return number time stamps which remain in one TF and will be buffered to the next TF
  unsigned int getNTimeStampsRemainder() const { return mTimeStampsRemainder; }

  /// \return offset from last time bin
  int getTimeBinsOff() const { return mTimeBinsOff; }

  /// draw ungrouped IDCs
  /// \param integrationInterval integration interval for which the IDCs will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCs(const unsigned int integrationInterval = 0, const std::string filename = "IDCs.pdf") const;

  /// \return returns maximum number of IDCs per region for all integration intervals
  unsigned int getMaxIDCs(const unsigned int region) const { return mMaxIDCs[region]; }

 private:
  inline static unsigned int mOrbitsPerTF{256};                                                                                               ///< length of one TF in units of orbits
  const unsigned int mSector{};                                                                                                               ///< sector for which the IDCs are integrated
  const unsigned int mNOrbits{12};                                                                                                            ///< integration intervals of IDCs in units of orbits
  const unsigned int mTimeStampsPerIntegrationInterval{(o2::constants::lhc::LHCMaxBunches * mNOrbits) / o2::tpc::constants::LHCBCPERTIMEBIN}; ///< number of time stamps for each integration interval (5346)
  const bool mAddInterval{(mOrbitsPerTF % mNOrbits) > 0 ? true : false};                                                                      ///< if the division has a remainder 256/12=21.333 then add an additional integration interval
  const unsigned int mIntegrationIntervalsPerTF{mOrbitsPerTF / mNOrbits + mAddInterval};                                                      ///< number of integration intervals per TF. Add 1: 256/12=21.333
  const unsigned int mTimeStampsRemainder{mTimeStampsPerIntegrationInterval * (mOrbitsPerTF % mNOrbits) / mNOrbits};                          ///< number time stamps which remain in one TF and will be buffered to the next TF
  int mTimeBinsOff{};                                                                                                                         ///< offset from last time bin
  bool mBufferIndex{false};                                                                                                                   ///< index for the buffer
  const std::array<unsigned int, Mapper::NREGIONS> mMaxIDCs{
    Mapper::PADSPERREGION[0] * mIntegrationIntervalsPerTF, // region 0
    Mapper::PADSPERREGION[1] * mIntegrationIntervalsPerTF, // region 1
    Mapper::PADSPERREGION[2] * mIntegrationIntervalsPerTF, // region 2
    Mapper::PADSPERREGION[3] * mIntegrationIntervalsPerTF, // region 3
    Mapper::PADSPERREGION[4] * mIntegrationIntervalsPerTF, // region 4
    Mapper::PADSPERREGION[5] * mIntegrationIntervalsPerTF, // region 5
    Mapper::PADSPERREGION[6] * mIntegrationIntervalsPerTF, // region 6
    Mapper::PADSPERREGION[7] * mIntegrationIntervalsPerTF, // region 7
    Mapper::PADSPERREGION[8] * mIntegrationIntervalsPerTF, // region 8
    Mapper::PADSPERREGION[9] * mIntegrationIntervalsPerTF  // region 9
  };                                                       ///< maximum number of IDCs per region
  std::array<std::vector<float>, Mapper::NREGIONS> mIDCs[2]{
    {std::vector<float>(mMaxIDCs[0]),  // region 0
     std::vector<float>(mMaxIDCs[1]),  // region 1
     std::vector<float>(mMaxIDCs[2]),  // region 2
     std::vector<float>(mMaxIDCs[3]),  // region 3
     std::vector<float>(mMaxIDCs[4]),  // region 4
     std::vector<float>(mMaxIDCs[5]),  // region 5
     std::vector<float>(mMaxIDCs[6]),  // region 6
     std::vector<float>(mMaxIDCs[7]),  // region 7
     std::vector<float>(mMaxIDCs[8]),  // region 8
     std::vector<float>(mMaxIDCs[9])}, // region 9
    {std::vector<float>(mMaxIDCs[0]),  // region 0
     std::vector<float>(mMaxIDCs[1]),  // region 1
     std::vector<float>(mMaxIDCs[2]),  // region 2
     std::vector<float>(mMaxIDCs[3]),  // region 3
     std::vector<float>(mMaxIDCs[4]),  // region 4
     std::vector<float>(mMaxIDCs[5]),  // region 5
     std::vector<float>(mMaxIDCs[6]),  // region 6
     std::vector<float>(mMaxIDCs[7]),  // region 7
     std::vector<float>(mMaxIDCs[8]),  // region 8
     std::vector<float>(mMaxIDCs[9])}  // region 9
  };                                   ///< IDCs for one sector. The array is needed to buffer the IDCs for the last integration interval

  /// \return returns the last time bin after which the buffer is switched
  unsigned int getLastTimeBinForSwitch() const;

  // set offset for the next time frame
  void setNewOffset();

  /// set all IDC values to 0
  void resetIDCs();

  /// return orbit for given timeStamp
  unsigned int getOrbit(const unsigned int timeStamp) const { return static_cast<unsigned int>((timeStamp + mTimeBinsOff) / mTimeStampsPerIntegrationInterval); }

  /// \return returns index in the vector of the mIDCs member
  /// \param timeStamp timeStamp for which the index is calculated
  /// \param region region in the sector
  /// \param row global pad row
  /// \param pad pad in row
  unsigned int getIndex(const unsigned int timeStamp, const unsigned int region, const unsigned int row, const unsigned int pad) const { return getOrbit(timeStamp) * Mapper::PADSPERREGION[region] + Mapper::getLocalPadNumber(row, pad); }

  static void createDebugTree(const IDCSim& idcsim, o2::utils::TreeStreamRedirector& pcstream);

  ClassDefNV(IDCSim, 1)
};

} // namespace o2::tpc

#endif
