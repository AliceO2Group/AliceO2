// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitTime.h
/// \brief Definition of the Time Bin container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_DigitTime_H_
#define ALICEO2_TPC_DigitTime_H_

#include "TPCSimulation/DigitGlobalPad.h"
#include "TPCBase/Mapper.h"

namespace o2 {
namespace TPC {
    
class Digit;
class DigitMCMetaData;

/// \class DigitTime
/// This is the third class of the intermediate Digit Containers, in which all incoming electrons from the hits are sorted into after amplification
/// The structure assures proper sorting of the Digits when later on written out for further processing.
/// This class holds the individual Pad Row containers and is contained within the CRU Container.
    
class DigitTime{
  public:
    
    /// Constructor
    /// \param mTimeBin time bin
    DigitTime();

    /// Destructor
    ~DigitTime() = default;

    void reset();

    /// Get the common mode value
    /// \param CRU CRU of the digit
    /// \return Common mode value in that time bin for a given GEM ROC
    float getCommonMode(const CRU &cru) const;

    /// Add digit to the row container
    /// \param eventID MC Event ID
    /// \param hitID MC Hit ID
    /// \param cru CRU of the digit
    /// \param row Pad row of digit
    /// \param pad Pad of digit
    /// \param charge Charge of the digit
    void setDigit(size_t eventID, size_t hitID, const CRU &cru, GlobalPadNumber globalPad, float charge);

    /// Fill output vector
    /// \param output Output container
    /// \param mcTruth MC Truth container
    /// \param debug Optional debug output container
    /// \param sector Sector ID
    /// \param timeBin Time bin
    void fillOutputContainer(std::vector<Digit> *output, dataformats::MCTruthContainer<MCCompLabel> &mcTruth,
                             std::vector<DigitMCMetaData> *debug, Sector sector, TimeBin timeBin);

  private:
    std::array <DigitGlobalPad, Mapper::getPadsInSector()> mGlobalPads; ///< Pad Container for the ADC value
    std::array <float, 4> mCommonMode;                  ///< Common mode container - 4 GEM ROCs per sector
};

inline
DigitTime::DigitTime()
  : mCommonMode(),
    mGlobalPads()
{}

inline
float DigitTime::getCommonMode(const CRU &cru) const
{
  static const Mapper& mapper = Mapper::instance();
  const auto gemStack = static_cast<int>(cru.gemStack());
  const auto nPads = mapper.getNumberOfPads(gemStack);
  return mCommonMode[gemStack]/static_cast<float>(nPads); /// simple case when there is no external capacitance on the ROC;
}

inline
void DigitTime::reset()
{
  for(auto &pad : mGlobalPads) {
      pad.reset();
    }
}

}
}

#endif // ALICEO2_TPC_DigitTime_H_
