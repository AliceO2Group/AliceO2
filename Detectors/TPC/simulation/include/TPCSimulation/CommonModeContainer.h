// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CommonModeContainer.h
/// \brief Definition of the Common Mode computation
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_CommonModeContainer_H_
#define ALICEO2_TPC_CommonModeContainer_H_

#include "TPCBase/ROC.h"
#include "TPCBase/Mapper.h"
#include <deque>

namespace o2 {
namespace TPC {

/// \class CommonModeContainer
/// This class computes and contains the Common Mode value and makes it accessible for the Digits

class CommonModeContainer{
  public:

    /// Default constructor
    CommonModeContainer();

    /// Destructor
    ~CommonModeContainer() = default;

    /// Add a single pad hit to the container
    /// \param cru CRU of the pad hit
    /// \param timeBin Time bin of the pad hit
    /// \param signal Pulse height of the pad hit
    void addDigit(const CRU cru, const int timeBin, const float signal);

    /// Clean up the container
    /// \param eventTime Time stamp of the event
    /// \param isContinuous true in case of continuous mode
    void cleanUp(int eventTime=0, bool isContinuous=true);

    /// Retrieve the common mode signal of a specific GEM stack
    /// \param cru CRU for which the common mode of the GEM stack is to be retrieved
    /// \param timeBin Time bin for which the common mode of the GEM stack is to be retrieved
    /// \return Common mode value
    float getCommonMode(const CRU cru, const int timeBin) const;

    /// Get the number of initialized time bins
    size_t getNtimeBins() const { return mCommonModeContainer.size(); }

  private:
    int                    mFirstTimeBin;           ///< Time bin which corresponds to the zeroth entry in the deque
    int                    mEffectiveTimeBin;       ///<
    /// @todo add GEM stack to mapper, instead of ROC - then remove somewhat hardcoded size of the array
    std::deque<std::array<float, ROC::MaxROC*2> >   mCommonModeContainer; ///< the array contains the 144 Common mode values (float) per timebin, which are handled within the deque
};

inline
CommonModeContainer::CommonModeContainer()
  : mFirstTimeBin(0),
    mEffectiveTimeBin(0),
    mCommonModeContainer(500)
{}

inline
float CommonModeContainer::getCommonMode(const CRU cru, const int timeBin) const
{
  if(timeBin-mFirstTimeBin < 0) return 9999.f;
  const int sector = cru.sector();
  const int gemStack = static_cast<int>(cru.gemStack());
  static const Mapper& mapper = Mapper::instance();
  const auto nPads = mapper.getNumberOfPads(cru.gemStack());
  return mCommonModeContainer[timeBin-mFirstTimeBin][4*sector+gemStack]/static_cast<float>(nPads); /// simple case when there is no external capacitance on the ROC;
}

}
}

#endif // ALICEO2_TPC_CommonModeContainer_H_
