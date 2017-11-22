// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Clusterer.h
/// \brief Base class for TPC clusterer
/// \author Sebastian klewin
#ifndef ALICEO2_TPC_Clusterer_H_
#define ALICEO2_TPC_Clusterer_H_

#include <vector>
#include <memory>

#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2{
namespace TPC {

class Digit;

/// \class Clusterer
/// \brief Base Class for TPC clusterer
class Clusterer {
  protected:
    using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

  public:

    /// Default Constructor
    Clusterer() = delete;

    /// Constructor
    /// \param rowsMax Max number of rows to process
    /// \param padsMax Max number of pads to process
    /// \param timeBinsMax Max number of timebins to process
    /// \param minQMax Minimum peak charge for cluster
    /// \param requirePositiveCharge Positive charge is required
    /// \param requireNeighbouringPad Requires at least 2 adjecent pads with charge above threshold
    Clusterer(
        int rowsMax = 18,
        int padsMax = 138,
        int timeBinsMax = 1024,
        int minQMax = 5,
        bool requirePositiveCharge = true,
        bool requireNeighbouringPad = true);

    /// Destructor
    virtual ~Clusterer() = default;

    /// Processing all digits
    /// \param digits Container with TPC digits
    /// \param mcDigitTruth MC Digit Truth container
    /// \param eventCount event counter
    /// \return Container with clusters
    virtual void Process(std::vector<o2::TPC::Digit> const &digits, MCLabelContainer const* mcDigitTruth, int eventCount) = 0;
    virtual void Process(std::vector<std::unique_ptr<Digit>>& digits, MCLabelContainer const* mcDigitTruth, int eventCount) = 0;

    void setRowsMax(int val)                    { mRowsMax = val; };
    void setPadsMax(int val)                    { mPadsMax = val; };
    void setTimeBinsMax(int val)                { mTimeBinsMax = val; };
    void setMinQMax(float val)                  { mMinQMax = val; };
    void setRequirePositiveCharge(bool val)     { mRequirePositiveCharge = val; };
    void setRequireNeighbouringPad(bool val)    { mRequireNeighbouringPad = val; };

    int     getRowsMax()                  const { return mRowsMax; };
    int     getPadsMax()                  const { return mPadsMax; };
    int     getTimeBinsMax()              const { return mTimeBinsMax; };
    float   getMinQMax()                  const { return mMinQMax; };
    bool    hasRequirePositiveCharge()    const { return mRequirePositiveCharge; };
    bool    hasRequireNeighbouringPad()   const { return mRequireNeighbouringPad; };

  protected:

    int     mRowsMax;                       ///< Maximum row number
    int     mPadsMax;                       ///< Maximum pad number
    int     mTimeBinsMax;                   ///< Maximum time bin
    float   mMinQMax;                       ///< Minimun Qmax for cluster
    bool    mRequirePositiveCharge;         ///< If true, require charge > 0
    bool    mRequireNeighbouringPad;        ///< If true, require 2+ pads minimum

};

//________________________________________________________________________
inline Clusterer::Clusterer(int rowsMax, int padsMax, int timeBinsMax, int minQMax,
    bool requirePositiveCharge, bool requireNeighbouringPad)
  : mRowsMax(rowsMax)
  , mPadsMax(padsMax)
  , mTimeBinsMax(timeBinsMax)
  , mMinQMax(minQMax)
  , mRequirePositiveCharge(requirePositiveCharge)
  , mRequireNeighbouringPad(requireNeighbouringPad)
{}

}
}


#endif
