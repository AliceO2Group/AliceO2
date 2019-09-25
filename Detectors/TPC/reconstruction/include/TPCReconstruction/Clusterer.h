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

#include "TPCBase/CalDet.h"

namespace o2
{
namespace tpc
{

class Digit;

/// \class Clusterer
/// \brief Base Class for TPC clusterer
class Clusterer
{
 protected:
  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

 public:
  /// Default Constructor
  Clusterer() = default;

  /// Destructor
  virtual ~Clusterer() = default;

  /// Copy constructor
  Clusterer(Clusterer const& other) = default;

  /// Processing all digits
  /// \param digits Container with TPC digits
  /// \param mcDigitTruth MC Digit Truth container
  virtual void process(std::vector<o2::tpc::Digit> const& digits, MCLabelContainer const* mcDigitTruth) = 0;
  virtual void finishProcess(std::vector<o2::tpc::Digit> const& digits, MCLabelContainer const* mcDigitTruth) = 0;

  /// Setter for noise object, noise will be added before cluster finding
  /// \param noiseObject CalDet object, containing noise simulation
  void setNoiseObject(CalDet<float>* noiseObject);

  /// Setter for pedestal object, pedestal value will be subtracted before cluster finding
  /// \param pedestalObject CalDet object, containing pedestals for each pad
  void setPedestalObject(CalDet<float>* pedestalObject);

 protected:
  CalDet<float>* mNoiseObject;    ///< Pointer to the CalDet object for noise simulation
  CalDet<float>* mPedestalObject; ///< Pointer to the CalDet object for the pedestal subtraction
};

inline void Clusterer::setNoiseObject(CalDet<float>* noiseObject)
{
  mNoiseObject = noiseObject;
}

inline void Clusterer::setPedestalObject(CalDet<float>* pedestalObject)
{
  mPedestalObject = pedestalObject;
}
} // namespace tpc
} // namespace o2

#endif
