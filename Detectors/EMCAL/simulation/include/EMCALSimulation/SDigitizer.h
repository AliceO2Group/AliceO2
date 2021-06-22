// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_SDIGITIZER_H
#define ALICEO2_EMCAL_SDIGITIZER_H

#include <memory>
#include <unordered_map>
#include <vector>
#include <list>

#include "Rtypes.h"  // for SDigitizer::Class, Double_t, ClassDef, etc
#include "TObject.h" // for TObject

#include "DataFormatsEMCAL/Digit.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/GeometryBase.h"
#include "EMCALBase/Hit.h"
#include "EMCALSimulation/LabeledDigit.h"

#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace emcal
{

/// \class SDigitizer
/// \brief EMCAL summed digitizer
/// \ingroup EMCALsimulation
/// \author Anders Knospe, University of Houston
/// \author Hadi Hassan, ORNL

class SDigitizer
{
 public:
  SDigitizer() = default;
  ~SDigitizer() = default;
  SDigitizer(const SDigitizer&) = delete;
  SDigitizer& operator=(const SDigitizer&) = delete;

  /// Steer conversion of hits to digits
  std::vector<o2::emcal::LabeledDigit> process(const std::vector<Hit>& hits);

  void setCurrSrcID(int v);
  int getCurrSrcID() const { return mCurrSrcID; }

  void setCurrEvID(int v);
  int getCurrEvID() const { return mCurrEvID; }

  void setGeometry(const o2::emcal::Geometry* gm) { mGeometry = gm; }

 private:
  const Geometry* mGeometry = nullptr; ///< EMCAL geometry
  int mCurrSrcID = 0;                  ///< current MC source from the manager
  int mCurrEvID = 0;                   ///< current event ID from the manager

  ClassDefNV(SDigitizer, 1);
};
} // namespace emcal
} // namespace o2

#endif /* ALICEO2_EMCAL_SDIGITIZER_H */
