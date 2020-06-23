// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file BaseDPLDigitizer.h
/// \brief Definition of the base digitizer task class

#ifndef ALICEO2_BASE_BASEDPLDIGITIZER_H_
#define ALICEO2_BASE_BASEDPLDIGITIZER_H_

#include <Framework/InitContext.h>
#include "DetectorsBase/Propagator.h"
#include "DataFormatsParameters/GRPObject.h"
#include <cstdint>

namespace o2
{
namespace base
{

/// class listing possible services
struct InitServices {
  using Type = std::uint32_t;
  static constexpr Type FIELD = 1 << 0;
  static constexpr Type GEOM = 1 << 1;
};

/// Class trying to reduce boilerplate (initialization) work/code for
/// digitizer tasks. In particular it will try to provide/initialize field and geometry
/// when needed. Also providing some common data members.
///
/// Deriving classes need to implement a function initDigitizerTask instead of just init.
class BaseDPLDigitizer
{
 public:
  BaseDPLDigitizer() = default;
  virtual ~BaseDPLDigitizer() = default;

  /// a constructor accepting a service code encoding the asked
  /// services in bits. Example:
  /// BaseDPLDigitizer(InitServices::FIELD | InitServices::GEOM)
  BaseDPLDigitizer(InitServices::Type servicecode);

  virtual void init(o2::framework::InitContext&) final;

  virtual void initDigitizerTask(o2::framework::InitContext& ic) = 0;

 private:
  bool mNeedField = false;
  bool mNeedGeom = false;
};

} // namespace base
} // namespace o2

#endif
