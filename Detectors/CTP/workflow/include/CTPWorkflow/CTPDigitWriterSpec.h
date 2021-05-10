// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CTPDigitWriterSpec.h
/// \author Roman Lietava

#ifndef O2_CTPDIGITWRITERSPEC_H
#define O2_CTPDIGITWRITERSPEC_H

#include "Framework/DataProcessorSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "DataFormatsCTP/Digits.h"

using namespace o2::framework;
namespace o2
{
namespace ctp
{
framework::DataProcessorSpec getCTPDigitWriterSpec(bool raw = true);
}
} // namespace o2

#endif //O2_CTPDIGITWRITERSPEC_H
