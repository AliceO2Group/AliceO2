// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file SimpleSTF.h
/// \brief Mocked STF with InputRecord for standalone tests (improved version of Matthias' structure in the test_DPLRawParser)
#ifndef ALICEO2_ITSMFT_SIMPLESTF_H_
#define ALICEO2_ITSMFT_SIMPLESTF_H_

#include <vector>
#include <memory>
#include <utility>
#include <gsl/span>
#include "Framework/InputRoute.h"
#include "Framework/InputRecord.h"
#include "Framework/InputSpan.h"

namespace o2
{
namespace framework
{
struct InputRoute;
class InputRecord;
} // namespace framework
namespace raw
{
namespace o2f = o2::framework;

struct SimpleSTF {
  using PartsRef = std::vector<std::pair<int, int>>;
  using Messages = std::vector<std::unique_ptr<std::vector<char>>>;

  SimpleSTF(std::vector<o2f::InputRoute>&& sch, PartsRef&& pref, Messages&& msg);
  bool empty() const { return partsRef.size() == 0; }
  int getNLinks() const { return partsRef.size(); }
  int getNParts(int il) const { return il < getNLinks() ? partsRef[il].second : 0; }
  const gsl::span<const char> getPart(int il, int part) const { return *messages[partsRef[il].first + (part << 1) + 1].get(); }

  std::vector<o2f::InputRoute> schema;
  PartsRef partsRef; // i-th entry is the 1st entry and N parts of multipart for i-th channel in the messages
  Messages messages;
  o2f::InputSpan span;
  o2f::InputRecord record;
};

} // namespace raw
} // namespace o2

#endif
