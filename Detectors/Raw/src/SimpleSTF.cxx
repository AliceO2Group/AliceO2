// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file SimpleSTF.cxx
/// \brief Mocked STF with InputRecord for standalone tests

#include "DetectorsRaw/SimpleSTF.h"
#include "Framework/DataRefUtils.h"

using namespace o2::raw;

SimpleSTF::SimpleSTF(std::vector<o2f::InputRoute>&& sch, PartsRef&& pref, Messages&& msg)
  : schema{std::move(sch)},
    partsRef{std::move(pref)},
    messages{std::move(msg)},
    span{[this](size_t i, size_t part) {                     // getter for the DataRef of a part in the input "i"
           auto ref = this->partsRef[i].first + (part << 1); // entry of the header for this part, the payload follows
           auto header = static_cast<char const*>(this->messages[ref]->data());
           auto payload = static_cast<char const*>(this->messages[ref + 1]->data());
           return o2f::DataRef{nullptr, header, payload};
         },
         [this](size_t i) { // getter for the nparts in the input "i"
           return i < partsRef.size() ? partsRef[i].second : 0;
         },
         this->partsRef.size()},
    record{schema, span}
{
}
