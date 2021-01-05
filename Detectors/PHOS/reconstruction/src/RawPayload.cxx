// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PHOSReconstruction/RawPayload.h"

using namespace o2::phos;

RawPayload::RawPayload(gsl::span<const uint32_t> payloadwords, int numpages) : mPayloadWords(payloadwords.size()),
                                                                               mNumberOfPages(numpages)
{
  for (auto word : payloadwords) {
    mPayloadWords.emplace_back(word);
  }
}

void RawPayload::appendPayloadWords(const gsl::span<const uint32_t> payloadwords)
{
  for (auto word : payloadwords) {
    mPayloadWords.emplace_back(word);
  }
}

void RawPayload::reset()
{
  mPayloadWords.clear();
  mNumberOfPages = 0;
}
