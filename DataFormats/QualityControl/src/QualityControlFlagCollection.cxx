// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// O2 include
#include "DataFormatsQualityControl/QualityControlFlagCollection.h"
#include "DataFormatsQualityControl/FlagTypeFactory.h"
#include "Framework/Logger.h"

#include <iostream>
#include <boost/algorithm/string/replace.hpp>
#include <boost/tokenizer.hpp>
#include <utility>

namespace o2::quality_control
{

constexpr const char* csvHeader = "start,end,flag_id,flag_name,flag_bad,comment,source";
constexpr size_t csvColumns = 7;

QualityControlFlagCollection::QualityControlFlagCollection(std::string name, std::string detector, RangeInterval validityRange,
                                                           int runNumber, std::string periodName, std::string passName,
                                                           std::string provenance)
  : mName(std::move(name)), mDetID(std::move(detector)), mValidityRange(validityRange), mRunNumber(runNumber), mPeriodName(std::move(periodName)), mPassName(passName), mProvenance(std::move(provenance))
{
}

void QualityControlFlagCollection::insert(QualityControlFlag&& trf)
{
  mQualityControlFlags.insert(std::move(trf));
}

void QualityControlFlagCollection::insert(const QualityControlFlag& trf)
{
  mQualityControlFlags.insert(trf);
}

size_t QualityControlFlagCollection::size() const
{
  return mQualityControlFlags.size();
}

void QualityControlFlagCollection::merge(QualityControlFlagCollection& other)
{
  if (mDetID != other.mDetID) {
    // We assume that one QualityControlFlagCollection should correspond to one detector at most.
    // However, if this becomes annoying, we can reconsider it.
    throw std::runtime_error(
      "The detector ID of the target collection '" + mDetID + "' is different than the other '" + mDetID);
  }
  mQualityControlFlags.merge(other.mQualityControlFlags);
}

void QualityControlFlagCollection::merge(const QualityControlFlagCollection& other)
{
  QualityControlFlagCollection otherCopy{other};
  merge(otherCopy);
}

QualityControlFlagCollection::collection_t::const_iterator QualityControlFlagCollection::begin() const
{
  return mQualityControlFlags.begin();
}

QualityControlFlagCollection::collection_t::const_iterator QualityControlFlagCollection::end() const
{
  return mQualityControlFlags.end();
}

void QualityControlFlagCollection::streamTo(std::ostream& output) const
{
  auto escapeComma = [](const std::string& str) {
    return boost::algorithm::replace_all_copy(str, ",", "\\,");
  };
  output << csvHeader << '\n';
  for (const auto& trf : *this) {
    output << fmt::format("{},{},{},\"{}\",{:d},\"{}\",\"{}\"\n",
                          trf.getStart(), trf.getEnd(),
                          trf.getFlag().getID(), escapeComma(trf.getFlag().getName()), trf.getFlag().getBad(),
                          escapeComma(trf.getComment()), escapeComma(trf.getSource()));
  }
}

void QualityControlFlagCollection::streamFrom(std::istream& input)
{
  std::string line;
  std::getline(input, line);
  if (line != csvHeader) {
    throw std::runtime_error(
      "Unsupported TRFCollection format, the first line is \"" + line + "\" instead of \"" + csvHeader + "\"");
  }

  while (std::getline(input, line)) {
    boost::tokenizer<boost::escaped_list_separator<char>> tok(line);

    QualityControlFlag::time_type start = 0;
    QualityControlFlag::time_type end = 0;
    FlagType flag = FlagTypeFactory::Invalid();
    std::string comment;
    std::string source;
    auto it = tok.begin();
    bool valid = true;
    size_t pos = 0;
    for (; it != tok.end() && valid; pos++, it++) {
      switch (pos) {
        case 0: {
          if (it->empty()) {
            LOG(error) << "Invalid line, empty start time of a flag, skipping...";
            valid = false;
          } else {
            start = static_cast<QualityControlFlag::time_type>(std::stoull(*it));
          }
          break;
        }
        case 1: {
          if (it->empty()) {
            LOG(error) << "Invalid line, empty end time of a flag, skipping...";
            valid = false;
          } else {
            end = static_cast<QualityControlFlag::time_type>(std::stoull(*it));
          }
          break;
        }
        case 2: {
          if (it->empty()) {
            LOG(error) << "Invalid line, empty flag id, skipping...";
            valid = false;
          } else {
            flag.mId = std::stoul(*it);
          }
          break;
        }
        case 3: {
          if (it->empty()) {
            LOG(error) << "Invalid line, empty flag name, skipping...";
            valid = false;
          } else {
            flag.mName = *it;
          }
          break;
        }
        case 4: {
          if (it->empty()) {
            LOG(error) << "Invalid line, empty flag 'bad' field, skipping...";
            valid = false;
          } else {
            flag.mBad = static_cast<bool>(std::stoul(*it));
          }
          break;
        }
        case 5: {
          comment = *it;
          break;
        }
        case 6: {
          source = *it;
          break;
        }
        default: {
          LOG(error) << "More columns (" << pos + 1 << ") than expected (" << csvColumns
                     << ") in this line, skipping...";
          valid = false;
          break;
        }
      }
    }
    if (valid && pos < csvColumns) {
      LOG(error) << "Less columns (" << pos << ") than expected (" << csvColumns << ") in this line, skipping...";
    } else if (valid) {
      insert({start, end, flag, comment, source});
    }
  }
}

std::ostream& operator<<(std::ostream& output, const QualityControlFlagCollection& data)
{
  data.streamTo(output);
  return output;
}

const std::string& QualityControlFlagCollection::getName() const
{
  return mName;
}

const std::string& QualityControlFlagCollection::getDetector() const
{
  return mDetID;
}

int QualityControlFlagCollection::getRunNumber() const
{
  return mRunNumber;
}

const std::string& QualityControlFlagCollection::getPeriodName() const
{
  return mPeriodName;
}

const std::string& QualityControlFlagCollection::getPassName() const
{
  return mPassName;
}
const std::string& QualityControlFlagCollection::getProvenance() const
{
  return mProvenance;
}

} // namespace o2::quality_control
