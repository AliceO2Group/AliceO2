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
#include "DataFormatsQualityControl/TimeRangeFlagCollection.h"
#include "DataFormatsQualityControl/FlagReasonFactory.h"
#include "Framework/Logger.h"

#include <iostream>
#include <boost/algorithm/string/replace.hpp>
#include <boost/tokenizer.hpp>
#include <utility>

namespace o2::quality_control
{

constexpr const char* csvHeader = "start,end,flag_id,flag_name,flag_bad,comment,source";
constexpr size_t csvColumns = 7;

TimeRangeFlagCollection::TimeRangeFlagCollection(std::string name, std::string detector, RangeInterval validityRange)
  : mName(std::move(name)), mDetID(std::move(detector)), mValidityRange(validityRange)
{}

void TimeRangeFlagCollection::insert(TimeRangeFlag&& trf)
{
  mTimeRangeFlags.insert(std::move(trf));
}

void TimeRangeFlagCollection::insert(const TimeRangeFlag& trf)
{
  mTimeRangeFlags.insert(trf);
}

size_t TimeRangeFlagCollection::size() const
{
  return mTimeRangeFlags.size();
}

void TimeRangeFlagCollection::merge(TimeRangeFlagCollection& other)
{
  if (mDetID != other.mDetID) {
    // We assume that one TimeRangeFlagCollection should correspond to one detector at most.
    // However, if this becomes annoying, we can reconsider it.
    throw std::runtime_error(
      "The detector ID of the target collection '" + mDetID + "' is different than the other '" + mDetID);
  }
  mTimeRangeFlags.merge(other.mTimeRangeFlags);
}

void TimeRangeFlagCollection::merge(const TimeRangeFlagCollection& other)
{
  TimeRangeFlagCollection otherCopy{other};
  merge(otherCopy);
}

TimeRangeFlagCollection::collection_t::const_iterator TimeRangeFlagCollection::begin() const
{
  return mTimeRangeFlags.begin();
}

TimeRangeFlagCollection::collection_t::const_iterator TimeRangeFlagCollection::end() const
{
  return mTimeRangeFlags.end();
}

void TimeRangeFlagCollection::streamTo(std::ostream& output) const
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

void TimeRangeFlagCollection::streamFrom(std::istream& input)
{
  std::string line;
  std::getline(input, line);
  if (line != csvHeader) {
    throw std::runtime_error(
      "Unsupported TRFCollection format, the first line is \"" + line + "\" instead of \"" + csvHeader + "\"");
  }

  while (std::getline(input, line)) {
    boost::tokenizer<boost::escaped_list_separator<char>> tok(line);

    TimeRangeFlag::time_type start = 0;
    TimeRangeFlag::time_type end = 0;
    FlagReason flag = FlagReasonFactory::Invalid();
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
            start = static_cast<TimeRangeFlag::time_type>(std::stoull(*it));
          }
          break;
        }
        case 1: {
          if (it->empty()) {
            LOG(error) << "Invalid line, empty end time of a flag, skipping...";
            valid = false;
          } else {
            end = static_cast<TimeRangeFlag::time_type>(std::stoull(*it));
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

std::ostream& operator<<(std::ostream& output, const TimeRangeFlagCollection& data)
{
  data.streamTo(output);
  return output;
}

const std::string& TimeRangeFlagCollection::getName() const
{
  return mName;
}

const std::string& TimeRangeFlagCollection::getDetector() const
{
  return mDetID;
}

} // namespace o2::quality_control
