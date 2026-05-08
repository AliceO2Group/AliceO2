// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "IndexJSONHelpers.h"

#include <rapidjson/reader.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/error/en.h>

#include <stack>
#include <iostream>

namespace o2::framework
{
namespace
{
struct IndexRecordsReader : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, IndexRecordsReader> {
  using Ch = rapidjson::UTF8<>::Ch;
  using SizeType = rapidjson::SizeType;

  enum struct State {
    IN_START,
    IN_LIST,
    IN_RECORD,
    IN_ERROR
  };

  std::stack<State> states;
  std::ostringstream debug;

  std::vector<o2::soa::IndexRecord> records;
  std::string currentKey;
  std::string label;
  std::string columnLabel;
  std::string matcherStr;
  o2::soa::IndexKind kind;
  int pos;

  IndexRecordsReader()
  {
    debug << ">>> Start" << std::endl;
    states.push(State::IN_START);
  }

  bool StartArray()
  {
    debug << "StartArray()" << std::endl;
    if (states.top() == State::IN_START && currentKey.compare("records") == 0) {
      states.push(State::IN_LIST);
      return true;
    }
    states.push(State::IN_ERROR);
    return false;
  }

  bool EndArray(SizeType)
  {
    debug << "EndArray()" << std::endl;
    if (states.top() == State::IN_LIST) {
      // records done
      states.pop();
      return true;
    }
    states.push(State::IN_ERROR);
    return false;
  }

  bool Key(const Ch* str, SizeType, bool)
  {
    debug << "Key(" << str << ")" << std::endl;
    currentKey = str;
    if (states.top() == State::IN_START) {
      if (currentKey.compare("records") == 0) {
        return true;
      }
    }

    if (states.top() == State::IN_RECORD) {
      if (currentKey.compare("label") == 0) {
        return true;
      }
      if (currentKey.compare("matcher") == 0) {
        return true;
      }
      if (currentKey.compare("column") == 0) {
        return true;
      }
      if (currentKey.compare("kind") == 0) {
        return true;
      }
      if (currentKey.compare("pos") == 0) {
        return true;
      }
    }

    states.push(State::IN_ERROR);
    return false;
  }

  bool StartObject()
  {
    debug << "StartObject()" << std::endl;
    if (states.top() == State::IN_START) {
      return true;
    }

    if (states.top() == State::IN_LIST) {
      states.push(State::IN_RECORD);
      label = "";
      kind = soa::IndexKind::IdxInvalid;
      pos = -2;
      return true;
    }

    states.push(State::IN_ERROR);
    return false;
  }

  bool EndObject(SizeType)
  {
    debug << "EndObject()" << std::endl;
    if (states.top() == State::IN_RECORD) {
      states.pop();
      // add a record
      records.emplace_back(label, DataSpecUtils::fromString(matcherStr), columnLabel, kind, pos);
      return true;
    }

    if (states.top() == State::IN_START) {
      return true;
    }

    states.push(State::IN_ERROR);
    return false;
  }

  bool Uint(unsigned i)
  {
    debug << "Uint(" << i << ") passed to Int()" << std::endl;
    return Int(i);
  }

  bool Int(int i)
  {
    debug << "Int(" << i << ")" << std::endl;
    if (states.top() == State::IN_RECORD) {
      if (currentKey.compare("kind") == 0) {
        kind = (soa::IndexKind)i;
        return true;
      }
      if (currentKey.compare("pos") == 0) {
        pos = i;
        return true;
      }
    }

    states.push(State::IN_ERROR);
    return false;
  }

  bool String(const Ch* str, SizeType, bool)
  {
    debug << "String(" << str << ")" << std::endl;
    if (states.top() == State::IN_RECORD) {
      if (currentKey.compare("label") == 0) {
        label = str;
        return true;
      }
      if (currentKey.compare("column") == 0) {
        columnLabel = str;
        return true;
      }
      if (currentKey.compare("matcher") == 0) {
        matcherStr = str;
        return true;
      }
    }

    states.push(State::IN_ERROR);
    return false;
  }
};
} // namespace

std::vector<o2::soa::IndexRecord> IndexJSONHelpers::read(std::istream& s)
{
  rapidjson::Reader reader;
  rapidjson::IStreamWrapper isw(s);
  IndexRecordsReader irreader;

  bool ok = reader.Parse(isw, irreader);

  if (!ok) {
    throw framework::runtime_error_f("Cannot parse serialized index records vector, error: %s at offset: %d", rapidjson::GetParseError_En(reader.GetParseErrorCode()), reader.GetErrorOffset());
  }
  return irreader.records;
}

namespace
{
void writeRecords(rapidjson::Writer<rapidjson::OStreamWrapper>& w, std::vector<o2::soa::IndexRecord>& records)
{
  for (auto& r : records) {
    w.StartObject();
    w.Key("label");
    w.String(r.label.c_str());
    w.Key("matcher");
    w.String(DataSpecUtils::describe(r.matcher).c_str());
    w.Key("column");
    w.String(r.columnLabel.c_str());
    w.Key("kind");
    w.Int((int)r.kind);
    w.Key("pos");
    w.Int(r.pos);
    w.EndObject();
  }
}
} // namespace

void IndexJSONHelpers::write(std::ostream& o, std::vector<o2::soa::IndexRecord>& irs)
{
  rapidjson::OStreamWrapper osw(o);
  rapidjson::Writer<rapidjson::OStreamWrapper> w(osw);
  w.StartObject();
  w.Key("records");
  w.StartArray();
  writeRecords(w, irs);
  w.EndArray();
  w.EndObject();
}
} // namespace o2::framework
