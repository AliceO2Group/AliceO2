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
#include "DIMessages.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include <fairlogger/Logger.h>

using namespace rapidjson;

Document toJson(const DIMessages::RegisterDevice::Specs::Input& input)
{
  Document doc;
  doc.SetObject();
  auto& alloc = doc.GetAllocator();

  doc.AddMember("binding", Value(input.binding.c_str(), alloc), alloc);
  doc.AddMember("sourceChannel", Value(input.sourceChannel.c_str(), alloc), alloc);
  doc.AddMember("timeslice", Value((uint64_t)input.timeslice), alloc);

  if (input.origin.has_value()) {
    doc.AddMember("origin", Value(input.origin.value().c_str(), alloc), alloc);
  }
  if (input.description.has_value()) {
    doc.AddMember("description", Value(input.description.value().c_str(), alloc), alloc);
  }
  if (input.subSpec.has_value()) {
    doc.AddMember("subSpec", Value(input.subSpec.value()), alloc);
  }

  return doc;
}

Document toJson(const DIMessages::RegisterDevice::Specs::Output& output)
{
  Document doc;
  doc.SetObject();
  auto& alloc = doc.GetAllocator();

  doc.AddMember("binding", Value(output.binding.c_str(), alloc), alloc);
  doc.AddMember("channel", Value(output.channel.c_str(), alloc), alloc);
  doc.AddMember("timeslice", Value((uint64_t)output.timeslice), alloc);
  doc.AddMember("maxTimeslices", Value((uint64_t)output.maxTimeslices), alloc);

  doc.AddMember("origin", Value(output.origin.c_str(), alloc), alloc);
  doc.AddMember("description", Value(output.description.c_str(), alloc), alloc);
  if (output.subSpec.has_value()) {
    doc.AddMember("subSpec", Value(output.subSpec.value()), alloc);
  }

  return doc;
}

Document toJson(const DIMessages::RegisterDevice::Specs::Forward& forward)
{
  Document doc;
  doc.SetObject();
  auto& alloc = doc.GetAllocator();

  doc.AddMember("binding", Value(forward.binding.c_str(), alloc), alloc);
  doc.AddMember("channel", Value(forward.channel.c_str(), alloc), alloc);
  doc.AddMember("timeslice", Value((uint64_t)forward.timeslice), alloc);
  doc.AddMember("maxTimeslices", Value((uint64_t)forward.maxTimeslices), alloc);

  if (forward.origin.has_value()) {
    doc.AddMember("origin", Value(forward.origin.value().c_str(), alloc), alloc);
  }
  if (forward.description.has_value()) {
    doc.AddMember("description", Value(forward.description.value().c_str(), alloc), alloc);
  }
  if (forward.subSpec.has_value()) {
    doc.AddMember("subSpec", Value(forward.subSpec.value()), alloc);
  }

  return doc;
}

Document specToJson(const DIMessages::RegisterDevice::Specs& specs)
{
  Document doc;
  doc.SetObject();
  auto& alloc = doc.GetAllocator();

  doc.AddMember("rank", Value((uint64_t)specs.rank), alloc);
  doc.AddMember("nSlots", Value((uint64_t)specs.nSlots), alloc);
  doc.AddMember("inputTimesliceId", Value((uint64_t)specs.inputTimesliceId), alloc);
  doc.AddMember("maxInputTimeslices", Value((uint64_t)specs.maxInputTimeslices), alloc);

  Value inputsArray;
  inputsArray.SetArray();
  for (auto& input : specs.inputs) {
    Value inputValue;
    inputValue.CopyFrom(toJson(input), alloc);
    inputsArray.PushBack(inputValue, alloc);
  }
  doc.AddMember("inputs", inputsArray, alloc);

  Value outputsArray;
  outputsArray.SetArray();
  for (auto& output : specs.outputs) {
    Value outputValue;
    outputValue.CopyFrom(toJson(output), alloc);
    outputsArray.PushBack(outputValue, alloc);
  }
  doc.AddMember("outputs", outputsArray, alloc);

  Value forwardsArray;
  forwardsArray.SetArray();
  for (auto& forward : specs.forwards) {
    Value forwardValue;
    forwardValue.CopyFrom(toJson(forward), alloc);
    forwardsArray.PushBack(forwardValue, alloc);
  }
  doc.AddMember("forwards", forwardsArray, alloc);

  return doc;
}

std::string DIMessages::RegisterDevice::toJson()
{
  Document doc;
  doc.SetObject();
  auto& alloc = doc.GetAllocator();

  doc.AddMember("name", Value(name.c_str(), alloc), alloc);
  doc.AddMember("runId", Value(runId.c_str(), alloc), alloc);

  Value specsValue;
  specsValue.CopyFrom(specToJson(specs), alloc);
  doc.AddMember("specs", specsValue, alloc);

  StringBuffer buffer;
  Writer<StringBuffer> writer(buffer);
  doc.Accept(writer);

  return {buffer.GetString(), buffer.GetSize()};
}
