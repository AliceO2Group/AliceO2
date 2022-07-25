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

/// @file   file-proxy.cxx
/// @author Roberto Preghenella
/// @since  2020-03-08
/// @brief  TOF compressed data analysis task

#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/CallbackService.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceSpec.h"
#include <fairmq/Device.h>
#include <fairmq/Parts.h>

#include "Headers/RAWDataHeader.h"
#include "DataFormatsTOF/RawDataFormat.h"

#include <iostream>

using namespace o2::framework;

class FileProxyTask : public Task
{
 public:
  FileProxyTask() = default;
  ~FileProxyTask() override = default;

  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  long readFLP()
  {
    if (mRDHversion == 4) {
      return readFLPv4();
    }
    if (mRDHversion == 6) {
      return readFLPv6();
    }
    return 0;
  };
  long readFLPv4();
  long readFLPv6();
  long readCONET();

  bool mStatus = false;
  bool mCONET = false;
  int mRDHversion = 4;
  bool mDumpData = false;
  std::ifstream mFile;
  char mBuffer[4194304];
};

void FileProxyTask::init(InitContext& ic)
{
  auto infilename = ic.options().get<std::string>("atc-file-proxy-input-filename");
  auto startfrom = ic.options().get<int>("atc-file-proxy-start-from");
  mCONET = ic.options().get<bool>("atc-file-proxy-conet-mode");
  mDumpData = ic.options().get<bool>("atc-file-proxy-dump-data");
  mRDHversion = ic.options().get<int>("atc-file-proxy-rdh-version");

  /** open input file **/
  std::cout << " --- Opening input file: " << infilename << std::endl;
  mFile.open(infilename, std::fstream::in | std::fstream::binary);
  if (!mFile.is_open()) {
    std::cout << " --- File is not open " << std::endl;
    mStatus = true;
  }
  std::cout << " --- Start reading from byte offset: " << startfrom << std::endl;
  mFile.seekg(startfrom, std::ios::beg);

  if (mCONET) {
    std::cout << " --- CONET mode " << std::endl;
  }
};

long FileProxyTask::readFLPv4()
{
  /** read input file **/
  char* pointer = mBuffer;
  if (!mFile.read(pointer, 64)) {
    std::cout << " --- Cannot read input file: " << strerror(errno) << std::endl;
    mStatus = true;
    return 0;
  }
  long payload = 64;
  auto rdh = reinterpret_cast<o2::header::RAWDataHeaderV4*>(pointer);
  while (!rdh->stop) {
    if (!mFile.read(pointer + rdh->headerSize, rdh->offsetToNext - rdh->headerSize)) {
      std::cout << " --- Cannot read input file: " << strerror(errno) << std::endl;
      mStatus = true;
      return 0;
    }
    payload += (rdh->offsetToNext - rdh->headerSize);
    pointer += rdh->offsetToNext;
    if (!mFile.read(pointer, 64)) {
      std::cout << " --- Cannot read input file: " << strerror(errno) << std::endl;
      mStatus = true;
      return 0;
    }
    payload += 64;
    rdh = reinterpret_cast<o2::header::RAWDataHeaderV4*>(pointer);
  }

  return payload;
}

long FileProxyTask::readFLPv6()
{
  /** read input file **/
  char* pointer = mBuffer;
  if (!mFile.read(pointer, 64)) {
    std::cout << " --- Cannot read input file: " << strerror(errno) << std::endl;
    mStatus = true;
    return 0;
  }
  long payload = 64;
  auto rdh = reinterpret_cast<o2::header::RAWDataHeaderV6*>(pointer);
  while (!rdh->stop) {
    if (!mFile.read(pointer + rdh->headerSize, rdh->offsetToNext - rdh->headerSize)) {
      std::cout << " --- Cannot read input file: " << strerror(errno) << std::endl;
      mStatus = true;
      return 0;
    }
    payload += (rdh->offsetToNext - rdh->headerSize);
    pointer += rdh->offsetToNext;
    if (!mFile.read(pointer, 64)) {
      std::cout << " --- Cannot read input file: " << strerror(errno) << std::endl;
      mStatus = true;
      return 0;
    }
    payload += 64;
    rdh = reinterpret_cast<o2::header::RAWDataHeaderV6*>(pointer);
  }

  return payload;
}

long FileProxyTask::readCONET()
{

  uint32_t word;
  auto current = mFile.tellg();

  /** read size of tof buffer **/
  if (!mFile.read((char*)&word, 4)) {
    std::cout << " --- Cannot read input file: " << strerror(errno) << std::endl;
    mStatus = true;
    return 0;
  }
  auto bytePayload = word * 4;
  std::cout << " --- tofBuffer: " << word << " (" << bytePayload << " bytes) at " << current << std::endl;

  /** read payload **/
  if (!mFile.read(mBuffer, bytePayload)) {
    std::cout << " --- Cannot read input file: " << strerror(errno) << std::endl;
    mStatus = true;
    return 0;
  }
  return bytePayload;
}

void FileProxyTask::run(ProcessingContext& pc)
{

  /** check status **/
  if (mStatus) {
    mFile.close();
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    return;
  }

  /** read data **/
  int payload = 0;
  if (mCONET) {
    payload = readCONET();
  } else {
    payload = readFLP();
  }
  if (payload == 0) {
    mStatus = true;
    return;
  }

  if (mDumpData) {
    std::cout << " --- dump data: " << payload << " bytes" << std::endl;
    uint32_t* word = reinterpret_cast<uint32_t*>(mBuffer);
    for (int i = 0; i < payload / 4; ++i) {
      printf(" 0x%08x \n", *word);
      word++;
    }
    std::cout << " --- end of dump data " << std::endl;
  }

  /** output **/
  auto device = pc.services().get<o2::framework::RawDeviceService>().device();
  auto outputRoutes = pc.services().get<o2::framework::RawDeviceService>().spec().outputs;
  auto fairMQChannel = outputRoutes.at(0).channel;
  auto payloadMessage = device->NewMessage(payload);
  std::memcpy(payloadMessage->GetData(), mBuffer, payload);
  o2::header::DataHeader header("RAWDATA", "TOF", 0);
  header.payloadSize = payload;
  o2::framework::DataProcessingHeader dataProcessingHeader{0};
  o2::header::Stack headerStack{header, dataProcessingHeader};
  auto headerMessage = device->NewMessage(headerStack.size());
  std::memcpy(headerMessage->GetData(), headerStack.data(), headerStack.size());

  /** send **/
  fair::mq::Parts parts;
  parts.AddPart(std::move(headerMessage));
  parts.AddPart(std::move(payloadMessage));
  device->Send(parts, fairMQChannel);

  /** check end of file **/
  if (mFile.eof()) {
    std::cout << " --- End of file " << std::endl;
    mStatus = true;
  }
};

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
}

#include "Framework/runDataProcessing.h" // the main driver

/// This function hooks up the the workflow specifications into the DPL driver.
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    DataProcessorSpec{"file-proxy",
                      Inputs{},
                      Outputs{OutputSpec(ConcreteDataTypeMatcher{"TOF", "RAWDATA"})},
                      AlgorithmSpec(adaptFromTask<FileProxyTask>()),
                      Options{
                        {"atc-file-proxy-input-filename", VariantType::String, "", {"Input file name"}},
                        {"atc-file-proxy-start-from", VariantType::Int, 0, {"Start reading from byte"}},
                        {"atc-file-proxy-dump-data", VariantType::Bool, false, {"Dump data"}},
                        {"atc-file-proxy-rdh-version", VariantType::Int, 4, {"RDH version"}},
                        {"atc-file-proxy-conet-mode", VariantType::Bool, false, {"CONET mode"}}}}};
}
