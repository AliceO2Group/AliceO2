// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "StreamCompaction.h"

#include <gpucf/common/ClEnv.h>
#include <gpucf/common/log.h>

#include <cmath>

using namespace gpucf;

StreamCompaction::Worker::Worker(
  cl::Program prg,
  cl::Device device,
  DeviceMemory mem,
  CompType type)
  : nativeScanUpStart(prg, "nativeScanUpStart_kernel"), nativeScanUp(prg, "nativeScanUp_kernel"), nativeScanTop(prg, "nativeScanTop_kernel"), nativeScanDown(prg, "nativeScanDown_kernel"), compactDigit(prg, "compactDigit_kernel"), compactCluster(prg, "compactClusterNative_kernel"), mem(mem), type(type)
{
  nativeScanUpStart.getWorkGroupInfo(
    device,
    CL_KERNEL_WORK_GROUP_SIZE,
    &scanUpStartWorkGroupSize);

  nativeScanUp.getWorkGroupInfo(
    device,
    CL_KERNEL_WORK_GROUP_SIZE,
    &scanUpWorkGroupSize);

  ASSERT(scanUpStartWorkGroupSize == scanUpWorkGroupSize);

  log::Debug() << "scanUpWorkGroupSize = " << scanUpWorkGroupSize;

  nativeScanTop.getWorkGroupInfo(
    device,
    CL_KERNEL_WORK_GROUP_SIZE,
    &scanTopWorkGroupSize);

  size_t scanDownWorkGroupSize;
  nativeScanDown.getWorkGroupInfo(
    device,
    CL_KERNEL_WORK_GROUP_SIZE,
    &scanDownWorkGroupSize);

  ASSERT(scanDownWorkGroupSize == scanUpWorkGroupSize);

  size_t compactDigitWorkGroupSize;
  compactDigit.getWorkGroupInfo(
    device,
    CL_KERNEL_WORK_GROUP_SIZE,
    &compactDigitWorkGroupSize);

  ASSERT(compactDigitWorkGroupSize == scanDownWorkGroupSize);

  log::Debug() << "scanTopWorkGroupSize = " << scanTopWorkGroupSize;
}

void StreamCompaction::setup(
  ClEnv& env,
  StreamCompaction::CompType type,
  size_t workernum, size_t digitnum)
{
  this->type = type;

  prg = env.getProgram();

  context = env.getContext();
  device = env.getDevice();

  setDigitNum(digitnum, workernum);
}

void StreamCompaction::setDigitNum(size_t digitnum, size_t workernum)
{
  this->digitNum = digitnum;

  mems = std::stack<DeviceMemory>();

  cl::Kernel scanUp(*prg, "nativeScanUp_kernel");
  size_t scanUpWorkGroupSize;
  scanUp.getWorkGroupInfo(
    device,
    CL_KERNEL_WORK_GROUP_SIZE,
    &scanUpWorkGroupSize);

  cl::Kernel scanTop(*prg, "nativeScanTop_kernel");
  size_t scanTopWorkGroupSize;
  scanTop.getWorkGroupInfo(
    device,
    CL_KERNEL_WORK_GROUP_SIZE,
    &scanTopWorkGroupSize);

  for (size_t i = 0; i < workernum; i++) {
    DeviceMemory mem;

    size_t d = digitNum;
    DBG(d);
    for (; d > scanTopWorkGroupSize;
         d = std::ceil(d / float(scanUpWorkGroupSize))) {
      mem.incrBufs.emplace_back(
        context,
        CL_MEM_READ_WRITE,
        sizeof(cl_int) * d);
      mem.incrBufSizes.push_back(d);
    }

    mem.incrBufs.emplace_back(
      context,
      CL_MEM_READ_WRITE,
      sizeof(cl_int) * d);
    mem.incrBufSizes.push_back(d);

    mems.push(mem);
  }
}

size_t StreamCompaction::Worker::run(
  size_t items,
  cl::CommandQueue queue,
  cl::Buffer digits,
  cl::Buffer digitsOut,
  cl::Buffer predicate,
  bool debug)
{
  if (debug) {
    log::Info() << "StreamCompaction debug on";
  }

  sumsDump.clear();
  scanEvents.clear();

  ASSERT(!mem.incrBufs.empty());
  ASSERT(!mem.incrBufSizes.empty());
  ASSERT(mem.incrBufs.size() == mem.incrBufSizes.size());

  size_t digitnum = items;

  /* queue.enqueueCopyBuffer( */
  /*         predicate, */
  /*         mem.incrBufs.front(), */
  /*         sizeof(cl_int) * digitnum, */
  /*         nullptr, */
  /*         addScanEvent()); */

  std::vector<size_t> digitnums;

  size_t stepnum = this->stepnum(digitnum);
  DBG(stepnum);
  DBG(mem.incrBufs.size());
  ASSERT(stepnum <= mem.incrBufs.size());

  for (size_t i = 1; i < stepnum; i++) {
    digitnums.push_back(digitnum);

    DBG(i);

    if (i == 1) {
      nativeScanUpStart.setArg(0, predicate);
      nativeScanUpStart.setArg(1, mem.incrBufs[i - 1]);
      nativeScanUpStart.setArg(2, mem.incrBufs[i]);

      queue.enqueueNDRangeKernel(
        nativeScanUpStart,
        cl::NDRange(0),
        cl::NDRange(digitnum),
        cl::NDRange(scanUpStartWorkGroupSize),
        nullptr,
        addScanEvent());
    } else {
      ASSERT(mem.incrBufSizes[i - 1] >= digitnum);

      nativeScanUp.setArg(0, mem.incrBufs[i - 1]);
      nativeScanUp.setArg(1, mem.incrBufs[i]);
      queue.enqueueNDRangeKernel(
        nativeScanUp,
        cl::NDRange(0),
        cl::NDRange(digitnum),
        cl::NDRange(scanUpWorkGroupSize),
        nullptr,
        addScanEvent());
    }

    if (debug) {
      dumpBuffer(queue, mem.incrBufs[i - 1], mem.incrBufSizes[i - 1]);
    }

    digitnum /= scanUpWorkGroupSize;
  }

  if (debug) {
    dumpBuffer(queue, mem.incrBufs[stepnum - 1], mem.incrBufSizes[stepnum - 1]);
  }

  ASSERT(digitnum <= scanTopWorkGroupSize);

  nativeScanTop.setArg(0, mem.incrBufs[stepnum - 1]);
  queue.enqueueNDRangeKernel(
    nativeScanTop,
    cl::NDRange(0),
    cl::NDRange(digitnum),
    cl::NDRange(scanTopWorkGroupSize),
    nullptr,
    addScanEvent());

  if (debug) {
    dumpBuffer(queue, mem.incrBufs[stepnum - 1], mem.incrBufSizes[stepnum - 1]);
  }

  ASSERT(digitnums.size() == stepnum - 1);
  for (size_t i = stepnum - 1; i > 1; i--) {
    digitnum = digitnums[i - 1];

    ASSERT(digitnum > scanUpWorkGroupSize);

    nativeScanDown.setArg(0, mem.incrBufs[i - 1]);
    nativeScanDown.setArg(1, mem.incrBufs[i]);
    nativeScanDown.setArg(1, (unsigned int)scanUpWorkGroupSize);
    queue.enqueueNDRangeKernel(
      nativeScanDown,
      cl::NDRange(0),
      cl::NDRange(digitnum - scanUpWorkGroupSize),
      cl::NDRange(scanUpWorkGroupSize),
      nullptr,
      addScanEvent());

    if (debug) {
      dumpBuffer(queue, mem.incrBufs[i - 1], mem.incrBufSizes[i - 1]);
    }
  }

  digitnum = digitnums.front();

  cl::Kernel compact;

  switch (type) {
    case CompType::Digit:
      log::Debug() << "Compact digits.";
      compact = compactDigit;
      break;
    case CompType::Cluster:
      compact = compactCluster;
      break;
  }

  log::Debug() << "Run compaction kernel.";

  compact.setArg(0, digits);
  compact.setArg(1, digitsOut);
  compact.setArg(2, predicate);
  compact.setArg(3, mem.incrBufs[0]);
  compact.setArg(4, mem.incrBufs[1]);
  queue.enqueueNDRangeKernel(
    compact,
    cl::NDRange(0),
    cl::NDRange(digitnum),
    cl::NDRange(scanUpWorkGroupSize),
    nullptr,
    compactArrEv.get());

  log::Debug() << "Get num of peaks.";

  cl_int newDigitNum;

  // Read the last element from index buffer to get the new number of digits
  queue.enqueueReadBuffer(
    mem.incrBufs.front(),
    CL_TRUE,
    (digitnum - 1) * sizeof(cl_int),
    sizeof(cl_int),
    &newDigitNum,
    nullptr,
    readNewDigitNum.get());

  DBG(newDigitNum);

  if (debug) {
    printDump();
  }

  ASSERT(size_t(newDigitNum) <= digitnum);

  return newDigitNum;
}

size_t StreamCompaction::Worker::stepnum(size_t items) const
{
  size_t c = 0;

  while (items > 0) {
    items /= scanUpWorkGroupSize;
    c++;
  }

  return c;
}

std::vector<std::vector<int>> StreamCompaction::Worker::getNewIdxDump() const
{
  return sumsDump;
}

Step StreamCompaction::Worker::asStep(const std::string& name) const
{
  Timestamp start = scanEvents.front().start();
  /* Timestamp end   = readNewDigitNum.end(); */
  Timestamp end = compactArrEv.end();
  return {name, start, start, start, end};
}

void StreamCompaction::Worker::printDump() const
{
  log::Error() << "Result dump: ";
  for (std::vector<int> interim : sumsDump) {
    constexpr size_t maxPrintOut = 100;
    size_t c = maxPrintOut;
    for (int idx : interim) {
      log::Error() << idx;
      c--;
      if (c == 0) {
        log::Error() << "...";
        break;
      }
    }
    log::Error();
  }
}

void StreamCompaction::Worker::dumpBuffer(
  cl::CommandQueue queue,
  cl::Buffer buf,
  size_t size)
{
  sumsDump.emplace_back(size);
  queue.enqueueReadBuffer(
    buf,
    CL_FALSE,
    0,
    sizeof(cl_int) * size,
    sumsDump.back().data());
}

cl::Event* StreamCompaction::Worker::addScanEvent()
{
  scanEvents.emplace_back();
  return scanEvents.back().get();
}

StreamCompaction::Worker StreamCompaction::worker()
{
  ASSERT(!mems.empty());

  DeviceMemory mem = mems.top();
  mems.pop();

  return Worker(*prg, device, mem, type);
}

// vim: set ts=4 sw=4 sts=4 expandtab:
