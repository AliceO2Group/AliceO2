/// @copyright
/// © Copyright 2014 Copyright Holders of the ALICE O2 collaboration.
/// See https://aliceinfo.cern.ch/AliceO2 for details on the Copyright holders.
/// This software is distributed under the terms of the
/// GNU General Public License version 3 (GPL Version 3).
///
/// License text in a separate file.
///
/// In applying this license, CERN does not waive the privileges and immunities
/// granted to it by virtue of its status as an Intergovernmental Organization
/// or submit itself to any jurisdiction.

/// @file ShmSource.h
///
/// @since 2017-01-01
/// @author M. Krzewicki <mkrzewic@cern.ch>


#ifndef SHMSOURCE_H_
#define SHMSOURCE_H_

#include "O2device/O2device.h"
#include "O2device/SharedMemory.h"
#include <chrono>

class ShmSource : public AliceO2::Base::O2device
{
  public:
    ShmSource();
    virtual ~ShmSource();

  protected:
    virtual void InitTask();
    virtual bool ConditionalRun();

  private:
    int mCounter;
    size_t mMaxSize;
    int mDelay;
    bool mDisableShmem;
};

#endif /*SHMSOURCE_H_*/
