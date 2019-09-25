// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or        *
//* (at your option) any later version.                                      *
//*                                                                          *
//* Primary Authors: Matthias Richter <richterm@scieq.net>                   *
//*                                                                          *
//* The authors make no claims about the suitability of this software for    *
//* any purpose. It is provided "as is" without express or implied warranty. *
//*                                                                          *
//* See original copyright notice below                                      *
//****************************************************************************

//  @file   HOMERFactory.cxx
//  @author Matthias Richter
//  @since  2014-05-07
//  @brief  Original AliHLTHOMERLibManager.h of AliRoot adapted to the
//          ALFA project

/**************************************************************************
 * This file is property of and copyright by the ALICE HLT Project        * 
 * ALICE Experiment at CERN, All rights reserved.                         *
 *                                                                        *
 * Primary Authors: Matthias Richter <Matthias.Richter@ift.uib.no>        *
 *                  for The ALICE HLT Project.                            *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

#include <cerrno>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <dlfcn.h>
#include "aliceHLTwrapper/HOMERFactory.h"
#include "aliceHLTwrapper/AliHLTHOMERReader.h"
#include "aliceHLTwrapper/AliHLTHOMERWriter.h"

using namespace o2::alice_hlt;

using std::cerr;
using std::cout;
using std::endl;

// global flag of the library status
int HOMERFactory::sLibraryStatus = 0;
// This list must be NULL terminated, since we use it as a marker to identify
// the end of the list.
const char* HOMERFactory::sLibraries[] = {"libAliHLTHOMER.so", "libHOMER.so", nullptr};
// The size of the list of reference counts must be one less than sLibraries.
int HOMERFactory::sLibRefCount[] = {0, 0};

HOMERFactory::HOMERFactory()
  : mFctCreateReaderFromTCPPort(nullptr), mFctCreateReaderFromTCPPorts(nullptr), mFctCreateReaderFromBuffer(nullptr), mFctDeleteReader(nullptr), mFctCreateWriter(nullptr), mFctDeleteWriter(nullptr), mLoadedLib(nullptr), mHandle(nullptr)
{
  // constructor
  //
  // Interface to the HLT Online Monitoring Including Root (HOMER) library.
  // It allows to decouple the HLT base library from this additional library
  // while providing the basic functionality to the component libraries
}

// destructor
//
// the library load strategy has been changed in March 2013 in order to
// stabilize the runtime memory layout of AliRoot in an attemp to get control
// over memory corruptions
HOMERFactory::~HOMERFactory() = default;

AliHLTHOMERReader* HOMERFactory::OpenReader(const char* hostname, unsigned short port)
{
  // Open Reader instance for host
  if (sLibraryStatus < 0)
    return nullptr;

  sLibraryStatus = LoadHOMERLibrary();
  if (sLibraryStatus <= 0) {
    return nullptr;
  }

  AliHLTHOMERReader* pReader = nullptr;
  if (mFctCreateReaderFromTCPPort != nullptr && (pReader = (((AliHLTHOMERReaderCreateFromTCPPort_t)mFctCreateReaderFromTCPPort)(hostname, port))) == nullptr) {
    cout << "can not create instance of HOMER reader from ports" << endl;
  }

  return pReader;
}

AliHLTHOMERReader* HOMERFactory::OpenReader(unsigned int tcpCnt, const char** hostnames, unsigned short* ports)
{
  // Open Reader instance for a list of hosts
  if (sLibraryStatus < 0)
    return nullptr;

  sLibraryStatus = LoadHOMERLibrary();
  if (sLibraryStatus <= 0) {
    return nullptr;
  }

  AliHLTHOMERReader* pReader = nullptr;
  if (mFctCreateReaderFromTCPPorts != nullptr && (pReader = (((AliHLTHOMERReaderCreateFromTCPPorts_t)mFctCreateReaderFromTCPPorts)(tcpCnt, hostnames, ports))) == nullptr) {
    //HLTError("can not create instance of HOMER reader (function %p)", mFctCreateReaderFromTCPPorts);
    cout << "can not create instance of HOMER reader from port" << endl;
  }

  return pReader;
}

AliHLTHOMERReader* HOMERFactory::OpenReaderBuffer(const AliHLTUInt8_t* pBuffer, int size)
{
  // Open Reader instance for a data buffer
  if (sLibraryStatus < 0)
    return nullptr;

  sLibraryStatus = LoadHOMERLibrary();
  if (sLibraryStatus <= 0) {
    return nullptr;
  }

  AliHLTHOMERReader* pReader = nullptr;
  if (mFctCreateReaderFromBuffer != nullptr && (pReader = (((AliHLTHOMERReaderCreateFromBuffer_t)mFctCreateReaderFromBuffer)(pBuffer, size))) == nullptr) {
    //HLTError("can not create instance of HOMER reader (function %p)", mFctCreateReaderFromBuffer);
  }

  return pReader;
}

int HOMERFactory::DeleteReader(AliHLTHOMERReader* pReader)
{
  // delete a reader

  // the actual deletion function is inside the HOMER library
  if (sLibraryStatus < 0)
    return sLibraryStatus;

  sLibraryStatus = LoadHOMERLibrary();
  if (sLibraryStatus <= 0) {
    return sLibraryStatus;
  }

  if (mFctDeleteReader != nullptr) {
    ((AliHLTHOMERReaderDelete_t)mFctDeleteReader)(pReader);
  }

  return 0;
}

AliHLTHOMERWriter* HOMERFactory::OpenWriter()
{
  // open a Writer instance
  if (sLibraryStatus < 0)
    return nullptr;

  sLibraryStatus = LoadHOMERLibrary();
  if (sLibraryStatus <= 0) {
    return nullptr;
  }

  AliHLTHOMERWriter* pWriter = nullptr;
  if (mFctCreateWriter != nullptr && (pWriter = (((AliHLTHOMERWriterCreate_t)mFctCreateWriter)())) == nullptr) {
    //     HLTError("can not create instance of HOMER writer (function %p)", mFctCreateWriter);
  }

  return pWriter;
}

int HOMERFactory::DeleteWriter(AliHLTHOMERWriter* pWriter)
{
  // see header file for class documentation
  if (sLibraryStatus < 0)
    return sLibraryStatus;

  sLibraryStatus = LoadHOMERLibrary();
  if (sLibraryStatus <= 0) {
    return sLibraryStatus;
  }

  if (mFctDeleteWriter != nullptr) {
    ((AliHLTHOMERWriterDelete_t)mFctDeleteWriter)(pWriter);
  }

  return 0;
}

int HOMERFactory::LoadHOMERLibrary()
{
  // delete a writer

  // the actual deletion function is inside the HOMER library
  int iResult = -EBADF;
  const char** library = &sLibraries[0];
  int* refcount = &sLibRefCount[0];
  do {
    mHandle = dlopen(*library, RTLD_NOW);
    if (mHandle) {
      ++(*refcount);
      mLoadedLib = *library;
      iResult = 1;
      break;
    }
    ++library;
    ++refcount;
  } while ((*library) != nullptr);

  if (iResult > 0 && *library != nullptr) {
    // print compile info
    using CompileInfo = void (*)(char*&, char*&);

    mFctCreateReaderFromTCPPort = (void (*)())dlsym(mHandle, ALIHLTHOMERREADER_CREATE_FROM_TCPPORT);
    mFctCreateReaderFromTCPPorts = (void (*)())dlsym(mHandle, ALIHLTHOMERREADER_CREATE_FROM_TCPPORTS);
    mFctCreateReaderFromBuffer = (void (*)())dlsym(mHandle, ALIHLTHOMERREADER_CREATE_FROM_BUFFER);
    mFctDeleteReader = (void (*)())dlsym(mHandle, ALIHLTHOMERREADER_DELETE);
    mFctCreateWriter = (void (*)())dlsym(mHandle, ALIHLTHOMERWRITER_CREATE);
    mFctDeleteWriter = (void (*)())dlsym(mHandle, ALIHLTHOMERWRITER_DELETE);
    if (mFctCreateReaderFromTCPPort == nullptr ||
        mFctCreateReaderFromTCPPorts == nullptr ||
        mFctCreateReaderFromBuffer == nullptr ||
        mFctDeleteReader == nullptr ||
        mFctCreateWriter == nullptr ||
        mFctDeleteWriter == nullptr) {
      iResult = -ENOSYS;
    } else {
    }
  }
  if (iResult < 0 || *library == nullptr) {
    mFctCreateReaderFromTCPPort = nullptr;
    mFctCreateReaderFromTCPPorts = nullptr;
    mFctCreateReaderFromBuffer = nullptr;
    mFctDeleteReader = nullptr;
    mFctCreateWriter = nullptr;
    mFctDeleteWriter = nullptr;
  }

  return iResult;
}

int HOMERFactory::UnloadHOMERLibrary()
{
  // unload HOMER library
  int iResult = 0;

  if (mLoadedLib != nullptr) {
    // Find the corresponding reference count.
    const char** library = &sLibraries[0];
    int* refcount = &sLibRefCount[0];
    while (*library != nullptr) {
      if (strcmp(*library, mLoadedLib) == 0)
        break;
      ++library;
      ++refcount;
    }

    // Decrease the reference count and remove the library if it is zero.
    if (*refcount >= 0)
      --(*refcount);
    if (*refcount == 0) { /* Matthias 2014-05-08 that part of the code has been disabled for the ROOT-independent
         version.
         TODO: dlopen is maintaining an internal reference count, which also makes the
         counters here obsolete

      // Check that the library we are trying to unload is actually the last library
      // in the gSystem->GetLibraries() list. If not then we must abort the removal.
      // This is because of a ROOT bug/feature/limitation. If we try unload the library
      // then ROOT will also wipe all libraries in the gSystem->GetLibraries() list
      // following the library we want to unload.
      TString libstring = gSystem->GetLibraries();
      TString token, lastlib;
      Ssiz_t from = 0;
      Int_t numOfLibs = 0, posOfLib = -1;
      while (libstring.Tokenize(token, from, " "))
      {
        ++numOfLibs;
        lastlib = token;
        if (token.Contains(mLoadedLib)) posOfLib = numOfLibs;
      }
      if (numOfLibs == posOfLib)
      {
        gSystem->Unload(mLoadedLib);

        // Check that the library is gone, since Unload() does not return a status code.
        libstring = gSystem->GetLibraries();
        if (libstring.Contains(mLoadedLib)) iResult = -EBADF;
      }
      else
      {
        AliHLTLogging log;
        log.LoggingVarargs(kHLTLogWarning, Class_Name(), FUNCTIONNAME(), __FILE__, __LINE__,
          Form("ROOT limitation! Cannot properly cleanup and unload the shared"
            " library '%s' since another library '%s' was loaded afterwards. Trying to"
            " unload this library will remove the others and lead to serious memory faults.",
            mLoadedLib, lastlib.Data()
        ));
      }
      */
      dlclose(mHandle);
    }
  }

  // Clear the function pointers.
  mFctCreateReaderFromTCPPort = nullptr;
  mFctCreateReaderFromTCPPorts = nullptr;
  mFctCreateReaderFromBuffer = nullptr;
  mFctDeleteReader = nullptr;
  mFctCreateWriter = nullptr;
  mFctDeleteWriter = nullptr;

  return iResult;
}
