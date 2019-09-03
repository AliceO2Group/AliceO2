// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef HOMERFACTORY_H
#define HOMERFACTORY_H
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

//  @file   HOMERFactory.h
//  @author Matthias Richter
//  @since  2014-05-07
//  @brief  Original AliHLTHOMERLibManager.h of AliRoot adapted to the
//          ALFA project

/* This file is property of and copyright by the ALICE HLT Project        * 
 * ALICE Experiment at CERN, All rights reserved.                         *
 * See cxx source for full Copyright notice                               */

#include "AliHLTDataTypes.h"
class AliHLTHOMERReader;
class AliHLTHOMERWriter;

namespace o2
{
namespace alice_hlt
{

/**
 * @class HOMERFactory
 * Dynamic manager of HOMER library.
 * The class allows to generate objects of HOMER readers and writers
 * dynamically and loads also the HOMER lib. In order to write HOMER library
 * independent code it is important to use the AliHLTMonitoringWriter/
 * AliHLTMonitoringReader classes when ever class methods are used. Those
 * classes just define a virtual interface. <br>
 *
 * Instead of creating a reader or writer by \em new and deleting it with
 * \em delete, one has to use the Open and Delete methods of this class.
 *
 * <pre>
 * HOMERFactory manager;
 *
 * // open a HOMER reader listening at port 23000 of the localhost
 * AliHLTMonitoringReader* pReader=manager.OpenReader(localhost, 23000);
 *
 * // read next event, timeout 5s
 * while (pReader && pReader->ReadNextEvent(5000000)==0) {
 *   unsigned long count=pReader->GetBlockCnt();
 *   gSystem->Sleep(5);
 *   ...
 * }
 *
 * // delete reader
 * manager.DeleteReader(pReader);
 * </pre>
 *
 * The manager does not provide methods to create a HOMER reader on
 * basis of shared memory. This is most likely a depricated functionality,
 * although kept for the sake of completeness. However, at some point it
 * might become useful. Please notify the developers if you need that
 * functionality.
 *
 * @ingroup alihlt_homer
 */
class HOMERFactory
{
 public:
  /** standard constructor */
  HOMERFactory();
  /** destructor */
  virtual ~HOMERFactory();

  /**
   * Open a homer reader working on a TCP port.
   */
  AliHLTHOMERReader* OpenReader(const char* hostname, unsigned short port);

  /**
   * Open a homer reader working on multiple TCP ports.
   */
  AliHLTHOMERReader* OpenReader(unsigned int tcpCnt, const char** hostnames, unsigned short* ports);

  /**
   * Open a HOMER reader for reading from a System V shared memory segment.
  AliHLTHOMERReader* OpenReader(key_t shmKey, int shmSize );
   */

  /**
   * Open a HOMER reader for reading from multiple System V shared memory segments
  AliHLTHOMERReader* OpenReader(unsigned int shmCnt, key_t* shmKey, int* shmSize );
   */

  /**
   * Open a HOMER reader for reading from multiple TCP ports and multiple System V shared memory segments
  AliHLTHOMERReader* OpenReader(unsigned int tcpCnt, const char** hostnames, unsigned short* ports, 
                                    unsigned int shmCnt, key_t* shmKey, int* shmSize );
   */

  /**
   * Open a HOMER reader.
   * Load HOMER library dynamically and create object working on the provided
   * buffer.
   */
  AliHLTHOMERReader* OpenReaderBuffer(const AliHLTUInt8_t* pBuffer, int size);

  /**
   * Delete a HOMER reader.
   * Clean-up of the object is done inside the HOMER library.
   */
  int DeleteReader(AliHLTHOMERReader* pReader);

  /**
   * Open a HOMER writer.
   * Load HOMER library dynamically and create object working on the provided
   * buffer.
   */
  AliHLTHOMERWriter* OpenWriter();

  /**
   * Delete a HOMER writer.
   * Clean-up of the object is done inside the HOMER library.
   */
  int DeleteWriter(AliHLTHOMERWriter* pWriter);

 protected:
 private:
  /** copy constructor prohibited */
  HOMERFactory(const HOMERFactory&);
  /** assignment operator prohibited */
  HOMERFactory& operator=(const HOMERFactory&);

  /**
   * Load the HOMER library.
   */
  int LoadHOMERLibrary();

  /**
   * Unloads the HOMER library.
   */
  int UnloadHOMERLibrary();

  /** status of the loading of the HOMER library */
  static int sLibraryStatus; //!transient

  /** entry in the HOMER library */
  void (*mFctCreateReaderFromTCPPort)(); //!transient

  /** entry in the HOMER library */
  void (*mFctCreateReaderFromTCPPorts)(); //!transient

  /** entry in the HOMER library */
  void (*mFctCreateReaderFromBuffer)(); //!transient

  /** entry in the HOMER library */
  void (*mFctDeleteReader)(); //!transient

  /** entry in the HOMER library */
  void (*mFctCreateWriter)(); //!transient

  /** entry in the HOMER library */
  void (*mFctDeleteWriter)(); //!transient

  /** Indicates the library that was actually (and if) loaded in LoadHOMERLibrary(). */
  const char* mLoadedLib; //!transient

  /** library handle returned by dlopen */
  void* mHandle;

  static const char* sLibraries[]; /// List of libraries to try and load.
  static int sLibRefCount[];       /// The library reference count to control when to unload the library.
};
} // namespace alice_hlt
} // namespace o2
#endif
