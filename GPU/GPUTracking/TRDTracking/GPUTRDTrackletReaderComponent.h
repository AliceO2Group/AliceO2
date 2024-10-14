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

/// \file GPUTRDTrackletReaderComponent.h
/// \brief A pre-processing component for TRD tracking/trigger data on FEP-level

/// \author Felix Rettig, Stefan Kirsch, Ole Schmidt

#ifndef GPUTRDTRACKLETREADERCOMPONENT_H
#define GPUTRDTRACKLETREADERCOMPONENT_H

#ifndef GPUCA_ALIROOT_LIB
#define GPUCA_ALIROOT_LIB
#endif

#include "AliHLTProcessor.h"

class AliRawReaderMemory;
class TTree;
class AliTRDrawStream;
class AliTRDonlineTrackingDataContainer;
class TClonesArray;

/**
 * @class GPUTRDTrackletReaderComponent
 * Component fetches raw data input objects in DDL format and extracts tracklets.
 *  It also instantiates a RawReader in order to be used with some reconstruction.
 *
 * More information and examples can be found here (relative to $ALICE_ROOT):
 *
 * -- HLT/BASE/AliHLTComponent.h/.cxx,  HLT/BASE/AliHLTProcessor.h/.cxx
 *    Interface definition and description
 * -- HLT/SampleLib: example implementations of components
 *
 *
 * <h2>General properties:</h2>
 *
 * Component ID: \b TRDReaderComponent <br>
 * Library: \b libAliHLTTRD.so     <br>
 * Input Data Types: @ref kAliHLTDataTypeDDLRaw|kAliHLTDataOriginTRD <br>
 * Output Data Types: @ref kAliHLTTrackDataTypeID|kAliHLTDataOriginTRD <br>
 *
 * <h2>Mandatory arguments:</h2>
 * none
 *
 * <h2>Optional arguments:</h2>
 * none
 *
 * <h2>Configuration:</h2>
 * none
 *
 * <h2>Default CDB entries:</h2>
 * none
 *
 * <h2>Performance:</h2>
 * minmal
 *
 * <h2>Memory consumption:</h2>
 * don't know yet
 *
 * <h2>Output size:</h2>
 * not very much
 *
 * @ingroup The component has no output data.
 */
class GPUTRDTrackletReaderComponent : public AliHLTProcessor
{
 public:
  GPUTRDTrackletReaderComponent();
  virtual ~GPUTRDTrackletReaderComponent();

  // AliHLTComponent interface functions
  const char* GetComponentID();
  void GetInputDataTypes(vector<AliHLTComponentDataType>& list);
  AliHLTComponentDataType GetOutputDataType();
  int32_t GetOutputDataTypes(AliHLTComponentDataTypeList& tgtList);
  void GetOutputDataSize(uint64_t& constBase, double& inputMultiplier);
  void GetOCDBObjectDescription(TMap* const targetMap);

  // Spawn function, return new class instance
  AliHLTComponent* Spawn();

 protected:
  // AliHLTComponent interface functions
  int32_t DoInit(int argc, const char** argv);
  int32_t DoDeinit();
  int32_t DoEvent(const AliHLTComponentEventData& evtData, AliHLTComponentTriggerData& trigData);
  int32_t ScanConfigurationArgument(int argc, const char** argv);
  int32_t Reconfigure(const char* cdbEntry, const char* chainId);
  int32_t ReadPreprocessorValues(const char* modules);

  using AliHLTProcessor::DoEvent;

 private:
  /** copy constructor prohibited */
  GPUTRDTrackletReaderComponent(const GPUTRDTrackletReaderComponent&);
  /** assignment operator prohibited */
  GPUTRDTrackletReaderComponent& operator=(const GPUTRDTrackletReaderComponent&);

  void DbgLog(const char* prefix, ...);

  // general
  static const AliHLTEventID_t fgkInvalidEventId = 18446744073709551615llu;

  UShort_t fDebugLevel;     //! set debug checks/output level, 0: debug off
  AliHLTEventID_t fEventId; //! event ID

  // trd specific data
  TClonesArray* fTrackletArray; //! internal tracklet array

  // rawreader instance
  AliRawReaderMemory* fRawReaderMem; //! TRD raw reader memory instance
  AliTRDrawStream* fRawReaderTrd;    //! TRD raw stream instance

  ClassDef(GPUTRDTrackletReaderComponent, 0);
};

#endif // GPUTRDTRACKLETREADERCOMPONENT_H
