//-*- Mode: C++ -*-
// $Id$
#ifndef ALIHLTTRDTRACKLETREADERCOMPONENT_H
#define ALIHLTTRDTRACKLETREADERCOMPONENT_H

//* This file is property of and copyright by the ALICE HLT/TRD Project    *
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               */

/// @file   AliHLTTRDTrackletReaderComponent.h
/// @author Felix Rettig, Stefan Kirsch
/// @date   2012-08-16
/// @brief  A FEP-level pre-processing component for TRD tracking/trigger data
/// @ingroup alihlt_trd_components

#include "AliHLTProcessor.h"

class AliRawReaderMemory;
class TTree;
class AliTRDrawStream;
class AliTRDonlineTrackingDataContainer;
class TClonesArray;

/**
 * @class AliHLTTRDTrackletReaderComponent
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
class AliHLTTRDTrackletReaderComponent : public AliHLTProcessor {
public:
  AliHLTTRDTrackletReaderComponent();
  virtual ~AliHLTTRDTrackletReaderComponent();

  // AliHLTComponent interface functions
  const char* GetComponentID();
  void GetInputDataTypes( vector<AliHLTComponentDataType>& list);
  AliHLTComponentDataType GetOutputDataType();
  int GetOutputDataTypes(AliHLTComponentDataTypeList& tgtList);
  void GetOutputDataSize( unsigned long& constBase, double& inputMultiplier );
  void GetOCDBObjectDescription( TMap* const targetMap);

  // Spawn function, return new class instance
  AliHLTComponent* Spawn();

 protected:
  // AliHLTComponent interface functions
  int DoInit( int argc, const char** argv );
  int DoDeinit();
  int DoEvent( const AliHLTComponentEventData& evtData, AliHLTComponentTriggerData& trigData);
  int ScanConfigurationArgument(int argc, const char** argv);
  int Reconfigure(const char* cdbEntry, const char* chainId);
  int ReadPreprocessorValues(const char* modules);

  using AliHLTProcessor::DoEvent;

private:
  /** copy constructor prohibited */
  AliHLTTRDTrackletReaderComponent(const AliHLTTRDTrackletReaderComponent&);
  /** assignment operator prohibited */
  AliHLTTRDTrackletReaderComponent& operator=(const AliHLTTRDTrackletReaderComponent&);

  void DbgLog(const char* prefix, ...);

  // general
  static const AliHLTEventID_t fgkInvalidEventId = 18446744073709551615llu;

  UShort_t fDebugLevel;                              //! set debug checks/output level, 0: debug off
  AliHLTEventID_t fEventId;                          //! event ID

  // trd specific data
  TClonesArray* fTrackletArray;                      //! internal tracklet array

  // rawreader instance
  AliRawReaderMemory* fRawReaderMem;                 //! TRD raw reader memory instance
  AliTRDrawStream*    fRawReaderTrd;                 //! TRD raw stream instance


  ClassDef(AliHLTTRDTrackletReaderComponent, 0)
};

#endif
