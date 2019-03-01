//-*- Mode: C++ -*-
// $Id$
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

///  @file   GPUTRDTrackerComponent.h
///  @author Marten Ole Schmidt <ole.schmidt@cern.ch>
///  @date   May 2016
///  @brief  A TRD tracker processing component for the GPU

#ifndef GPUTRDTRACKERCOMPONENT_H
#define GPUTRDTRACKERCOMPONENT_H

#ifndef GPUCA_ALIROOT_LIB
#define GPUCA_ALIROOT_LIB
#endif

#include "AliHLTProcessor.h"
#include "AliHLTComponentBenchmark.h"
#include "AliHLTDataTypes.h"

class TH1F;
class TList;
class GPUTRDTracker;
class GPUTRDGeometry;


class GPUTRDTrackerComponent : public AliHLTProcessor {
public:

  /*
   * ---------------------------------------------------------------------------------
   *                            Constructor / Destructor
   * ---------------------------------------------------------------------------------
   */

  /** constructor */
  GPUTRDTrackerComponent();

  /** dummy copy constructor, defined according to effective C++ style */
  GPUTRDTrackerComponent( const GPUTRDTrackerComponent& );

  /** dummy assignment op, but defined according to effective C++ style */
  GPUTRDTrackerComponent& operator=( const GPUTRDTrackerComponent& );

  /** destructor */
  virtual ~GPUTRDTrackerComponent();

  /*
   * ---------------------------------------------------------------------------------
   * Public functions to implement AliHLTComponent's interface.
   * These functions are required for the registration process
   * ---------------------------------------------------------------------------------
   */

  /** interface function, see @ref AliHLTComponent for description */
  const char* GetComponentID();

  /** interface function, see @ref AliHLTComponent for description */
  void GetInputDataTypes( vector<AliHLTComponentDataType>& list);

  /** interface function, see @ref AliHLTComponent for description */
  AliHLTComponentDataType GetOutputDataType();

  /** @see component interface @ref AliHLTComponent::GetOutputDataType */
  int  GetOutputDataTypes(AliHLTComponentDataTypeList& tgtList);

  /** interface function, see @ref AliHLTComponent for description */
  void GetOutputDataSize( unsigned long& constBase, double& inputMultiplier );

  /** interface function, see @ref AliHLTComponent for description */
  AliHLTComponent* Spawn();

  int ReadConfigurationString( const char* arguments );

 protected:

  /*
   * ---------------------------------------------------------------------------------
   * Protected functions to implement AliHLTComponent's interface.
   * These functions provide initialization as well as the actual processing
   * capabilities of the component.
   * ---------------------------------------------------------------------------------
   */

  // AliHLTComponent interface functions

  /** interface function, see @ref AliHLTComponent for description */
  int DoInit( int argc, const char** argv );

  /** interface function, see @ref AliHLTComponent for description */
  int DoDeinit();

  /** interface function, see @ref AliHLTComponent for description */
  int DoEvent( const AliHLTComponentEventData& evtData, const AliHLTComponentBlockData* blocks,
               AliHLTComponentTriggerData& trigData, AliHLTUInt8_t* outputPtr,
               AliHLTUInt32_t& size, vector<AliHLTComponentBlockData>& outputBlocks );


  /** interface function, see @ref AliHLTComponent for description */
  int Reconfigure(const char* cdbEntry, const char* chainId);


  ///////////////////////////////////////////////////////////////////////////////////

private:

  /*
   * ---------------------------------------------------------------------------------
   * Private functions to implement AliHLTComponent's interface.
   * These functions provide initialization as well as the actual processing
   * capabilities of the component.
   * ---------------------------------------------------------------------------------
   */

  /*
   * ---------------------------------------------------------------------------------
   *                              Helper
   * ---------------------------------------------------------------------------------
   */

  /*
   * ---------------------------------------------------------------------------------
   *                             Members - private
   * ---------------------------------------------------------------------------------
   */
  GPUTRDTracker *fTracker; // the tracker itself
  GPUTRDGeometry *fGeo;       // TRD geometry needed by the tracker

  TList* fTrackList;
  bool fDebugTrackOutput; // output GPUTRDTracks instead AliHLTExternalTrackParam
  bool fVerboseDebugOutput; // more verbose information is printed
  bool fRequireITStrack;  // only TPC tracks with ITS match are used as seeds for tracking
  AliHLTComponentBenchmark fBenchmark; // benchmark

  ClassDef(GPUTRDTrackerComponent, 0)
};
#endif
