//-*- Mode: C++ -*-
// $Id$
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef ALIHLTTPCCATRACKEROUTPUTCONVERTER_H
#define ALIHLTTPCCATRACKEROUTPUTCONVERTER_H

/// @file   AliHLTTPCCATrackerOutputConverter.h
/// @author Sergey Gorbunov
/// @date
/// @brief  Converter of CA tracker output
///

#include "AliHLTProcessor.h"
#include "AliHLTComponentBenchmark.h"


/**
 * @class AliHLTTPCCATrackerOutputConverter
 * Converter of the AliHLTTPCCATracker output
 * 
 */
class AliHLTTPCCATrackerOutputConverter : public AliHLTProcessor
{
  public:
    /**
     * Constructs a AliHLTTPCCATrackerOutputConverter.
     */
    AliHLTTPCCATrackerOutputConverter();

    /**
     * Destructs the AliHLTTPCCATrackerOutputConverter
     */
    virtual ~AliHLTTPCCATrackerOutputConverter() {};

    // Public functions to implement AliHLTComponent's interface.
    // These functions are required for the registration process

    /**
     * @copydoc AliHLTComponent::GetComponentID
     */
    const char *GetComponentID();

    /**
     * @copydoc AliHLTComponent::GetInputDataTypes
     */
    void GetInputDataTypes( AliHLTComponentDataTypeList &list );

    /**
     * @copydoc AliHLTComponent::GetOutputDataType
     */
    AliHLTComponentDataType GetOutputDataType();

    /**
     * @copydoc AliHLTComponent::GetOutputDataSize
     */
    virtual void GetOutputDataSize( unsigned long& constBase, double& inputMultiplier );

    /**
     * @copydoc AliHLTComponent::Spawn
     */
    AliHLTComponent *Spawn();

  protected:

    // Protected functions to implement AliHLTComponent's interface.
    // These functions provide initialization as well as the actual processing
    // capabilities of the component.

    /**
     * @copydoc AliHLTComponent::DoInit
     */
    int DoInit( int argc, const char **argv );

    /**
     * @copydoc AliHLTComponent::DoDeinit
     */
    int DoDeinit();

    /** reconfigure **/
    int Reconfigure( const char* cdbEntry, const char* chainId );

    /**
     * @copydoc @ref AliHLTProcessor::DoEvent
     */
    int DoEvent( const AliHLTComponentEventData &evtData, const AliHLTComponentBlockData *blocks,
                 AliHLTComponentTriggerData &trigData, AliHLTUInt8_t *outputPtr,
                 AliHLTUInt32_t &size, AliHLTComponentBlockDataList &outputBlocks );

    using AliHLTProcessor::DoEvent;

  private:

    static AliHLTTPCCATrackerOutputConverter fgAliHLTTPCCATrackerOutputConverter;

    // disable copy
    AliHLTTPCCATrackerOutputConverter( const AliHLTTPCCATrackerOutputConverter & );
    AliHLTTPCCATrackerOutputConverter &operator=( const AliHLTTPCCATrackerOutputConverter & );

    /** set configuration parameters **/
    void SetDefaultConfiguration();
    int ReadConfigurationString(  const char* arguments );
    int ReadCDBEntry( const char* cdbEntry, const char* chainId );
    int Configure( const char* cdbEntry, const char* chainId, const char *commandLine );

    AliHLTComponentBenchmark fBenchmark;// benchmark

    ClassDef( AliHLTTPCCATrackerOutputConverter, 0 )
};

#endif //ALIHLTTPCCAGLOBALMERGERCOMPONENT_H
