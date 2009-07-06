// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef ALIHLTTPCCAGLOBALMERGERCOMPONENT_H
#define ALIHLTTPCCAGLOBALMERGERCOMPONENT_H

/** @file   AliHLTTPCCAGlobalMergerComponent.h
    @author Matthias Kretz
    @date
    @brief  HLT TPC CA global merger component.
*/

#include "AliHLTProcessor.h"

class AliHLTTPCCAMerger;
class AliHLTTPCVertex;

/**
 * @class AliHLTTPCCAGlobalMergerComponent
 * The TPC global merger component
 *
 * Interface to the global merger of the CA tracker for HLT.
 */
class AliHLTTPCCAGlobalMergerComponent : public AliHLTProcessor
{
  public:
    /**
     * Constructs a AliHLTTPCCAGlobalMergerComponent.
     */
    AliHLTTPCCAGlobalMergerComponent();

    /**
     * Destructs the AliHLTTPCCAGlobalMergerComponent
     */
    virtual ~AliHLTTPCCAGlobalMergerComponent() {};

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

    static AliHLTTPCCAGlobalMergerComponent fgAliHLTTPCCAGlobalMergerComponent;

    // disable copy
    AliHLTTPCCAGlobalMergerComponent( const AliHLTTPCCAGlobalMergerComponent & );
    AliHLTTPCCAGlobalMergerComponent &operator=( const AliHLTTPCCAGlobalMergerComponent & );

    /** set configuration parameters **/
    void SetDefaultConfiguration();
    int ReadConfigurationString(  const char* arguments );
    int ReadCDBEntry( const char* cdbEntry, const char* chainId );
    int Configure( const char* cdbEntry, const char* chainId, const char *commandLine );

    /** the global merger object */
    AliHLTTPCCAMerger *fGlobalMerger; //!

    double fSolenoidBz;  // magnetic field

    ClassDef( AliHLTTPCCAGlobalMergerComponent, 0 )
};

#endif
