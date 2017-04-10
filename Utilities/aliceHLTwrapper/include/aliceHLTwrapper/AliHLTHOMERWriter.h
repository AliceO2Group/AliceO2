// XEMacs -*-C++-*-
#ifndef ALIHLTHOMERWRITER_H
#define ALIHLTHOMERWRITER_H
/************************************************************************
**
**
** This file is property of and copyright by the Technical Computer
** Science Group, Kirchhoff Institute for Physics, Ruprecht-Karls-
** University, Heidelberg, Germany, 2001
** This file has been written by Timm Morten Steinbeck, 
** timm@kip.uni-heidelberg.de
**
**
** See the file license.txt for details regarding usage, modification,
** distribution and warranty.
** Important: This file is provided without any warranty, including
** fitness for any particular purpose.
**
**
** Newer versions of this file's package will be made available from 
** http://web.kip.uni-heidelberg.de/Hardwinf/L3/ 
** or the corresponding page of the Heidelberg Alice Level 3 group.
**
*************************************************************************/

/*
***************************************************************************
**
** $Author$ - Initial Version by Timm Morten Steinbeck
**
** $Id$ 
**
***************************************************************************
*/

/** @file   AliHLTHOMERWriter.h
    @author Timm Steinbeck
    @date   Sep 14 2007
    @brief  HLT Online Monitoring Environment including ROOT - Writer   
    @note   migrated from PubSub HLT-stable-20070905.141318 (rev 2375)    
            2014-05-08 included to ALFA for HOMER support
*/

// see below for class documentation
// or
// refer to README to build package
// or
// visit http://web.ift.uib.no/~kjeks/doc/alice-hlt


#include "AliHLTHOMERData.h"
#include <vector>

/**
 * @class AliHLTMonitoringWriter
 * A pure virtual interface definition for HLT monitoring writers.
 * 
 * @ingroup alihlt_homer
 */
class AliHLTMonitoringWriter
    {
    public:
        AliHLTMonitoringWriter() = default;
        virtual ~AliHLTMonitoringWriter() = default;

	virtual void Clear() = 0;

	virtual void AddBlock( const void* descriptor, const void* data ) = 0;

	virtual homer_uint32 GetTotalMemorySize( bool includeData = true ) = 0;
	virtual void Copy( void* destination, homer_uint64 eventType, homer_uint64 eventNr, homer_uint64 statusFlags, homer_uint64 nodeID, bool includeData = true ) = 0;
     };

/**
 * @class AliHLTHOMERWriter
 * The HOMER writer assembles several data blocks of different properties
 * into one big buffer and adds meta information to describe the data's
 * alignment and byte order.
 */
class AliHLTHOMERWriter : public AliHLTMonitoringWriter
    {
    public:

	AliHLTHOMERWriter();
	~AliHLTHOMERWriter() override;

        /**
	 * Resets the writer and clears the block list.
         */
	void Clear() override;

        /**
         * Add a data block to the writer.
	 * The HOMER header must contain all meta information including the
	 * size of the data.
         * @param homerHeader    pointer to the header describing the block
	 * @param data           pointer to data
         */
	void AddBlock( const void* homerHeader, const void* data ) override;

        /**
         * Add a data block to the writer.
         * The function has certainly been introduced to make type
         * conversion easier. In fact it makes it worse. The presence of the
         * function with void* argument leads to a wrong interpretation when
         * passing a non const pointer to HOMERBlockDescriptor. Then the
         * other function is called directly, leading to pointer mess up.
         */
	void AddBlock( const HOMERBlockDescriptor* descriptor, const void* data )
		{
		AddBlock( descriptor->GetHeader(), data );
		}

        /**
         * Add a data block to the writer.
         * Function added to avoid potential pointer mismatches
         */
	void AddBlock( HOMERBlockDescriptor* descriptor, const void* data )
		{
		AddBlock( descriptor->GetHeader(), data );
		}

        /**
         * Get the total buffer size required to write all data into one buffer
         */
	homer_uint32 GetTotalMemorySize( bool includeData = true ) override;

        /**
         * Copy the data into a buffer.
         * The buffer is supposed to be big enough, the capacity should be queried
         * by calling @ref GetTotalMemorySize.
         */
	void Copy( void* destination, homer_uint64 eventType, homer_uint64 eventNr, homer_uint64 statusFlags, homer_uint64 nodeID, bool includeData = true ) override;

        /** determine alignment of 64 bit variables */
	static homer_uint8 DetermineUInt64Alignment();
        /** determine alignment of 32 bit variables */
	static homer_uint8 DetermineUInt32Alignment();
        /** determine alignment of 16 bit variables */
	static homer_uint8 DetermineUInt16Alignment();
        /** determine alignment of 8 bit variables */
	static homer_uint8 DetermineUInt8Alignment();
        /** determine alignment of double type variables */
	static homer_uint8 DetermineDoubleAlignment();
        /** determine alignment of float type bit variables */
	static homer_uint8 DetermineFloatAlignment();


        /** test structure for the alignment determination of 64 bit variables */
        struct AliHLTHOMERWriterAlignment64TestStructure
        {
        	homer_uint64 f64Fill;   // !
        	homer_uint64 f64Test64; // !
        	homer_uint32 f32Fill;   // !
        	homer_uint64 f64Test32; // !
        	homer_uint16 f16Fill;   // !
        	homer_uint64 f64Test16; // !
        	homer_uint8  f8Fill;    // !
        	homer_uint64 f64Test8;  // !
        };
        /** test structure for the alignment determination of 32 bit variables */
        struct AliHLTHOMERWriterAlignment32TestStructure
        {
        	homer_uint64 f64Fill;   // !
        	homer_uint32 f32Test64; // !
        	homer_uint32 f32Fill;   // !
        	homer_uint32 f32Test32; // !
        	homer_uint16 f16Fill;   // !
        	homer_uint32 f32Test16; // !
        	homer_uint8  f8Fill;    // !
        	homer_uint32 f32Test8;  // !
        };
        /** test structure for the alignment determination of 16 bit variables */
        struct AliHLTHOMERWriterAlignment16TestStructure
        {
        	homer_uint64 f64Fill;   // !
            	homer_uint16 f16Test64; // !
        	homer_uint32 f32Fill;   // !
        	homer_uint16 f16Test32; // !
        	homer_uint16 f16Fill;   // !
        	homer_uint16 f16Test16; // !
        	homer_uint8  f8Fill;    // !
        	homer_uint16 f16Test8;  // !
        };
        /** test structure for the alignment determination of 8 bit variables */
        struct AliHLTHOMERWriterAlignment8TestStructure
        {
        	homer_uint64 f64Fill; // !
        	homer_uint8 f8Test64; // !
        	homer_uint32 f32Fill; // !
        	homer_uint8 f8Test32; // !
        	homer_uint16 f16Fill; // !
        	homer_uint8 f8Test16; // !
        	homer_uint8  f8Fill;  // !
        	homer_uint8 f8Test8;  // !
        };
        /** test structure for the alignment determination of double type variables */
        struct AliHLTHOMERWriterAlignmentDoubleTestStructure
        {
        	homer_uint64 f64Fill; // !
        	double fDoubleTest64; // !
        	homer_uint32 f32Fill; // !
        	double fDoubleTest32; // !
        	homer_uint16 f16Fill; // !
        	double fDoubleTest16; // !
        	homer_uint8  f8Fill;  // !
        	double fDoubleTest8;  // !
        };
        /** test structure for the alignment determination of float type variables */
        struct AliHLTHOMERWriterAlignmentFloatTestStructure
        {
        	homer_uint64 f64Fill; // !
        	float fFloatTest64;   // !
        	homer_uint32 f32Fill; // !
        	float fFloatTest32;   // !
        	homer_uint16 f16Fill; // !
        	float fFloatTest16;   // !
        	homer_uint8  f8Fill;  // !
        	float fFloatTest8;    // !
        };
    protected:

        /**
         * Block descriptor structure.
         * The descriptor contains a header for meta information and position
         * and a pointer to the data.
         */
	struct TBlockData
	    {
	      homer_uint64 fDescriptor[kCount_64b_Words]; //!transient
	      const void* fData; //!transient
	    };

        unsigned long fDataOffset; //!transient

        /** list of data blocks */
        std::vector<TBlockData> fBlocks; //!transient
#ifdef USE_ROOT
      ClassDefOverride(AliHLTHOMERWriter,0);
#endif
    };


/** defined for backward compatibility */
typedef AliHLTHOMERWriter HOMERWriter;

// external interface of the HOMER writer
#define ALIHLTHOMERWRITER_CREATE "AliHLTHOMERWriterCreate"
#define ALIHLTHOMERWRITER_DELETE "AliHLTHOMERWriterDelete"

#ifdef __cplusplus
extern "C" {
#endif

  typedef AliHLTHOMERWriter* (*AliHLTHOMERWriterCreate_t)();
  typedef void (*AliHLTHOMERWriterDelete_t)(AliHLTHOMERWriter* pInstance);
  /**
   * Create instance of HOMER writer.
   */
  AliHLTHOMERWriter* AliHLTHOMERWriterCreate();

  /**
   * Delete instance of HOMER writer.
   */
  void AliHLTHOMERWriterDelete(AliHLTHOMERWriter* pInstance);
#ifdef __cplusplus
}
#endif



/*
***************************************************************************
**
** $Author$ - Initial Version by Timm Morten Steinbeck
**
** $Id$ 
**
***************************************************************************
*/

#endif // ALIHLTHOMERWRITER_H
