// XEMacs -*-C++-*-
#ifndef AliHLTHOMERREADER_H
#define AliHLTHOMERREADER_H
/* This file is property of and copyright by the ALICE HLT Project        * 
 * ALICE Experiment at CERN, All rights reserved.                         *
 * See cxx source for full Copyright notice                               */

/** @file   AliHLTHOMERReader.h
    @author Timm Steinbeck
    @date   Sep 14 2007
    @brief  HLT Online Monitoring Environment including ROOT - Reader
    @note   migrated from PubSub HLT-stable-20070905.141318 (rev 2375)
            2014-05-08 included to ALFA for HOMER support
*/

#include <climits>
#include <sys/ipc.h>
#include <sys/shm.h>
#include "AliHLTHOMERData.h"



/**
 * @class AliHLTMonitoringReader
 * The class provides a virtual interface for the HOMER reader.
 * Used for dynamic generation of HOMER readers and dynamic loading of
 * the libAliHLTHOMER library.
 * @see AliHLTHOMERLibManager
 * 
 * @ingroup alihlt_homer
 */
class AliHLTMonitoringReader
    {
    public:

	AliHLTMonitoringReader() = default;
	virtual ~AliHLTMonitoringReader() = default;

	/* Return the status of the connection as established by one of the constructors.
	   0 means connection is ok, non-zero specifies the type of error that occured. */
        virtual int GetConnectionStatus() const = 0;

	/* Return the index of the connection for which an error given by the above
	   function occured. */
        virtual unsigned int GetErrorConnectionNdx() const = 0;
	
	/* Read in the next available event */
	virtual int ReadNextEvent() = 0;
	/* Read in the next available event, wait max. timeout microsecs. */
	virtual int ReadNextEvent( unsigned long timeout ) = 0;
	
	/* Return the type of the current event */
	virtual homer_uint64 GetEventType() const = 0;

	/* Return the ID of the current event */
	virtual homer_uint64 GetEventID() const = 0;
	
	/* Return the number of data blocks in the current event */
	virtual unsigned long GetBlockCnt() const = 0;
	
	/* Return the size (in bytes) of the current event's data
	   block with the given block index (starting at 0). */
	virtual unsigned long GetBlockDataLength( unsigned long ndx ) const = 0;
	/* Return a pointer to the start of the current event's data
	   block with the given block index (starting at 0). */
	virtual const void* GetBlockData( unsigned long ndx ) const = 0;
	/* Return IP address or hostname of node which sent the 
	   current event's data block with the given block index 
	   (starting at 0). */
	virtual const char* GetBlockSendNodeID( unsigned long ndx ) const = 0;
	/* Return byte order of the data stored in the 
	   current event's data block with the given block 
	   index (starting at 0). 
	   0 is unknown alignment, 
	   1 ist little endian, 
	   2 is big endian. */
	virtual homer_uint8 GetBlockByteOrder( unsigned long ndx ) const = 0;
	/* Return the alignment (in bytes) of the given datatype 
	   in the data stored in the current event's data block
	   with the given block index (starting at 0). 
	   Possible values for the data type are
	   0: homer_uint64
	   1: homer_uint32
	   2: uin16
	   3: homer_uint8
	   4: double
	   5: float
	*/
	virtual homer_uint8 GetBlockTypeAlignment( unsigned long ndx, homer_uint8 dataType ) const = 0;

	virtual homer_uint64 GetBlockStatusFlags( unsigned long ndx ) const = 0;

	/* Return the type of the data in the current event's data
	   block with the given block index (starting at 0). */
	virtual homer_uint64 GetBlockDataType( unsigned long ndx ) const = 0;
	/* Return the origin of the data in the current event's data
	   block with the given block index (starting at 0). */
	virtual homer_uint32 GetBlockDataOrigin( unsigned long ndx ) const = 0;
	/* Return a specification of the data in the current event's data
	   block with the given block index (starting at 0). */
	virtual homer_uint32 GetBlockDataSpec( unsigned long ndx ) const = 0;

	/* Find the next data block in the current event with the given
	   data type, origin, and specification. Returns the block's 
	   index. */
	virtual unsigned long FindBlockNdx( homer_uint64 type, homer_uint32 origin, 
				    homer_uint32 spec, unsigned long startNdx=0 ) const = 0;

	/* Find the next data block in the current event with the given
	   data type, origin, and specification. Returns the block's 
	   index. */
	virtual unsigned long FindBlockNdx( char type[8], char origin[4], 
				    homer_uint32 spec, unsigned long startNdx=0 ) const = 0;
#ifdef USE_ROOT
        ClassDefOverride(AliHLTMonitoringReader,1);
#endif
    };



class AliHLTHOMERReader: public AliHLTMonitoringReader
    {
    public:
#ifdef USE_ROOT
	AliHLTHOMERReader();
#endif

	/* Constructors & destructors, HOMER specific */
	/* For reading from a TCP port */
	AliHLTHOMERReader( const char* hostname, unsigned short port );
	/* For reading from multiple TCP ports */
	AliHLTHOMERReader( unsigned int tcpCnt, const char** hostnames, const unsigned short* ports );
	/* For reading from a System V shared memory segment */
	AliHLTHOMERReader( key_t shmKey, int shmSize );
	/* For reading from multiple System V shared memory segments */
	AliHLTHOMERReader( unsigned int shmCnt, const key_t* shmKey, const int* shmSize );
	/* For reading from multiple TCP ports and multiple System V shared memory segments */
	AliHLTHOMERReader( unsigned int tcpCnt, const char** hostnames, const unsigned short* ports, 
		     unsigned int shmCnt, const key_t* shmKey, const int* shmSize );
	/* For reading from a buffer */
	AliHLTHOMERReader( const void* pBuffer, int size );
	~AliHLTHOMERReader() override;

	/* Return the status of the connection as established by one of the constructors.
	   0 means connection is ok, non-zero specifies the type of error that occured. */
	int GetConnectionStatus() const override
		{
		return fConnectionStatus;
		}

	/* Return the index of the connection for which an error given by the above
	   function occured. */
	unsigned int GetErrorConnectionNdx() const override
		{
		return fErrorConnection;
		}

	void SetEventRequestAdvanceTime( unsigned long time )
		{
		// advance time in us
		fEventRequestAdvanceTime = time;
		}

	/* Defined in AliHLTMonitoringReader */
	/** Read in the next available event */
	int  ReadNextEvent() override;
	/** Read in the next available event */
	int ReadNextEvent( unsigned long timeout ) override;

	/** Return the type of the current event */
	homer_uint64 GetEventType() const override
		{
		return fCurrentEventType;
		}

	/** Return the ID of the current event */
	homer_uint64 GetEventID() const override
		{
		return fCurrentEventID;
		}

	/** Return the number of data blocks in the current event */
	unsigned long GetBlockCnt() const override
		{
		return fBlockCnt;
		}

	/** Return a pointer to the start of the current event's data
	   block with the given block index (starting at 0). */
	const void* GetBlockData( unsigned long ndx ) const override;
	/** Return the size (in bytes) of the current event's data
	   block with the given block index (starting at 0). */
	unsigned long GetBlockDataLength( unsigned long ndx ) const override;
	/** Return IP address or hostname of node which sent the 
	   current event's data block with the given block index 
	   (starting at 0).
	   For HOMER this is the ID of the node on which the subscriber 
	   that provided this data runs/ran. */
	const char* GetBlockSendNodeID( unsigned long ndx ) const override;
	/** Return byte order of the data stored in the 
	   current event's data block with the given block 
	   index (starting at 0). 
	   0 is unknown alignment, 
	   1 ist little endian, 
	   2 is big endian. */
	homer_uint8 GetBlockByteOrder( unsigned long ndx ) const override;
	/** Return the alignment (in bytes) of the given datatype 
	   in the data stored in the current event's data block
	   with the given block index (starting at 0). 
	   Possible values for the data type are
	   0: homer_uint64
	   1: homer_uint32
	   2: uin16
	   3: homer_uint8
	   4: double
	   5: float
	*/
	homer_uint8 GetBlockTypeAlignment( unsigned long ndx, homer_uint8 dataType ) const override;

	homer_uint64 GetBlockStatusFlags( unsigned long ndx ) const override;

	/* HOMER specific */
	/** Return the type of the data in the current event's data
	   block with the given block index (starting at 0). */
	homer_uint64 GetBlockDataType( unsigned long ndx ) const override;
	/** Return the origin of the data in the current event's data
	   block with the given block index (starting at 0). */
	homer_uint32 GetBlockDataOrigin( unsigned long ndx ) const override;
	/** Return a specification of the data in the current event's data
	   block with the given block index (starting at 0). */
	homer_uint32 GetBlockDataSpec( unsigned long ndx ) const override;

	/** Return the time stamp of when the data block was created.
	   This is a UNIX time stamp in seconds.
	   \param ndx  The index of the block (starting at 0). */
	homer_uint64 GetBlockBirthSeconds( unsigned long ndx ) const;

	/** Return the micro seconds part of the time stamp of when the data
	   block was created.
	   \param ndx  The index of the block (starting at 0). */
	homer_uint64 GetBlockBirthMicroSeconds( unsigned long ndx ) const;

	/** Find the next data block in the current event with the given
	   data type, origin, and specification. Returns the block's 
	   index. */
	unsigned long FindBlockNdx( homer_uint64 type, homer_uint32 origin, 
				    homer_uint32 spec, unsigned long startNdx=0 ) const override;

	/** Find the next data block in the current event with the given
	   data type, origin, and specification. Returns the block's 
	   index. */
	unsigned long FindBlockNdx( char type[8], char origin[4], 
				    homer_uint32 spec, unsigned long startNdx=0 ) const override;
	
	/** Return the ID of the node that actually produced this data block.
	   This may be different from the node which sent the data to this
	   monitoring object as returned by GetBlockSendNodeID. */
	const char* GetBlockCreateNodeID( unsigned long ndx ) const;

    protected:

      enum DataSourceType { kUndef=0, kTCP, kShm, kBuf};
	struct DataSource
	    {
	        DataSourceType fType; // source type (TCP or Shm)
		unsigned fNdx; // This source's index
		const char* fHostname; // Filled for both Shm and TCP
	        unsigned short fTCPPort; // port if type TCP
	        key_t fShmKey; // shm key if type Shm
	        int fShmSize; // shm size if type Shm
		int fTCPConnection; // File descriptor for the TCP connection
		int fShmID; // ID of the shared memory area
		void* fShmPtr; // Pointer to shared memory area
		void* fData; // Pointer to data read in for current event from this source
		unsigned long fDataSize; // Size of data (to be) read in for current event from this source
		unsigned long fDataRead; // Data actually read for current event
	    };

	void Init();
	
	bool AllocDataSources( unsigned int sourceCnt );
	int AddDataSource( const char* hostname, unsigned short port, DataSource& source );
	int AddDataSource( key_t shmKey, int shmSize, DataSource& source );
        int AddDataSource( void* pBuffer, int size, DataSource& source );
	void FreeDataSources();
	int FreeShmDataSource( DataSource& source );
	int FreeTCPDataSource( DataSource& source );
	int ReadNextEvent( bool useTimeout, unsigned long timeout );
	void ReleaseCurrentEvent();
	int TriggerTCPSource( DataSource& source, bool useTimeout, unsigned long timeout );
	int TriggerShmSource( DataSource& source, bool useTimeout, unsigned long timeout ) const;
	int ReadDataFromTCPSources( unsigned sourceCnt, DataSource* sources, bool useTimeout, unsigned long timeout );
	int ReadDataFromShmSources( unsigned sourceCnt, DataSource* sources, bool useTimeout, unsigned long timeout );
	int ParseSourceData( const DataSource& source );
	int ReAllocBlocks( unsigned long newCnt );
	homer_uint64 GetSourceEventID( const DataSource& source );
	homer_uint64 GetSourceEventType( const DataSource& source );
        homer_uint64 Swap( homer_uint8 destFormat, homer_uint8 sourceFormat, homer_uint64 source ) const;

        homer_uint32 Swap( homer_uint8 destFormat, homer_uint8 sourceFormat, homer_uint32 source ) const;


	struct DataBlock
	    {
		unsigned int fSource; // Index of originating data source
	        void* fData; // pointer to data
	        unsigned long fLength; // buffer length
		homer_uint64* fMetaData; // Pointer to meta data describing data itself.
	        const char* fOriginatingNodeID; // node id from which the data originates
	    };

        /** type of the current event */
      	homer_uint64 fCurrentEventType;                             //!transient
      	/** ID of the current event */
      	homer_uint64 fCurrentEventID;                               //!transient
      	/** no of blocks currently used */
      	unsigned long fBlockCnt;                                    //!transient
      	/** available space in the block array */
      	unsigned long fMaxBlockCnt;                                 //!transient
      	/** block array */
      	DataBlock* fBlocks;                                         //!transient
      		
      	/** total no of data sources */
      	unsigned int fDataSourceCnt;                                //!transient
      	/** no of TCP data sources */
      	unsigned int fTCPDataSourceCnt;                             //!transient
      	/** no of Shm data sources */
      	unsigned int fShmDataSourceCnt;                             //!transient
      	/** available space in the sources array */
      	unsigned int fDataSourceMaxCnt;                             //!transient
      	/** array of data source descriptions */
      	DataSource* fDataSources;                                   //!transient
      	
      	/** status of the connection */
      	int fConnectionStatus;                                      //!transient
      	/** flag an error for */
      	unsigned fErrorConnection;                                  //!transient
      	
      	/** */
      	unsigned long fEventRequestAdvanceTime;                     //!transient
    private:
      	/** copy constructor prohibited */
      	AliHLTHOMERReader(const AliHLTHOMERReader&);
      	/** assignment operator prohibited */
      	AliHLTHOMERReader& operator=(const AliHLTHOMERReader&);
      	
#ifdef USE_ROOT
        ClassDefOverride(AliHLTHOMERReader,2);
#endif
    };

/** defined for backward compatibility */
typedef AliHLTMonitoringReader MonitoringReader;
/** defined for backward compatibility */
typedef AliHLTHOMERReader HOMERReader;

// external interface of the HOMER reader
#define ALIHLTHOMERREADER_CREATE_FROM_TCPPORT  "AliHLTHOMERReaderCreateFromTCPPort"
#define ALIHLTHOMERREADER_CREATE_FROM_TCPPORTS "AliHLTHOMERReaderCreateFromTCPPorts"
#define ALIHLTHOMERREADER_CREATE_FROM_BUFFER   "AliHLTHOMERReaderCreateFromBuffer"
#define ALIHLTHOMERREADER_DELETE               "AliHLTHOMERReaderDelete"

#ifdef __cplusplus
extern "C" {
#endif

  typedef AliHLTHOMERReader* (*AliHLTHOMERReaderCreateFromTCPPort_t)(const char* hostname, unsigned short port );
  typedef AliHLTHOMERReader* (*AliHLTHOMERReaderCreateFromTCPPorts_t)(unsigned int tcpCnt, const char** hostnames, unsigned short* ports);
  typedef AliHLTHOMERReader* (*AliHLTHOMERReaderCreateFromBuffer_t)(const void* pBuffer, int size);
  typedef void (*AliHLTHOMERReaderDelete_t)(AliHLTHOMERReader* pInstance);

  /**
   * Create instance of HOMER reader working on a TCP port.
   */
  AliHLTHOMERReader* AliHLTHOMERReaderCreateFromTCPPort(const char* hostname, unsigned short port );
  
  /**
   * Create instance of HOMER reader working on multiple TCP ports.
   */
  AliHLTHOMERReader* AliHLTHOMERReaderCreateFromTCPPorts(unsigned int tcpCnt, const char** hostnames, unsigned short* ports);

  /**
   * Create instance of HOMER reader working on buffer.
   */
  AliHLTHOMERReader* AliHLTHOMERReaderCreateFromBuffer(const void* pBuffer, int size);

  /**
   * Delete instance of HOMER reader.
   */
  void AliHLTHOMERReaderDelete(AliHLTHOMERReader* pInstance);
#ifdef __cplusplus
}
#endif

#endif /* AliHLTHOMERREADER_H */
