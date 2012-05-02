// $Id$
#ifndef ALIHLTTPCCACOMPRESSEDINPUTDATA_H
#define ALIHLTTPCCACOMPRESSEDINPUTDATA_H

#ifdef HLTCA_STANDALONE
#include "AliHLTTPCRootTypes.h"
#else
#include "Rtypes.h"
#endif

/**
 * @struct AliHLTTPCCACompressedCluster
 * Data structure to pack the TPC clusters
 * before send them to the TPCCASliceTracker component.
 * Data is packed in 8 bytes: fP0={p03,..,p00}, fP1={p13,..,p10} 
 * X cluster(    p13,p03) = rowX+( ((fP1&0xF000)>>16)+(fP0>>24)-32768. )*1.e-4
 * Y cluster(p02,p01,p00) = ( (fP0&0xFFF) - 8388608. )*1.e-4
 * Z cluster(p12,p11,p10) = ( (fP1&0xFFF) - 8388608. )*1.e-4
 */
struct AliHLTTPCCACompressedCluster{
  UInt_t fP0;       // First  4 bytes of the packed data
  UInt_t fP1;       // Second 4 bytes of the packed data
};
typedef struct AliHLTTPCCACompressedCluster AliHLTTPCCACompressedCluster;


/**
 * @struct AliHLTTPCCACompressedClusterRow
 * Data structure to pack the TPC clusters
 * before send them to the TPCCASliceTracker component.
 * contains the PadRow information
 */
struct AliHLTTPCCACompressedClusterRow
{
  UShort_t fSlicePatchRowID; // packed slice, patch and row number 
                             // ((slice<<10)+(patch<<6)+row)
                             // the row number is local withing the patch
  UShort_t fNClusters;  // number of clusters in the row

#if defined(__HP_aCC) || defined(__DECCXX) || defined(__SUNPRO_CC)
  AliHLTTPCCACompressedCluster  fClusters[1]; // array of assigned clusters  
#else
  AliHLTTPCCACompressedCluster  fClusters[0]; // array of assigned clusters 
#endif
};
typedef struct AliHLTTPCCACompressedClusterRow AliHLTTPCCACompressedClusterRow;


#endif
