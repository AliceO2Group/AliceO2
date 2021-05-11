# ifndef  _PADPROCESSING_H
# define  _PADPROCESSING_H

typedef struct {
    PadIdx_t i;
    PadIdx_t j;
} MapKToIJ_t;

typedef short Group_t;

// 8 neigbours + the center pad itself + separator (-1)
static const int MaxNeighbors = 12;

inline static PadIdx_t *getNeighborsOf( PadIdx_t *neigh,  PadIdx_t i) { return &neigh[MaxNeighbors*i]; };
inline static PadIdx_t getTheFirstNeighborOf( PadIdx_t *neigh,  PadIdx_t i) { return neigh[MaxNeighbors*i]; };

extern "C" {
    
  // Helper to get results from Python
  int getNbrProjectedPads();
  //
  void setNbrProjectedPads( int n);
  
  void copyProjectedPads(double *xyDxy, double *chA, double *chB);
  
  PadIdx_t *getFirstNeighbors( const double *xyDxy, int N, int allocatedN);
  
  // Used when there is 1 cathode
  // The neighbors list must be computed for the laplacian
  void computeAndStoreFirstNeighbors( const double *xyDxy, int N, int allocatedN);
  
  int projectChargeOnOnePlane( 
          const double *xy0InfSup, const double *ch0, 
          const double *xy1InfSup, const double *ch1, 
          PadIdx_t N0, PadIdx_t N1, int includeAlonePads);
  
  void buildProjectedSaturatedPads( const Mask_t *saturated0, const Mask_t *saturated1, Mask_t *saturatedProj);

  int getConnectedComponentsOfProjPads( short *padGrp );
  
  int findLocalMaxWithLaplacian( const double *xyDxy, const double *z,
        Group_t *padToGrp,
        int nGroups,
        int N, int K, double *laplacian,  double *theta, PadIdx_t *thetaIndexes, 
        Group_t *thetaToGrp);

  // ??? To remove
  int findLocalMaxWithLaplacianV0( const double *xyDxy, const double *z, const PadIdx_t *grpIdxToProjIdx, int N, int xyDxyAllocated, double *laplacian,  double *theta);
  
  void assignOneCathPadsToGroup( short *padGroup, int nPads, int nGrp, int nCath0, int nCath1, short *wellSplitGroup);
  
  void assignCathPadsToGroupFromProj( short *padGroup, int nPads, int nGrp, int nCath0, int nCath1, 
                                      short *wellSplitGroup, short *matGrpGrp);

  int assignCathPadsToGroup( short *matGrpGrp, int nGrp, int nCath0, int nCath1, short *grpToGrp );
  
  void storeProjectedPads ( const double *xyDxyProj, const double *z, int nPads);
  //
  void copyCathToGrpFromProj( short *cath0Grp, short *cath1Grp, int nCath0, int nCath1);
  
  void getMaskCathToGrpFromProj( short g, short* mask0, short *mask1, int nCath0, int nCath1);
  
  void freeMemoryPadProcessing();
  
  void printMatrixShort( const char *str, const short *matrix, int N, int M);
  void printMatrixInt  ( const char *str, const int   *matrix, int N, int M);
}
#endif // _PADPROCESSING_H

