
# ifndef  _CLUSTERPROCESSING_H
# define  _CLUSTERPROCESSING_H

typedef std::pair< int, const double*> DataBlock_t;

extern "C" {
  void setMathiesonVarianceApprox( int chId, double *theta, int K );
  
  int clusterProcess( const double *xyDxyi, const Mask_t *cathi, const Mask_t *saturated, const double *zi, 
                       int chId, int nPads);
  
  void collectTheta( double *theta, Group_t *thetaToGroup, int N);
  
  int getNbrOfPadsInGroups();

  int getNbrOfProjPads();
  
  void collectPadsAndCharges( double *xyDxy, double *z, Group_t *padToGroup, int nTot);
  
  void collectLaplacian( double *laplacian, int N);

  int getKThetaInit();
  
  void collectThetaInit( double *thetai, int N);
  
  int getNbrOfThetaEMFinal();

  void collectThetaEMFinal( double *thetaEM, int K);
  
  void cleanClusterProcessVariables( int uniqueCath);
}
# endif
