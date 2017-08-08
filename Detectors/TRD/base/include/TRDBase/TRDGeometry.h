// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRDGEOMETRY_H
#define O2_TRDGEOMETRY_H

#include <string>
#include <vector>

class TGeoHMatrix;

namespace o2
{
namespace trd
{
class TRDPadPlane;

class TRDGeometry
{
 public:
  enum { kNlayer = 6, kNstack = 5, kNsector = 18, kNdet = 540, kNdets = 30 };

  TRDGeometry();
  ~TRDGeometry();

  void CreateGeometry(std::vector<int> const& idtmed);
  int IsVersion() { return 1; }
  bool IsHole(int la, int st, int se) const;
  bool IsOnBoundary(int det, float y, float z, float eps = 0.5) const;
  bool RotateBack(int det, const double* const loc, double* glb) const;

  // bool           ChamberInGeometry(int det);

  void AssembleChamber(int ilayer, int istack);
  void CreateFrame(std::vector<int> const& idtmed);
  void CreateServices(std::vector<int> const& idtmed);

  //  static  bool           CreateClusterMatrixArray();
  // static  TGeoHMatrix     *GetClusterMatrix(int det);

  void SetSMstatus(int sm, char status) { fgSMstatus[sm] = status; }
  static int GetDetectorSec(int layer, int stack);
  static int GetDetector(int layer, int stack, int sector);
  static int GetLayer(int det);
  static int GetStack(int det);
  int GetStack(double z, int layer);
  static int GetSector(int det);

  static void CreatePadPlaneArray();
  static TRDPadPlane* CreatePadPlane(int layer, int stack);
  static TRDPadPlane* GetPadPlane(int layer, int stack);
  static TRDPadPlane* GetPadPlane(int det) { return GetPadPlane(GetLayer(det), GetStack(det)); }
  static int GetRowMax(int layer, int stack, int /*sector*/);
  static int GetColMax(int layer);
  static double GetRow0(int layer, int stack, int /*sector*/);
  static double GetCol0(int layer);

  static float GetTime0(int layer) { return fgkTime0[layer]; }
  static double GetXtrdBeg() { return fgkXtrdBeg; }
  static double GetXtrdEnd() { return fgkXtrdEnd; }
  char GetSMstatus(int sm) const { return fgSMstatus[sm]; }
  static float GetChamberWidth(int layer) { return fgkCwidth[layer]; }
  static float GetChamberLength(int layer, int stack) { return fgkClength[layer][stack]; }
  static double GetAlpha() { return 2.0 * 3.14159265358979324 / fgkNsector; }
  static int Nsector() { return fgkNsector; }
  static int Nlayer() { return fgkNlayer; }
  static int Nstack() { return fgkNstack; }
  static int Ndet() { return fgkNdet; }
  static float Cheight() { return fgkCH; }
  static float CheightSV() { return fgkCHsv; }
  static float Cspace() { return fgkVspace; }
  static float CraHght() { return fgkCraH; }
  static float CdrHght() { return fgkCdrH; }
  static float CamHght() { return fgkCamH; }
  static float CroHght() { return fgkCroH; }
  static float CsvHght() { return fgkCsvH; }
  static float CroWid() { return fgkCroW; }
  static float AnodePos() { return fgkAnodePos; }
  static float MyThick() { return fgkRMyThick; }
  static float DrThick() { return fgkDrThick; }
  static float AmThick() { return fgkAmThick; }
  static float DrZpos() { return fgkDrZpos; }
  static float RpadW() { return fgkRpadW; }
  static float CpadW() { return fgkCpadW; }
  static float Cwidcha() { return (fgkSwidth2 - fgkSwidth1) / fgkSheight * (fgkCH + fgkVspace); }
  static int MCMmax() { return fgkMCMmax; }
  static int MCMrow() { return fgkMCMrow; }
  static int ROBmaxC0() { return fgkROBmaxC0; }
  static int ROBmaxC1() { return fgkROBmaxC1; }
  static int ADCmax() { return fgkADCmax; }
  static int TBmax() { return fgkTBmax; }
  static int Padmax() { return fgkPadmax; }
  static int Colmax() { return fgkColmax; }
  static int RowmaxC0() { return fgkRowmaxC0; }
  static int RowmaxC1() { return fgkRowmaxC1; }
  std::vector<std::string> const& getSensitiveTRDVolumes() const { return mSensitiveVolumeNames; }
 protected:
  static const int fgkNsector; //  Number of sectors in the full detector (18)
  static const int fgkNlayer;  //  Number of layers of the TRD (6)
  static const int fgkNstack;  //  Number of stacks in z-direction (5)
  static const int fgkNdet;    //  Total number of detectors (18 * 6 * 5 = 540)

  static const float fgkTlength; //  Length of the TRD-volume in spaceframe (BTRD)

  static const float fgkSheight; //  Height of the supermodule
  static const float fgkSwidth1; //  Lower width of the supermodule
  static const float fgkSwidth2; //  Upper width of the supermodule
  static const float fgkSlength; //  Length of the supermodule

  static const float fgkFlength; //  Length of the service space in front of a supermodule

  static const float fgkSMpltT; //  Thickness of the super module side plates

  static const float fgkCraH; //  Height of the radiator part of the chambers
  static const float fgkCdrH; //  Height of the drift region of the chambers
  static const float fgkCamH; //  Height of the amplification region of the chambers
  static const float fgkCroH; //  Height of the readout of the chambers
  static const float fgkCsvH; //  Height of the services on top of the chambers
  static const float fgkCH;   //  Total height of the chambers (w/o services)
  static const float fgkCHsv; //  Total height of the chambers (with services)

  static const float fgkAnodePos; //  Distance of anode wire plane relative to alignabl volume

  static const float fgkVspace; //  Vertical spacing of the chambers
  static const float fgkHspace; //  Horizontal spacing of the chambers
  static const float fgkVrocsm; //  Radial distance of the first ROC to the outer SM plates

  static const float fgkCalT;    //  Thickness of the lower aluminum frame
  static const float fgkCalW;    //  Width of additional aluminum ledge on lower frame
  static const float fgkCalH;    //  Height of additional aluminum ledge on lower frame
  static const float fgkCalWmod; //  Width of additional aluminum ledge on lower frame
  static const float fgkCalHmod; //  Height of additional aluminum ledge on lower frame
  static const float fgkCwsW;    //  Width of additional wacosit ledge on lower frame
  static const float fgkCwsH;    //  Height of additional wacosit ledge on lower frame
  static const float fgkCclsT;   //  Thickness of the lower Wacosit frame sides
  static const float fgkCclfT;   //  Thickness of the lower Wacosit frame front
  static const float fgkCglT;    //  Thichness of the glue around the radiator
  static const float fgkCcuTa;   //  Thickness of the upper Wacosit frame around amp. region
  static const float fgkCcuTb;   //  Thickness of the upper Wacosit frame around amp. region
  static const float fgkCauT;    //  Thickness of the aluminum frame of the back panel
  static const float fgkCroW;    //  Additional width of the readout chamber frames

  static const float fgkCpadW; //  Difference of outer chamber width and pad plane width
  static const float fgkRpadW; //  Difference of outer chamber width and pad plane width

  static const float fgkXeThick; //  Thickness of the gas volume
  static const float fgkDrThick; //  Thickness of the drift region
  static const float fgkAmThick; //  Thickness of the amplification region
  static const float fgkWrThick; //  Thickness of the wire planes

  static const float fgkPPdThick; //  Thickness of copper of the pad plane
  static const float fgkPPpThick; //  Thickness of PCB board of the pad plane
  static const float fgkPGlThick; //  Thickness of the glue layer
  static const float fgkPCbThick; //  Thickness of the carbon layers
  static const float fgkPHcThick; //  Thickness of the honeycomb support structure
  static const float fgkPPcThick; //  Thickness of the PCB readout boards
  static const float fgkPRbThick; //  Thickness of the PCB copper layers
  static const float fgkPElThick; //  Thickness of all other electronics components (caps, etc.)

  static const float fgkRFbThick; //  Thickness of the fiber layers in the radiator
  static const float fgkRRhThick; //  Thickness of the rohacell layers in the radiator
  static const float fgkRGlThick; //  Thickness of the glue layers in the radiator
  static const float fgkRCbThick; //  Thickness of the carbon layers in the radiator
  static const float fgkRMyThick; //  Thickness of the mylar layers in the radiator

  static const float fgkDrZpos;  //  Position of the drift region
  static const float fgkAmZpos;  //  Position of the amplification region
  static const float fgkWrZposA; //  Position of the wire planes
  static const float fgkWrZposB; //  Position of the wire planes
  static const float fgkCalZpos; //  Position of the additional aluminum ledges

  static const int fgkMCMmax;   //  Maximum number of MCMs per ROB
  static const int fgkMCMrow;   //  Maximum number of MCMs per ROB Row
  static const int fgkROBmaxC0; //  Maximum number of ROBs per C0 chamber
  static const int fgkROBmaxC1; //  Maximum number of ROBs per C1 chamber
  static const int fgkADCmax;   //  Maximum number of ADC channels per MCM
  static const int fgkTBmax;    //  Maximum number of Time bins
  static const int fgkPadmax;   //  Maximum number of pads per MCM
  static const int fgkColmax;   //  Maximum number of pads per padplane row
  static const int fgkRowmaxC0; //  Maximum number of Rows per C0 chamber
  static const int fgkRowmaxC1; //  Maximum number of Rows per C1 chamber

  static const float fgkCwidth[kNlayer];           //  Outer widths of the chambers
  static const float fgkClength[kNlayer][kNstack]; //  Outer lengths of the chambers

  static const double fgkTime0Base;     //  Base value for calculation of Time-position of pad 0
  static const float fgkTime0[kNlayer]; //  Time-position of pad 0

  static const double fgkXtrdBeg; //  X-coordinate in tracking system of begin of TRD mother volume
  static const double fgkXtrdEnd; //  X-coordinate in tracking system of end of TRD mother volume

  static TObjArray* fgClusterMatrixArray; //! Transformation matrices loc. cluster to tracking cs
  static std::vector<TRDPadPlane*>* fgPadPlaneArray;

  static char fgSMstatus[kNsector]; //  Super module status byte

 private:
  std::vector<std::string> mSensitiveVolumeNames; //!< vector keeping track of sensitive TRD volumes

  // helper function to create volumes and registering them automatically
  void createVolume(const char* name, const char* shape, int nmed, float* upar, int np);

  ClassDefNV(TRDGeometry, 1) //  TRD geometry class
};
} // end namespace trd
} // end namespace o2
#endif
