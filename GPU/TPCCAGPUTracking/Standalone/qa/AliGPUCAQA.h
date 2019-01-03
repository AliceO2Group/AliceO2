#ifndef ALIGPUCAQA
#define ALIGPUCAQA

#include "AliHLTTPCCASettings.h"
#include <math.h>

class AliGPUReconstruction;
class TH1F;
class TH2F;
class TCanvas;
class TPad;
class TLegend;
class TPad;
class TH1;
class TFile;
class TH1D;
typedef short int Color_t;
struct AliHLTTPCClusterMCWeight;

class AliGPUCAQA
{
public:
	AliGPUCAQA(AliGPUReconstruction* rec) : mRec(rec) {}
	~AliGPUCAQA() = default;
	
	struct additionalMCParameters
	{
		float pt, phi, theta, eta, nWeightCls;
	};

	struct additionalClusterParameters
	{
		int attached, fakeAttached, adjacent, fakeAdjacent;
		float pt;
	};
	
	void InitQA();
	void RunQA(bool matchOnly = false);
	int DrawQAHistograms();
	void SetMCTrackRange(int min, int max);
	bool SuppressTrack(int iTrack) const;
	bool SuppressHit(int iHit) const;
	int GetMCLabel(unsigned int trackId) const;
	bool clusterRemovable(int cid, bool prot) const;
	
private:
	void SetAxisSize(TH1F* e);
	void SetLegend(TLegend* l);
	double* CreateLogAxis(int nbins, float xmin, float xmax);
	void ChangePadTitleSize(TPad* p, float size);
	void DrawHisto(TH1* histo, char* filename, char* options);
	void doPerfFigure(float x, float y, float size);
	void GetName(char* fname, int k);
	template <class T> T* GetHist(T* &ee, std::vector<TFile*>& tin, int k, int nNewInput);
	
	AliGPUReconstruction* mRec;
	
	//-------------------------: Some compile time settings....
	static const constexpr bool plotroot = 0;
	static const constexpr bool fixscales = 0;
	static const constexpr bool perffigure = 0;
	static const constexpr float fixedScalesMin[5] = {-0.05, -0.05, -0.2, -0.2, -0.5};
	static const constexpr float fixedScalesMax[5] = {0.4, 0.7, 5, 3, 6.5};
	static const constexpr float logPtMin = -1.;

	const char* str_perf_figure_1 = "ALICE Performance 2018/03/20";
	//const char* str_perf_figure_2 = "2015, MC pp, #sqrt{s} = 5.02 TeV";
	const char* str_perf_figure_2 = "2015, MC Pb-Pb, #sqrt{s_{NN}} = 5.02 TeV";
	//-------------------------

	std::vector<int> trackMCLabels;
	std::vector<int> trackMCLabelsReverse;
	std::vector<int> recTracks;
	std::vector<int> fakeTracks;
	std::vector<additionalClusterParameters> clusterParam;
	std::vector<additionalMCParameters> mcParam;
	int totalFakes = 0;

	TH1F* eff[4][2][2][5][2]; //eff,clone,fake,all - findable - secondaries - y,z,phi,eta,pt - work,result
	TCanvas *ceff[6];
	TPad* peff[6][4];
	TLegend* legendeff[6];

	TH1F* res[5][5][2]; //y,z,phi,lambda,pt,ptlog res - param - res,mean
	TH2F* res2[5][5];
	TCanvas *cres[7];
	TPad* pres[7][5];
	TLegend* legendres[6];

	TH1F* pull[5][5][2]; //y,z,phi,lambda,pt,ptlog res - param - res,mean
	TH2F* pull2[5][5];
	TCanvas *cpull[7];
	TPad* ppull[7][5];
	TLegend* legendpull[6];

	static constexpr int N_CLS_HIST = 8;
	static constexpr int N_CLS_TYPE = 3;
	enum CL_types {CL_attached = 0, CL_fake = 1, CL_att_adj = 2, CL_fakeAdj = 3, CL_tracks = 4, CL_physics = 5, CL_prot = 6, CL_all = 7};
	TH1D* clusters[N_CLS_TYPE * N_CLS_HIST - 1]; //attached, fakeAttached, attach+adjacent, fakeAdjacent, physics, protected, tracks, all / count, rel, integral
	TCanvas* cclust[N_CLS_TYPE];
	TPad* pclust[N_CLS_TYPE];
	TLegend* legendclust[N_CLS_TYPE];

	long long int recClustersRejected = 0,  recClustersTube = 0, recClustersTube200 = 0, recClustersLoopers = 0, recClustersLowPt = 0, recClusters200MeV = 0, recClustersPhysics = 0, recClustersProt = 0, recClustersUnattached = 0, recClustersTotal = 0,
		recClustersHighIncl = 0, recClustersAbove400 = 0, recClustersFakeRemove400 = 0, recClustersFullFakeRemove400 = 0, recClustersBelow40 = 0, recClustersFakeProtect40 = 0;
	double recClustersUnaccessible = 0;

	TH1F* tracks;
	TCanvas* ctracks;
	TPad* ptracks;
	TLegend* legendtracks;

	TH1F* ncl;
	TCanvas* cncl;
	TPad* pncl;
	TLegend* legendncl;

	int nEvents = 0;
	std::vector<std::vector<int>> mcEffBuffer;
	std::vector<std::vector<int>> mcLabelBuffer;
	std::vector<std::vector<bool>> goodTracks;
	std::vector<std::vector<bool>> goodHits;

	#define DEBUG 0
	#define TIMING 0

	static constexpr float Y_MAX = 40;
	static constexpr float Z_MAX = 100;
	static constexpr float PT_MIN = MIN_TRACK_PT_DEFAULT;
	static constexpr float PT_MIN2 = 0.1;
	static constexpr float PT_MIN_PRIM = 0.1;
	static constexpr float PT_MIN_CLUST = MIN_TRACK_PT_DEFAULT;
	static constexpr float PT_MAX = 20;
	static constexpr float ETA_MAX = 1.5;
	static constexpr float ETA_MAX2 = 0.9;

	static constexpr float  MIN_WEIGHT_CLS = 40;
	static constexpr float  FINDABLE_WEIGHT_CLS = 70;

	static constexpr int MC_LABEL_INVALID = -1e9;

	static constexpr bool CLUST_HIST_INT_SUM = false;

	static constexpr const int ColorCount = 12;
	Color_t* colorNums;

	static const constexpr char* EffTypes[4] = {"Rec", "Clone", "Fake", "All"};
	static const constexpr char* FindableNames[2] = {"", "Findable"};
	static const constexpr char* PrimNames[2] = {"Prim", "Sec"};
	static const constexpr char* ParameterNames[5] = {"Y", "Z", "#Phi", "#lambda", "Relative #it{p}_{T}"};
	static const constexpr char* ParameterNamesNative[5] = {"Y", "Z", "sin(#Phi)", "tan(#lambda)", "q/#it{p}_{T} (curvature)"};
	static const constexpr char* VSParameterNames[6] = {"Y", "Z", "Phi", "Eta", "Pt", "Pt_log"};
	static const constexpr char* EffNames[3] = {"Efficiency", "Clone Rate", "Fake Rate"};
	static const constexpr char* EfficiencyTitles[4] = {"Efficiency (Primary Tracks, Findable)", "Efficiency (Secondary Tracks, Findable)", "Efficiency (Primary Tracks)", "Efficiency (Secondary Tracks)"};
	static const constexpr double Scale[5] = {10., 10., 1000., 1000., 100.};
	static const constexpr double ScaleNative[5] = {10., 10., 1000., 1000., 1.};
	static const constexpr char* XAxisTitles[5] = {"#it{y}_{mc} (cm)", "#it{z}_{mc} (cm)", "#Phi_{mc} (rad)", "#eta_{mc}", "#it{p}_{Tmc} (GeV/#it{c})"};
	static const constexpr char* AxisTitles[5] = {"#it{y}-#it{y}_{mc} (mm) (Resolution)", "#it{z}-#it{z}_{mc} (mm) (Resolution)", "#phi-#phi_{mc} (mrad) (Resolution)", "#lambda-#lambda_{mc} (mrad) (Resolution)", "(#it{p}_{T} - #it{p}_{Tmc}) / #it{p}_{Tmc} (%) (Resolution)"};
	static const constexpr char* AxisTitlesNative[5] = {"#it{y}-#it{y}_{mc} (mm) (Resolution)", "#it{z}-#it{z}_{mc} (mm) (Resolution)", "sin(#phi)-sin(#phi_{mc}) (Resolution)", "tan(#lambda)-tan(#lambda_{mc}) (Resolution)", "q*(q/#it{p}_{T} - q/#it{p}_{Tmc}) (Resolution)"};
	static const constexpr char* AxisTitlesPull[5] = {"#it{y}-#it{y}_{mc}/#sigma_{y} (Pull)", "#it{z}-#it{z}_{mc}/#sigma_{z} (Pull)", "sin(#phi)-sin(#phi_{mc})/#sigma_{sin(#phi)} (Pull)", "tan(#lambda)-tan(#lambda_{mc})/#sigma_{tan(#lambda)} (Pull)", "q*(q/#it{p}_{T} - q/#it{p}_{Tmc})/#sigma_{q/#it{p}_{T}} (Pull)"};
	static const constexpr char* ClustersNames[N_CLS_HIST] = {"Correctly attached clusters", "Fake attached clusters", "Attached + adjacent clusters", "Fake adjacent clusters", "Clusters of reconstructed tracks", "Used in Physics", "Protected", "All clusters"};
	static const constexpr char* ClusterTitles[N_CLS_TYPE] = {"Clusters Pt Distribution / Attachment", "Clusters Pt Distribution / Attachment (relative to all clusters)", "Clusters Pt Distribution / Attachment (integrated)"};
	static const constexpr char* ClusterNamesShort[N_CLS_HIST] = {"Attached", "Fake", "AttachAdjacent", "FakeAdjacent", "FoundTracks", "Physics", "Protected", "All"};
	static const constexpr char* ClusterTypes[N_CLS_TYPE] = {"", "Ratio", "Integral"};
	static const constexpr int colorsHex[ColorCount] = {0xB03030, 0x00A000, 0x0000C0, 0x9400D3, 0x19BBBF, 0xF25900, 0x7F7F7F, 0xFFD700, 0x07F707, 0x07F7F7, 0xF08080, 0x000000};

	static const constexpr int ConfigDashedMarkers = 0;

	static const constexpr float kPi = M_PI;
	static const constexpr float axes_min[5] = {-Y_MAX, -Z_MAX, 0.f, -ETA_MAX, PT_MIN};
	static const constexpr float axes_max[5] = {Y_MAX, Z_MAX, 2.f *  kPi, ETA_MAX, PT_MAX};
	static const constexpr int axis_bins[5] = {51, 51, 144, 31, 50};
	static const constexpr int res_axis_bins[] = {1017, 113}; //Consecutive bin sizes, histograms are binned down until the maximum entry is 50, each bin size should evenly divide its predecessor.
	static const constexpr float res_axes[5] = {1., 1., 0.03, 0.03, 1.0};
	static const constexpr float res_axes_native[5] = {1., 1., 0.1, 0.1, 5.0};
	static const constexpr float pull_axis = 10.f;
	
	int mcTrackMin = -1, mcTrackMax = -1;
};

#ifndef BUILD_QA
inline bool AliGPUCAQA::SuppressTrack(int iTrack) const {return false;}
inline bool AliGPUCAQA::SuppressHit(int iHit) const {return false;}
inline int AliGPUCAQA::GetMCLabel(unsigned int trackId) const {return -1;};
inline bool AliGPUCAQA::clusterRemovable(int cid, bool prot) const {return false;}
#endif

#endif
