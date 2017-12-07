#include "Rtypes.h"

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCASliceData.h"
#include "AliHLTTPCCAStandaloneFramework.h"
#include "AliHLTTPCCATrack.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCATrackerFramework.h"
#include "AliHLTTPCGMMergedTrack.h"
#include "AliHLTTPCGMPropagator.h"
#include "include.h"
#include <algorithm>
#include <cstdio>

#include "TH1F.h"
#include "TH2F.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TLegend.h"
#include "TColor.h"
#include "TPaveText.h"
#include "TF1.h"
#include "TFile.h"
#include "TStyle.h"
#include <sys/stat.h>

#include "../cmodules/qconfig.h"
#include "../cmodules/timer.h"

struct additionalMCParameters
{
	float pt, phi, theta, eta, nWeightCls;
};

struct additionalClusterParameters
{
	int attached, fakeAttached;
	float pt;
};

std::vector<int> trackMCLabels;
std::vector<int> trackMCLabelsReverse;
std::vector<int> recTracks;
std::vector<int> fakeTracks;
std::vector<additionalClusterParameters> clusterParam;
std::vector<additionalMCParameters> mcParam;
static int totalFakes = 0;

static TH1F* eff[4][2][2][5][2]; //eff,clone,fake,all - findable - secondaries - y,z,phi,eta,pt - work,result
static TCanvas *ceff[6];
static TPad* peff[6][4];
static TLegend* legendeff[6];

static TH1F* res[5][5][2]; //y,z,phi,lambda,pt,ptlog res - param - res,mean
static TH2F* res2[5][5];
static TCanvas *cres[7];
static TPad* pres[7][5];
static TLegend* legendres[6];

static TH1F* pull[5][5][2]; //y,z,phi,lambda,pt,ptlog res - param - res,mean
static TH2F* pull2[5][5];
static TCanvas *cpull[7];
static TPad* ppull[7][5];
static TLegend* legendpull[6];


static TH1F* clusters[11]; //attached, fakeAttached, tracks, all, attachedRel, fakeAttachedRel, treaksRel, attachedInt, fakeAttachedInt, treaksInt, AllInt
static TCanvas* cclust[3];
static TPad* pclust[3];
static TLegend* legendclust[3];

TH1F* tracks;
TCanvas* ctracks;
TPad* ptracks;
TLegend* legendtracks;

TH1F* ncl;
TCanvas* cncl;
TPad* pncl;
TLegend* legendncl;

#define DEBUG 0
#define TIMING 0

bool MCComp(const AliHLTTPCClusterMCWeight& a, const AliHLTTPCClusterMCWeight& b) {return(a.fMCID > b.fMCID);}

#define Y_MAX 40
#define Z_MAX 100
#define PT_MIN MIN_TRACK_PT_DEFAULT
#define PT_MIN2 0.1
#define PT_MIN_PRIM 0.1
#define PT_MIN_CLUST MIN_TRACK_PT_DEFAULT
#define PT_MAX 20
#define ETA_MAX 1.5
#define ETA_MAX2 0.9

#define MIN_WEIGHT_CLS 40 // SG!!! 40

#define FINDABLE_WEIGHT_CLS 70

static const int ColorCount = 12;
static Color_t colorNums[ColorCount];

static const char* EffTypes[4] = {"Rec", "Clone", "Fake", "All"};
static const char* FindableNames[2] = {"", "Findable"};
static const char* PrimNames[2] = {"Prim", "Sec"};
static const char* ParameterNames[5] = {"Y", "Z", "#Phi", "#lambda", "Relative p_{T}"};
static const char* ParameterNamesNative[5] = {"Y", "Z", "sin(#Phi)", "tan(#lambda)", "q/p_{T} (curvature)"};
static const char* VSParameterNames[6] = {"Y", "Z", "Phi", "Eta", "Pt", "Pt_log"};
static const char* EffNames[3] = {"Efficiency", "Clone Rate", "Fake Rate"};
static const char* EfficiencyTitles[4] = {"Efficiency (Primary Tracks, Findable)", "Efficiency (Secondary Tracks, Findable)", "Efficiency (Primary Tracks)", "Efficiency (Secondary Tracks)"};
static const double Scale[5] = {10., 10., 1000., 1000., 100.};
static const double ScaleNative[5] = {10., 10., 1000., 1000., 1.};
static const char* XAxisTitles[5] = {"y_{mc} [cm]", "z_{mc} [cm]", "#Phi_{mc} [rad]", "#eta_{mc}", "p_{Tmc} [Gev/c]"};
static const char* AxisTitles[5] = {"y-y_{mc} [mm] (Resolution)", "z-z_{mc} [mm] (Resolution)", "#phi-#phi_{mc} [mrad] (Resolution)", "#lambda-#lambda_{mc} [mrad] (Resolution)", "(p_{T} - p_{Tmc}) / p_{Tmc} [%] (Resolution)"};
static const char* AxisTitlesNative[5] = {"y-y_{mc} [mm] (Resolution)", "z-z_{mc} [mm] (Resolution)", "sin(#phi)-sin(#phi_{mc}) (Resolution)", "tan(#lambda)-tan(#lambda_{mc}) (Resolution)", "(q/p_{T} - q/p_{Tmc}) (Resolution)"};
static const char* AxisTitlesPull[5] = {"y-y_{mc}/#sigma_{y} (Pull)", "z-z_{mc}/#sigma_{z} (Pull)", "sin(#phi)-sin(#phi_{mc})/#sigma_{sin(#phi)} (Pull)", "tan(#lambda)-tan(#lambda_{mc})/#sigma_{tan(#lambda)} (Pull)", "(q/p_{T} - q/p_{Tmc})/#sigma{q/p_{T}} (Pull)"};
static const char* ClustersNames[4] = {"Correctly attached clusters", "Fake attached clusters", "Clusters of reconstructed tracks", "All clusters"};
static const char* ClusterTitles[3] = {"Clusters Pt Distribution / Attachment", "Clusters Pt Distribution / Attachment (relative to all clusters)", "Clusters Pt Distribution / Attachment (integrated)"};
static const char* ClusterNamesShort[4] = {"Attached", "Fake", "FoundTracks", "All"};
static const char* ClusterTypes[3] = {"", "Ratio", "Integral"};
static int colorsHex[ColorCount] = {0xB03030, 0x00A000, 0x0000C0, 0x9400D3, 0x19BBBF, 0xF25900, 0x7F7F7F, 0xFFD700, 0x07F707, 0x07F7F7, 0xF08080, 0x000000};

static int ConfigDashedMarkers = 0;

static constexpr float kPi = M_PI;
static const float axes_min[5] = {-Y_MAX, -Z_MAX, 0.f, -ETA_MAX, PT_MIN};
static const float axes_max[5] = {Y_MAX, Z_MAX, 2.f *  kPi, ETA_MAX, PT_MAX};
static const int axis_bins[5] = {51, 51, 144, 31, 50};
static const int res_axis_bins[] = {1017, 113}; //Consecutive bin sizes, histograms are binned down until the maximum entry is 50, each bin size should evenly divide its predecessor.
static const float res_axes[5] = {1., 1., 0.03, 0.03, 1.0};
static const float pull_axis = 10.f;

static void SetAxisSize(TH1F* e)
{
	e->GetYaxis()->SetTitleOffset(1.0);
	e->GetYaxis()->SetTitleSize(0.045);
	e->GetYaxis()->SetLabelSize(0.045);
	e->GetXaxis()->SetTitleSize(0.045);
	e->GetXaxis()->SetLabelSize(0.045);
}

static void SetLegend(TLegend* l)
{
	l->SetTextFont(72);
	l->SetTextSize(0.016);
	l->SetFillColor(0);
}

static double* CreateLogAxis(int nbins, float xmin, float xmax)
{
	float logxmin = std::log10(xmin);
	float logxmax = std::log10(xmax);
	float binwidth = (logxmax-logxmin)/nbins;

	double *xbins =  new double[nbins+1];

	xbins[0] = xmin;
	for (int i=1;i<=nbins;i++)
	{
		xbins[i] = std::pow(10,logxmin+i*binwidth);
	}
	return xbins;
}

static void ChangePadTitleSize(TPad* p, float size)
{
	p->Update();
	TPaveText *pt = (TPaveText*)(p->GetPrimitive("title")); 
	if (pt == NULL)
	{
		printf("Error changing title\n");
	}
	else
	{
		pt->SetTextSize(size);
		p->Modified(); 
	}
}

void DrawHisto(TH1* histo, char* filename, char* options)
{
	TCanvas tmp;
	tmp.cd();
	histo->Draw(options);
	tmp.Print(filename);
}

int mcTrackMin = -1, mcTrackMax = -1;
void SetMCTrackRange(int min, int max)
{
	mcTrackMin = min;
	mcTrackMax = max;
}

void InitQA()
{
	char name[1024], fname[1024];

	for (int i = 0;i < ColorCount;i++)
	{
		float f1 = (float) ((colorsHex[i] >> 16) & 0xFF) / (float) 0xFF;
		float f2 = (float) ((colorsHex[i] >> 8) & 0xFF) / (float) 0xFF;
		float f3 = (float) ((colorsHex[i] >> 0) & 0xFF) / (float) 0xFF;
		TColor* c = new TColor(10000 + i, f1, f2, f3);
		colorNums[i] = c->GetNumber();
	}
	
	//Create Efficiency Histograms
	for (int i = 0;i < 4;i++)
	{
		for (int j = 0;j < 2;j++)
		{
			for (int k = 0;k < 2;k++)
			{
				for (int l = 0;l < 5;l++)
				{
					for (int m = 0;m < 2;m++)
					{
						sprintf(name, "%s%s%s%sVs%s", m ? "eff" : "tracks", EffTypes[i], FindableNames[j], PrimNames[k], VSParameterNames[l]);
						if (l == 4)
						{
							double* binsPt = CreateLogAxis(axis_bins[4], k == 0 ? PT_MIN_PRIM : axes_min[4], axes_max[4]);
							eff[i][j][k][l][m] = new TH1F(name, name, axis_bins[l], binsPt);
							delete[] binsPt;
						}
						else
						{
							eff[i][j][k][l][m] = new TH1F(name, name, axis_bins[l], axes_min[l], axes_max[l]);
						}
						eff[i][j][k][l][m]->Sumw2();
					}
				}
			}
		}
	}
	
	//Create Resolution Histograms
	for (int i = 0;i < 5;i++)
	{
		for (int j = 0;j < 5;j++)
		{
			sprintf(name, "rms_%s_vs_%s", VSParameterNames[i], VSParameterNames[j]);
			sprintf(fname, "mean_%s_vs_%s", VSParameterNames[i], VSParameterNames[j]);
			if (j == 4)
			{
				double* binsPt = CreateLogAxis(axis_bins[4], axes_min[4], axes_max[4]);
				res[i][j][0] = new TH1F(name, name, axis_bins[j], binsPt);
				res[i][j][1] = new TH1F(fname, fname, axis_bins[j], binsPt);
				delete[] binsPt;
			}
			else
			{
				res[i][j][0] = new TH1F(name, name, axis_bins[j], axes_min[j], axes_max[j]);
				res[i][j][1] = new TH1F(fname, fname, axis_bins[j], axes_min[j], axes_max[j]);
			}
			sprintf(name, "res_%s_vs_%s", VSParameterNames[i], VSParameterNames[j]);
			if (j == 4)
			{
				double* binsPt = CreateLogAxis(axis_bins[4], axes_min[4], axes_max[4]);
				res2[i][j] = new TH2F(name, name, res_axis_bins[0], -res_axes[i], res_axes[i], axis_bins[j], binsPt);
				delete[] binsPt;
			}
			else
			{
				res2[i][j] = new TH2F(name, name, res_axis_bins[0], -res_axes[i], res_axes[i], axis_bins[j], axes_min[j], axes_max[j]);
			}
		}
	}

	//Create Pull Histograms
	for (int i = 0;i < 5;i++)
	{
		for (int j = 0;j < 5;j++)
		{
			sprintf(name, "pull_rms_%s_vs_%s", VSParameterNames[i], VSParameterNames[j]);
			sprintf(fname, "pull_mean_%s_vs_%s", VSParameterNames[i], VSParameterNames[j]);
			if (j == 4)
			{
				double* binsPt = CreateLogAxis(axis_bins[4], axes_min[4], axes_max[4]);
				pull[i][j][0] = new TH1F(name, name, axis_bins[j], binsPt);
				pull[i][j][1] = new TH1F(fname, fname, axis_bins[j], binsPt);
				delete[] binsPt;
			}
			else
			{
				pull[i][j][0] = new TH1F(name, name, axis_bins[j], axes_min[j], axes_max[j]);
				pull[i][j][1] = new TH1F(fname, fname, axis_bins[j], axes_min[j], axes_max[j]);
			}
			sprintf(name, "pull_%s_vs_%s", VSParameterNames[i], VSParameterNames[j]);
			if (j == 4)
			{
				double* binsPt = CreateLogAxis(axis_bins[4], axes_min[4], axes_max[4]);
				pull2[i][j] = new TH2F(name, name, res_axis_bins[0], -pull_axis, pull_axis, axis_bins[j], binsPt);
				delete[] binsPt;
			}
			else
			{
				pull2[i][j] = new TH2F(name, name, res_axis_bins[0], -pull_axis, pull_axis, axis_bins[j], axes_min[j], axes_max[j]);
			}
		}
	}
	
	//Create Cluster Histograms
	for (int i = 0;i < 11;i++)
	{
		int ioffset = i >= 7 ? 7 : i >= 4 ? 4 : 0;
		int itype   = i >= 7 ? 2 : i >= 4 ? 1 : 0;
		sprintf(name, "clusters%s%s", ClusterNamesShort[i - ioffset], ClusterTypes[itype]);
		double* binsPt = CreateLogAxis(axis_bins[4], PT_MIN_CLUST, PT_MAX);
		clusters[i] = new TH1F(name, name, axis_bins[4], binsPt);
		delete[] binsPt;
	}
	{
		sprintf(name, "nclusters");
		ncl = new TH1F(name, name, 160, 0, 159);
	}
	
	//Create Tracks Histograms
	{
		sprintf(name, "tracks");
		double* binsPt = CreateLogAxis(axis_bins[4], PT_MIN_CLUST, PT_MAX);
		tracks = new TH1F(name, name, axis_bins[4], binsPt);
	}
	
	mkdir("plots", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

void RunQA()
{
	//Initialize Arrays
	AliHLTTPCCAStandaloneFramework &hlt = AliHLTTPCCAStandaloneFramework::Instance();
	const AliHLTTPCGMMerger &merger = hlt.Merger();
	trackMCLabels.resize(merger.NOutputTracks());
	trackMCLabelsReverse.resize(hlt.GetNMCInfo());
	recTracks.resize(hlt.GetNMCInfo());
	fakeTracks.resize(hlt.GetNMCInfo());
	mcParam.resize(hlt.GetNMCInfo());
	memset(recTracks.data(), 0, recTracks.size() * sizeof(recTracks[0]));
	memset(fakeTracks.data(), 0, fakeTracks.size() * sizeof(fakeTracks[0]));
	for (size_t i = 0;i < trackMCLabelsReverse.size();i++) trackMCLabelsReverse[i] = -1;
	clusterParam.resize(hlt.GetNMCLabels());
	memset(clusterParam.data(), 0, clusterParam.size() * sizeof(clusterParam[0]));
	totalFakes = 0;
	structConfigQA& config = configStandalone.configQA;	
	HighResTimer timer;

	if (hlt.GetNMCInfo() && hlt.GetNMCLabels())
	{
		//Assign Track MC Labels
		timer.Start();
		bool ompError = false;
#if DEBUG == 0
#pragma omp parallel for
#endif
		for (int i = 0; i < merger.NOutputTracks(); i++)
		{
			if (ompError) continue;
			int nClusters = 0;
			const AliHLTTPCGMMergedTrack &track = merger.OutputTracks()[i];
			std::vector<AliHLTTPCClusterMCWeight> labels;
			for (int k = 0;k < track.NClusters();k++)
			{
				if (merger.Clusters()[track.FirstClusterRef() + k].fState < 0) continue;
				nClusters++;
				int hitId = merger.Clusters()[track.FirstClusterRef() + k].fId;
				if (hitId >= hlt.GetNMCLabels()) {printf("Invalid hit id %d > %d\n", hitId, hlt.GetNMCLabels());ompError = true;break;}
				for (int j = 0;j < 3;j++)
				{
					if (hlt.GetMCLabels()[hitId].fClusterID[j].fMCID >= hlt.GetNMCInfo()) {printf("Invalid label %d > %d\n", hlt.GetMCLabels()[hitId].fClusterID[j].fMCID, hlt.GetNMCInfo());ompError = true;break;}
					if (hlt.GetMCLabels()[hitId].fClusterID[j].fMCID >= 0)
					{
						if (DEBUG >= 3 && track.OK()) printf("Track %d Cluster %d Label %d: %d (%f)\n", i, k, j, hlt.GetMCLabels()[hitId].fClusterID[j].fMCID, hlt.GetMCLabels()[hitId].fClusterID[j].fWeight);
						labels.push_back(hlt.GetMCLabels()[hitId].fClusterID[j]);
					}
				}
				if (ompError) break;
			}
			if (ompError) continue;
			if (labels.size() == 0)
			{
				trackMCLabels[i] = -1e9;
				totalFakes++;
				continue;
			}
			std::sort(labels.data(), labels.data() + labels.size(), MCComp);
			
			AliHLTTPCClusterMCWeight maxLabel;
			AliHLTTPCClusterMCWeight cur = labels[0];
			float sumweight = 0.f;
			int curcount = 1, maxcount = 0;
			if (DEBUG >= 2 && track.OK()) for (unsigned int k = 0;k < labels.size();k++) printf("\t%d %f\n", labels[k].fMCID, labels[k].fWeight);
			for (unsigned int k = 1;k <= labels.size();k++)
			{
				if (k == labels.size() || labels[k].fMCID != cur.fMCID)
				{
					sumweight += cur.fWeight;
					if (curcount > maxcount)
					{
						maxLabel = cur;
						maxcount = curcount;
					}
					if (k < labels.size())
					{
						cur = labels[k];
						curcount = 1;
					}
				}
				else
				{
					cur.fWeight += labels[k].fWeight;
					curcount++;
				}
			}
			
			if (maxcount < config.recThreshold * nClusters) maxLabel.fMCID = -2 - maxLabel.fMCID;
			trackMCLabels[i] = maxLabel.fMCID;
			if (DEBUG && track.OK() && hlt.GetNMCInfo() > maxLabel.fMCID)
			{
				const AliHLTTPCCAMCInfo& mc = hlt.GetMCInfo()[maxLabel.fMCID >= 0 ? maxLabel.fMCID : (-maxLabel.fMCID - 2)];
				printf("Track %d label %d weight %f clusters %d (fitted %d) (%f%% %f%%) Pt %f\n", i, maxLabel.fMCID >= 0 ? maxLabel.fMCID : (maxLabel.fMCID + 2), maxLabel.fWeight, nClusters, track.NClustersFitted(), maxLabel.fWeight / sumweight, (float) maxcount / (float) nClusters, sqrt(mc.fPx * mc.fPx + mc.fPy * mc.fPy));
			}
		}
		if (ompError) return;
		for (int i = 0; i < merger.NOutputTracks(); i++)
		{
			if (trackMCLabels[i] == -1e9) continue;
			const AliHLTTPCGMMergedTrack &track = merger.OutputTracks()[i];
			if (!track.OK()) continue;
			int label = trackMCLabels[i] < 0 ? (-trackMCLabels[i] - 2) : trackMCLabels[i];
			for (int k = 0;k < track.NClusters();k++)
			{
				if (merger.Clusters()[track.FirstClusterRef() + k].fState < 0) continue;
				int hitId = merger.Clusters()[track.FirstClusterRef() + k].fId;
				bool correct = false;
				for (int j = 0;j < 3;j++) if (hlt.GetMCLabels()[hitId].fClusterID[j].fMCID == label) {correct=true;break;}
				if (correct) clusterParam[hitId].attached++;
				else clusterParam[hitId].fakeAttached++;
			}
			if (trackMCLabels[i] < 0)
			{
				fakeTracks[label]++;
			}
			else
			{
				recTracks[label]++;
				
				int& revLabel = trackMCLabelsReverse[label];
				if (revLabel == -1 ||
					!merger.OutputTracks()[revLabel].OK() ||
					(merger.OutputTracks()[i].OK() && fabs(merger.OutputTracks()[i].GetParam().GetZ()) < fabs(merger.OutputTracks()[revLabel].GetParam().GetZ())))
				{
					revLabel = i;
				}
			}
		}
		
		if (TIMING) printf("QA Time: Assign Track Labels:\t\t%6.0f us\n", timer.GetCurrentElapsedTime() * 1e6);
		timer.ResetStart();
		
		//Recompute fNWeightCls (might have changed after merging events into timeframes)
		for (int i = 0;i < hlt.GetNMCInfo();i++) mcParam[i].nWeightCls = 0.;
		for (int i = 0;i < hlt.GetNMCLabels();i++)
		{
			const AliHLTTPCClusterMCLabel* labels = hlt.GetMCLabels();
			float weightTotal = 0.f;
			for (int j = 0;j < 3;j++) if (labels[i].fClusterID[j].fMCID >= 0) weightTotal += labels[i].fClusterID[j].fWeight;
			for (int j = 0;j < 3;j++) if (labels[i].fClusterID[j].fMCID >= 0)
			{
				mcParam[labels[i].fClusterID[j].fMCID].nWeightCls += labels[i].fClusterID[j].fWeight / weightTotal;
			}
		}
		if (TIMING) printf("QA Time: Compute cluster label weights:\t%6.0f us\n", timer.GetCurrentElapsedTime() * 1e6);
		timer.ResetStart();
		
#pragma omp parallel for
		for (int i = 0;i < hlt.GetNMCInfo();i++)
		{
			const AliHLTTPCCAMCInfo& info = hlt.GetMCInfo()[i];
			additionalMCParameters& mc2 = mcParam[i];
			mc2.pt = std::sqrt(info.fPx * info.fPx + info.fPy * info.fPy);
			mc2.phi = kPi + std::atan2(-info.fPy,-info.fPx);
			float p = info.fPx*info.fPx+info.fPy*info.fPy+info.fPz*info.fPz;
			if (p < 1e-18)
			{
				mc2.theta = mc2.eta = 0.f;
			}
			else
			{
				mc2.theta = info.fPz == 0 ? (kPi / 2) : (std::acos(info.fPz / std::sqrt(p)));
				mc2.eta = -std::log(std::tan(0.5 * mc2.theta));
			}
		}
		
		//Compute MC Track Parameters for MC Tracks
		//Fill Efficiency Histograms
		for (int i = 0;i < hlt.GetNMCInfo();i++)
		{
			if ((mcTrackMin != -1 && i < mcTrackMin) || (mcTrackMax != -1 && i >= mcTrackMax)) continue;
			const AliHLTTPCCAMCInfo& info = hlt.GetMCInfo()[i];
			const additionalMCParameters& mc2 = mcParam[i];
			if (mc2.nWeightCls == 0.f) continue;
			const float& mcpt = mc2.pt;
			const float& mcphi = mc2.phi;
			const float& mceta = mc2.eta;

			if (info.fPrim && info.fPrimDaughters) continue;
			if (mc2.nWeightCls < MIN_WEIGHT_CLS) continue;
			int findable = mc2.nWeightCls >= FINDABLE_WEIGHT_CLS;
			if (info.fPID < 0) continue;
			if (info.fCharge == 0.f) continue;
			
			if (fabs(mceta) > ETA_MAX || mcpt < PT_MIN || mcpt > PT_MAX) continue;
			
			float alpha = std::atan2(info.fY, info.fX);
			alpha /= kPi / 9.f;
			alpha = std::floor(alpha);
			alpha *= kPi / 9.f;
			alpha += kPi / 18.f;
			
			float c = std::cos(alpha);
			float s = std::sin(alpha);
			float localY = -info.fX * s + info.fY * c;
			
			for (int j = 0;j < 4;j++)
			{
				for (int k = 0;k < 2;k++)
				{
					if (k == 0 && findable == 0) continue;
					
					int val = (j == 0) ? (recTracks[i] ? 1 : 0) :
						(j == 1) ? (recTracks[i] ? recTracks[i] - 1 : 0) :
						(j == 2) ? fakeTracks[i] :
						1;
					
					for (int l = 0;l < 5;l++)
					{
						if (info.fPrim && mcpt < PT_MIN_PRIM) continue;
						if (l != 3 && fabs(mceta) > ETA_MAX2) continue;
						if (l < 4 && mcpt < 1.f / config.qpt) continue;
						
						float pos = l == 0 ? localY : l == 1 ? info.fZ : l == 2 ? mcphi : l == 3 ? mceta : mcpt;

						eff[j][k][!info.fPrim][l][0]->Fill(pos, val);
					}
				}
			}
		}
		if (TIMING) printf("QA Time: Fill efficiency histograms:\t%6.0f us\n", timer.GetCurrentElapsedTime() * 1e6);
		timer.ResetStart();

		//Fill Resolution Histograms
		AliHLTTPCGMPropagator prop;
		const float kRho = 1.025e-3;//0.9e-3;
		const float kRadLen = 29.532;//28.94;
		prop.SetMaxSinPhi( .999 );
		prop.SetMaterial( kRadLen, kRho );
		prop.SetPolynomialField( merger.pField() );		
		prop.SetUseMeanMomentum(kFALSE );
		prop.SetContinuousTracking( kFALSE );
		prop.SetHomemadeEvents( merger.SliceParam().HomemadeEventsFlag());

		for (int i = 0; i < merger.NOutputTracks(); i++)
		{
			if (trackMCLabels[i] < 0) continue;
			const AliHLTTPCCAMCInfo& mc1 = hlt.GetMCInfo()[trackMCLabels[i]];
			const additionalMCParameters& mc2 = mcParam[trackMCLabels[i]];
			const AliHLTTPCGMMergedTrack& track = merger.OutputTracks()[i];
			
			if ((mcTrackMin != -1 && trackMCLabels[i] < mcTrackMin) || (mcTrackMax != -1 && trackMCLabels[i] >= mcTrackMax)) continue;
			
			if (!track.OK()) continue;
			if (fabs(mc2.eta) > ETA_MAX || mc2.pt < PT_MIN || mc2.pt > PT_MAX) continue;
			if (mc1.fCharge == 0.f) continue;
			if (mc1.fPID < 0) continue;
			if (mc2.nWeightCls < MIN_WEIGHT_CLS) continue;
			if (config.resPrimaries == 1 && (!mc1.fPrim || mc1.fPrimDaughters)) continue;
			else if (config.resPrimaries == 2 && (mc1.fPrim || mc1.fPrimDaughters)) continue;
			
			float mclocal[4]; //Rotated x,y,Px,Py mc-coordinates - the MC data should be rotated since the track is propagated best along x
			float c = std::cos(track.GetAlpha());
			float s = std::sin(track.GetAlpha());
			float x = mc1.fX;
			float y = mc1.fY;
			mclocal[0] = x*c + y*s;
			mclocal[1] =-x*s + y*c;
			float px = mc1.fPx;
			float py = mc1.fPy;
			mclocal[2] = px*c + py*s;
			mclocal[3] =-px*s + py*c;
			
			AliHLTTPCGMTrackParam param = track.GetParam();
			
			if (mclocal[0] < 80) continue;
			if (mclocal[0] > param.GetX() + 20) continue;
			if (param.GetX() > config.maxResX) continue;

			float alpha = track.GetAlpha();		
			prop.SetTrack(&param, alpha);	
			bool inFlyDirection = 0;
			if (config.strict && (param.X() - mclocal[0]) * (param.X() - mclocal[0]) + (param.Y() - mclocal[1]) * (param.Y() - mclocal[1]) + (param.Z() + param.ZOffset() - mc1.fZ) * (param.Z() + param.ZOffset() - mc1.fZ) > 25) continue;
			
			if (prop.PropagateToXAlpha( mclocal[0], alpha, inFlyDirection ) ) continue;
			if (fabs(param.Y() - mclocal[1]) > (config.strict ? 1.f : 4.f) || fabs(param.Z() + param.ZOffset() - mc1.fZ) > (config.strict ? 1.f : 4.f)) continue;
			
			float deltaY = param.GetY() - mclocal[1];
			float deltaZ = param.GetZ() + param.ZOffset() - mc1.fZ;
			float deltaPhiNative = param.GetSinPhi() - mclocal[3] / mc2.pt;
			float deltaPhi = std::asin(param.GetSinPhi()) - std::atan2(mclocal[3], mclocal[2]);
			float deltaLambdaNative = param.GetDzDs() - mc1.fPz / mc2.pt;
			float deltaLambda = std::atan(param.GetDzDs()) - std::atan2(mc1.fPz, mc2.pt);
			float deltaPtNative = param.GetQPt() - (mc1.fCharge / 3.f) / mc2.pt;
			float deltaPt = (fabs(1.f / param.GetQPt()) - mc2.pt) / mc2.pt;
			
			float paramval[5] = {mclocal[1], mc1.fZ, mc2.phi, mc2.eta, mc2.pt};
			float resval[5] = {deltaY, deltaZ, config.nativeFitResolutions ? deltaPhiNative : deltaPhi, config.nativeFitResolutions ? deltaLambdaNative : deltaLambda, config.nativeFitResolutions ? deltaPtNative : deltaPt};
			float pullval[5] = {deltaY / std::sqrt(param.GetErr2Y()), deltaZ / std::sqrt(param.GetErr2Z()), deltaPhiNative / std::sqrt(param.GetErr2SinPhi()), deltaLambdaNative / std::sqrt(param.GetErr2DzDs()), deltaPtNative / std::sqrt(param.GetErr2QPt())};
			
			for (int j = 0;j < 5;j++)
			{
				for (int k = 0;k < 5;k++)
				{
					if (k != 3 && fabs(mc2.eta) > ETA_MAX2) continue;
					if (k < 4 && mc2.pt < 1.f / config.qpt) continue;
					res2[j][k]->Fill(resval[j], paramval[k]);
					pull2[j][k]->Fill(pullval[j], paramval[k]);
				}
			}
		}
		if (TIMING) printf("QA Time: Fill resolution histograms:\t%6.0f us\n", timer.GetCurrentElapsedTime() * 1e6);
		timer.ResetStart();
		
		//Fill Cluster Histograms
		for (int i = 0;i < hlt.GetNMCInfo();i++)
		{
			const additionalMCParameters& mc2 = mcParam[i];
			
			float pt = mc2.pt < PT_MIN_CLUST ? PT_MIN_CLUST : mc2.pt;
			clusters[3]->Fill(pt, mc2.nWeightCls);
			if (recTracks[i] || fakeTracks[i]) clusters[2]->Fill(pt, mc2.nWeightCls);
		}
	 	for (int i = 0;i < hlt.GetNMCLabels();i++)
	 	{
			float totalAttached = clusterParam[i].attached + clusterParam[i].fakeAttached;
			if (totalAttached <= 0) continue;
			float totalWeight = 0.;
			for (int j = 0;j < 3;j++) if (hlt.GetMCLabels()[i].fClusterID[j].fMCID >= 0) totalWeight += hlt.GetMCLabels()[i].fClusterID[j].fWeight;
			if (totalWeight > 0)
			{
				for (int j = 0;j < 3;j++)
				{
					if (hlt.GetMCLabels()[i].fClusterID[j].fMCID >= 0)
					{
						float pt = mcParam[hlt.GetMCLabels()[i].fClusterID[j].fMCID].pt;
						if (pt < PT_MIN_CLUST) pt = PT_MIN_CLUST;
						clusters[0]->Fill(pt, clusterParam[i].attached / totalAttached * hlt.GetMCLabels()[i].fClusterID[j].fWeight / totalWeight);
						clusters[1]->Fill(pt, clusterParam[i].fakeAttached / totalAttached * hlt.GetMCLabels()[i].fClusterID[j].fWeight / totalWeight);
					}
				}
			}
		}
		if (TIMING) printf("QA Time: Fill cluster histograms:\t%6.0f us\n", timer.GetCurrentElapsedTime() * 1e6);
		timer.ResetStart();
	}
	else if (!config.inputHistogramsOnly)
	{
		printf("No MC information available, cannot run QA!\n");
	}
	
	//Fill other histograms
	for (int i = 0;i < merger.NOutputTracks(); i++)
	{
		const AliHLTTPCGMMergedTrack &track = merger.OutputTracks()[i];
		if (!track.OK()) continue;
		tracks->Fill(1.f / fabs(track.GetParam().GetQPt()));
		ncl->Fill(track.NClustersFitted());
	}
	if (TIMING) printf("QA Time: Others:\t%6.0f us\n", timer.GetCurrentElapsedTime() * 1e6);
	
	//Create CSV DumpTrackHits
	if (config.csvDump)
	{
		int totalNCls = hlt.GetNMCLabels();
		if (totalNCls == 0) for (int iSlice = 0; iSlice < 36; iSlice++) totalNCls += hlt.ClusterData(iSlice).NumberOfClusters();
			
		std::vector<float> clusterInfo(totalNCls);
		memset(clusterInfo.data(), 0, clusterInfo.size() * sizeof(clusterInfo[0]));
		for (int i = 0; i < merger.NOutputTracks(); i++)
		{
			const AliHLTTPCGMMergedTrack &track = merger.OutputTracks()[i];
			if (!track.OK()) continue;
			for (int k = 0;k < track.NClusters();k++)
			{
				if (merger.Clusters()[track.FirstClusterRef() + k].fState < 0) continue;
				int hitId = merger.Clusters()[track.FirstClusterRef() + k].fId;
				float pt = fabs(1.f/track.GetParam().GetQPt());
				if (pt > clusterInfo[hitId]) clusterInfo[hitId] = pt;
			}
		}
		static int csvNum = 0;
		char fname[256];
		sprintf(fname, "dump.%d.csv", csvNum);
		FILE* fp = fopen(fname, "w+");
		fprintf(fp, "x;y;z;reconstructedPt;individualMomentum;individualTransverseMomentum;trackLabel1;trackLabel2;trackLabel3\n\n");
		int clustersAttached = 0;
		for (int iSlice = 0; iSlice < 36; iSlice++)
		{
			const AliHLTTPCCAClusterData &cdata = hlt.ClusterData(iSlice);
			for (int i = 0; i < cdata.NumberOfClusters(); i++)
			{
				const int cid = cdata.Id(i);
				float x, y, z;
				merger.SliceParam().Slice2Global(cdata.X(i), cdata.Y(i), cdata.Z(i), &x, &y, &z);
				float totalWeight = 0.f;
				if (hlt.GetNMCInfo() && hlt.GetNMCLabels())
					for (int j = 0;j < 3;j++)
						if (hlt.GetMCLabels()[cid].fClusterID[j].fMCID >= 0)
							totalWeight += hlt.GetMCLabels()[cid].fClusterID[j].fWeight;
				
				float maxPt = 0.;
				float p = 0.;

				if (totalWeight > 0)
				{
					for (int j = 0;j < 3;j++)
					{
						const AliHLTTPCClusterMCWeight label = hlt.GetMCLabels()[cid].fClusterID[j];
						if (label.fMCID >= 0 && label.fWeight > 0.1 * totalWeight)
						{
							const AliHLTTPCCAMCInfo& info = hlt.GetMCInfo()[label.fMCID];
							const additionalMCParameters& mc2 = mcParam[label.fMCID];
							const float pt = fabs(mc2.pt);
							if (pt > maxPt)
							{
								maxPt = pt;
								p = std::sqrt(info.fPx * info.fPx + info.fPy * info.fPy + info.fPz * info.fPz);
							}
						}
					}
				}
				if (clusterInfo[cid] > 0.f) clustersAttached++;
				int labels[3] = {};
				if (hlt.GetNMCInfo() && hlt.GetNMCLabels())
					for (int i = 0;i < 3;i++)
						labels[i] = hlt.GetMCLabels()[cid].fClusterID[i].fMCID;
				fprintf(fp, "%f;%f;%f;%f;%f;%f;%d;%d;%d\n", x, y, z, clusterInfo[cid], p, maxPt, labels[0], labels[1], labels[2]);
			}
		}
		fclose(fp);
		if (hlt.GetNMCInfo() && hlt.GetNMCLabels())
		{
			sprintf(fname, "dump_event.%d.csv", csvNum++);
			fp = fopen(fname, "w+");
			fprintf(fp, "trackLabel;trackMomentum;trackMomentumTransverse;trackMomentumZ\n\n");
			for (int i = 0;i < hlt.GetNMCInfo();i++)
			{
				const AliHLTTPCCAMCInfo& info = hlt.GetMCInfo()[i];
				additionalMCParameters& mc2 = mcParam[i];
				if (mc2.nWeightCls > 0) fprintf(fp, "%d;%f;%f;%f\n", i, sqrt(info.fPx * info.fPx + info.fPy * info.fPy + info.fPz * info.fPz), mc2.pt, info.fPz);
			}
			fclose(fp);
		}
		printf("Wrote %s, %d out of %lu clusters attached (%6.2f%%)\n", fname, clustersAttached, clusterInfo.size(), 100.f * clustersAttached / clusterInfo.size());
	} 
}

void GetName(char* fname, int k)
{
	const structConfigQA& config = configStandalone.configQA;
	const int nNewInput = config.inputHistogramsOnly ? 0 : 1;
	if (k || config.inputHistogramsOnly || config.name) sprintf(fname, "%s - ", config.inputHistogramsOnly || k ? (config.compareInputNames.size() > (unsigned) (k - nNewInput) ? config.compareInputNames[k - nNewInput] : config.compareInputs[k - nNewInput]) : config.name);
	else fname[0] = 0;
}

template <class T> T* GetHist(T* &ee, std::vector<TFile*>& tin, int k, int nNewInput)
{
	const structConfigQA& config = configStandalone.configQA;
	T* e = ee;
	if ((config.inputHistogramsOnly || k) && (e = dynamic_cast<T*>(tin[k - nNewInput]->Get(e->GetName()))) == NULL)
	{
		printf("Missing histogram in input %s: %s\n", config.compareInputs[k - nNewInput], ee->GetName());
		return(NULL);
	}
	ee = e;
	return(e);
}

int DrawQAHistograms()
{
	char name[1024], fname[1024];
	
	const structConfigQA& config = configStandalone.configQA;
	const int nNewInput = config.inputHistogramsOnly ? 0 : 1;
	const int ConfigNumInputs = nNewInput + config.compareInputs.size();
	
	std::vector<TFile*> tin(config.compareInputs.size());
	for (unsigned int i = 0;i < config.compareInputs.size();i++)
	{
		tin[i] = new TFile(config.compareInputs[i]);
	}
	TFile* tout = NULL;
	if (config.output) tout = new TFile(config.output, "RECREATE");
	
	float legendSpacingString = 0.025;
	for (int i = 0;i < ConfigNumInputs;i++)
	{
		GetName(fname, i);
		if (strlen(fname) * 0.006 > legendSpacingString) legendSpacingString = strlen(fname) * 0.006;
	}

	//Create Canvas / Pads for Efficiency Histograms
	for (int ii = 0;ii < 6;ii++)
	{
		int i = ii == 5 ? 4 : ii;
		sprintf(fname, "ceff_%d", ii);
		sprintf(name, "Efficiency versus %s", VSParameterNames[i]);
		ceff[ii] = new TCanvas(fname,name,0,0,700,700.*2./3.);
		ceff[ii]->cd();
		float dy = 1. / 2.;
		peff[ii][0] = new TPad( "p0","",0.0,dy*0,0.5,dy*1); peff[ii][0]->Draw();peff[ii][0]->SetRightMargin(0.04);
		peff[ii][1] = new TPad( "p1","",0.5,dy*0,1.0,dy*1); peff[ii][1]->Draw();peff[ii][1]->SetRightMargin(0.04);
		peff[ii][2] = new TPad( "p2","",0.0,dy*1,0.5,dy*2-.001); peff[ii][2]->Draw();peff[ii][2]->SetRightMargin(0.04);
		peff[ii][3] = new TPad( "p3","",0.5,dy*1,1.0,dy*2-.001); peff[ii][3]->Draw();peff[ii][3]->SetRightMargin(0.04);
		legendeff[ii] = new TLegend(0.92 - legendSpacingString * 1.45, 0.83 - (0.93 - 0.83) / 2. * (float) ConfigNumInputs,0.98,0.849); SetLegend(legendeff[ii]);
	}

	//Create Canvas / Pads for Resolution Histograms
	for (int ii = 0;ii < 7;ii++)
	{
		int i = ii == 5 ? 4 : ii;
		sprintf(fname, "cres_%d", ii);
		if (ii == 6) sprintf(name, "Integral Resolution");
		else sprintf(name, "Resolution versus %s", VSParameterNames[i]);
		cres[ii] = new TCanvas(fname,name,0,0,700,700.*2./3.);
		cres[ii]->cd();
		gStyle->SetOptFit(1);

		float dy = 1. / 2.;
		pres[ii][3] = new TPad( "p0","",0.0,dy*0,0.5,dy*1); pres[ii][3]->Draw();pres[ii][3]->SetRightMargin(0.04);
		pres[ii][4] = new TPad( "p1","",0.5,dy*0,1.0,dy*1); pres[ii][4]->Draw();pres[ii][4]->SetRightMargin(0.04);
		pres[ii][0] = new TPad( "p2","",0.0,dy*1,1./3.,dy*2-.001); pres[ii][0]->Draw();pres[ii][0]->SetRightMargin(0.04);pres[ii][0]->SetLeftMargin(0.15);
		pres[ii][1] = new TPad( "p3","",1./3.,dy*1,2./3.,dy*2-.001); pres[ii][1]->Draw();pres[ii][1]->SetRightMargin(0.04);pres[ii][1]->SetLeftMargin(0.135);
		pres[ii][2] = new TPad( "p4","",2./3.,dy*1,1.0,dy*2-.001); pres[ii][2]->Draw();pres[ii][2]->SetRightMargin(0.06);pres[ii][2]->SetLeftMargin(0.135);
		if (ii < 6) {legendres[ii] = new TLegend(0.9 - legendSpacingString * 1.45, 0.93 - (0.93 - 0.87) / 2. * (float) ConfigNumInputs, 0.98, 0.949); SetLegend(legendres[ii]);}
	}
	
	//Create Canvas / Pads for Pull Histograms
	for (int ii = 0;ii < 7;ii++)
	{
		int i = ii == 5 ? 4 : ii;
		sprintf(fname, "cpull_%d", ii);
		if (ii == 6) sprintf(name, "Integral Pull");
		else sprintf(name, "Pull versus %s", VSParameterNames[i]);
		cpull[ii] = new TCanvas(fname,name,0,0,700,700.*2./3.);
		cpull[ii]->cd();
		gStyle->SetOptFit(1);

		float dy = 1. / 2.;
		ppull[ii][3] = new TPad( "p0","",0.0,dy*0,0.5,dy*1); ppull[ii][3]->Draw();ppull[ii][3]->SetRightMargin(0.04);
		ppull[ii][4] = new TPad( "p1","",0.5,dy*0,1.0,dy*1); ppull[ii][4]->Draw();ppull[ii][4]->SetRightMargin(0.04);
		ppull[ii][0] = new TPad( "p2","",0.0,dy*1,1./3.,dy*2-.001); ppull[ii][0]->Draw();ppull[ii][0]->SetRightMargin(0.04);ppull[ii][0]->SetLeftMargin(0.15);
		ppull[ii][1] = new TPad( "p3","",1./3.,dy*1,2./3.,dy*2-.001); ppull[ii][1]->Draw();ppull[ii][1]->SetRightMargin(0.04);ppull[ii][1]->SetLeftMargin(0.135);
		ppull[ii][2] = new TPad( "p4","",2./3.,dy*1,1.0,dy*2-.001); ppull[ii][2]->Draw();ppull[ii][2]->SetRightMargin(0.06);ppull[ii][2]->SetLeftMargin(0.135);
		if (ii < 6) {legendpull[ii] = new TLegend(0.9 - legendSpacingString * 1.45, 0.93 - (0.93 - 0.87) / 2. * (float) ConfigNumInputs, 0.98, 0.949); SetLegend(legendpull[ii]);}
	}

	//Create Canvas for Cluster Histos
	for (int i = 0;i < 3;i++)
	{
		sprintf(fname, "cclust_%d", i);
		cclust[i] = new TCanvas(fname,ClusterTitles[i],0,0,700,700.*2./3.); cclust[i]->cd();
		pclust[i] = new TPad( "p0","",0.0,0.0,1.0,1.0); pclust[i]->Draw();
		float y1 = i != 1 ? 0.83 : 0.48, y2 = i != 1 ? 0.9 : 0.55;
		legendclust[i] = new TLegend(i == 2 ? 0.1 : 0.65, y2 - (y2 - y1) * ConfigNumInputs, i == 2 ? 0.35 : 0.9, y2);SetLegend(legendclust[i]);
	}

	//Create Canvas for other histos
	{
		ctracks = new TCanvas("ctracks","Track Pt",0,0,700,700.*2./3.);	ctracks->cd();
		ptracks = new TPad( "p0","",0.0,0.0,1.0,1.0); ptracks->Draw();
		legendtracks = new TLegend(0.9 - legendSpacingString * 1.45, 0.93 - (0.93 - 0.87) / 2. * (float) ConfigNumInputs, 0.98, 0.949); SetLegend(legendtracks);

		cncl = new TCanvas("cncl","Number of clusters per track",0,0,700,700.*2./3.);	cncl->cd();
		pncl = new TPad( "p0","",0.0,0.0,1.0,1.0); pncl->Draw();
		legendncl = new TLegend(0.9 - legendSpacingString * 1.45, 0.93 - (0.93 - 0.87) / 2. * (float) ConfigNumInputs, 0.98, 0.949); SetLegend(legendncl);
	}
	
	if (!config.inputHistogramsOnly) printf("QA Stats: Eff: Tracks Prim %d (Eta %d, Pt %d) %f%% (%f%%) Sec %d (Eta %d, Pt %d) %f%% (%f%%) -  Res: Tracks %d (Eta %d, Pt %d)\n",
		(int) eff[3][1][0][0][0]->GetEntries(), (int) eff[3][1][0][3][0]->GetEntries(), (int) eff[3][1][0][4][0]->GetEntries(),
		eff[0][0][0][0][0]->GetSumOfWeights() / std::max(1., eff[3][0][0][0][0]->GetSumOfWeights()), eff[0][1][0][0][0]->GetSumOfWeights() / std::max(1., eff[3][1][0][0][0]->GetSumOfWeights()),
		(int) eff[3][1][1][0][0]->GetEntries(), (int) eff[3][1][1][3][0]->GetEntries(), (int) eff[3][1][1][4][0]->GetEntries(),
		eff[0][0][1][0][0]->GetSumOfWeights() / std::max(1., eff[3][0][1][0][0]->GetSumOfWeights()), eff[0][1][1][0][0]->GetSumOfWeights() / std::max(1., eff[3][1][1][0][0]->GetSumOfWeights()),
		(int) res2[0][0]->GetEntries(), (int) res2[0][3]->GetEntries(), (int) res2[0][4]->GetEntries()
	);
	
	//Process / Draw Efficiency Histograms
	for (int ii = 0;ii < 6;ii++)
	{
		int i = ii == 5 ? 4 : ii;
		for (int k = 0;k < ConfigNumInputs;k++)
		{
			for (int j = 0;j < 4;j++)
			{
				peff[ii][j]->cd();
				for (int l = 0;l < 3;l++)
				{
					if (k == 0 && config.inputHistogramsOnly == 0 && ii != 5)
					{
						if (l == 0)
						{
							//Divide eff, compute all for fake/clone
							eff[0][j / 2][j % 2][i][1]->Divide(eff[l][j / 2][j % 2][i][0], eff[3][j / 2][j % 2][i][0], 1, 1, "B");
							eff[3][j / 2][j % 2][i][1]->Reset(); //Sum up rec + clone + fake for clone/fake rate
							eff[3][j / 2][j % 2][i][1]->Add(eff[0][j / 2][j % 2][i][0]);
							eff[3][j / 2][j % 2][i][1]->Add(eff[1][j / 2][j % 2][i][0]);
							eff[3][j / 2][j % 2][i][1]->Add(eff[2][j / 2][j % 2][i][0]);
						}
						else
						{
							//Divide fake/clone
							eff[l][j / 2][j % 2][i][1]->Divide(eff[l][j / 2][j % 2][i][0], eff[3][j / 2][j % 2][i][1], 1, 1, "B");
						}
					}
				
					TH1F* e = eff[l][j / 2][j % 2][i][1];
						
					e->SetStats(kFALSE);
					e->SetMaximum(1.02);
					e->SetMinimum(-0.02);
					if (!config.inputHistogramsOnly && k == 0)
					{
						e->SetTitle(EfficiencyTitles[j]);
						e->GetYaxis()->SetTitle("(Efficiency)");
						e->GetXaxis()->SetTitle(XAxisTitles[i]);
						if (tout)
						{
							eff[l][j / 2][j % 2][i][0]->Write();
							e->Write();
							if (l == 2) eff[3][j / 2][j % 2][i][0]->Write(); //Store also all histogram!
						}
					}
					else if (GetHist(e, tin, k, nNewInput) == NULL) continue;

					e->SetMarkerColor(kBlack);
					e->SetLineWidth(1);
					e->SetLineColor(colorNums[(l == 2 ? (ConfigNumInputs * 2 + k) : (k * 2 + l)) % ColorCount]);
					e->SetLineStyle(ConfigDashedMarkers ? k + 1 : 1);
					SetAxisSize(e);
					e->Draw(k || l ? "same" : "");
					if (j == 0)
					{
						GetName(fname, k);
						sprintf(name, "%s%s", fname, EffNames[l]);
						legendeff[ii]->AddEntry(e, name, "l");
					}
					if (ii == 5) peff[ii][j]->SetLogx();
				}
				ceff[ii]->cd();
				ChangePadTitleSize(peff[ii][j], 0.056);
			}
		}
		legendeff[ii]->Draw();
		sprintf(fname, "plots/eff_vs_%s.pdf", VSParameterNames[ii]);
		ceff[ii]->Print(fname);
	}

	//Process / Draw Resolution Histograms
	TH1D *resIntegral[5] = {}, *pullIntegral[5] = {};
	for (int p = 0;p < 2;p++)
	{
		for (int ii = 0;ii < 6;ii++)
		{
			TCanvas* can = p ? cpull[ii] : cres[ii];
			TLegend* leg = p ? legendpull[ii] : legendres[ii];
			int i = ii == 5 ? 4 : ii;
			for (int j = 0;j < 5;j++)
			{
				TH2F* src = p ? pull2[j][i] : res2[j][i];
				TH1F** dst = p ? pull[j][i] : res[j][i];
				TH1D* &dstIntegral = p ? pullIntegral[j] : resIntegral[j];
				TPad* pad = p ? ppull[ii][j] : pres[ii][j];

				if (!config.inputHistogramsOnly && ii != 5)
				{
					TCanvas cfit;
					cfit.cd();
					
					TAxis* axis = src->GetYaxis();
					int nBins = axis->GetNbins();
					int integ = 1;
					for (int bin = 1;bin <= nBins;bin++)
					{
						int bin0 = std::max(bin - integ, 0);
						int bin1 = std::min(bin + integ, nBins);
						TH1D* proj = src->ProjectionX("proj", bin0, bin1);
						if (proj->GetEntries())
						{
							unsigned int rebin = 1;
							while (proj->GetMaximum() < 50 && rebin < sizeof(res_axis_bins) / sizeof(res_axis_bins[0]))
							{
								proj->Rebin(res_axis_bins[rebin - 1]/res_axis_bins[rebin]);
								rebin++;
							}
							
							if (proj->GetEntries() < 20 || proj->GetRMS() < 0.00001)
							{
								dst[0]->SetBinContent(bin, proj->GetRMS());
								dst[0]->SetBinError(bin, sqrt(proj->GetRMS()));
								dst[1]->SetBinContent(bin, proj->GetMean());
								dst[1]->SetBinError(bin, sqrt(proj->GetRMS()));
							}
							else
							{
								proj->Fit("gaus", proj->GetMaximum() < 20 ? "sQl" : "sQ");
								TF1* fitFunc = proj->GetFunction("gaus");
								float sigma = fabs(fitFunc->GetParameter(2));
								if (sigma > 0.f)
								{
									dst[0]->SetBinContent(bin, fabs(fitFunc->GetParameter(2)));
									dst[1]->SetBinContent(bin, fitFunc->GetParameter(1));
								}
								else
								{
									dst[0]->SetBinContent(bin, 0);
									dst[1]->SetBinContent(bin, 0);
								}
								dst[0]->SetBinError(bin, fitFunc->GetParError(2));
								dst[1]->SetBinError(bin, fitFunc->GetParError(1));

								bool fail = fabs(proj->GetMean() - dst[1]->GetBinContent(bin)) > res_axes[j] || dst[0]->GetBinError(bin) > 1 || dst[1]->GetBinError(bin) > 1;
								if (fail)
								{
									dst[0]->SetBinContent(bin, proj->GetMean());
									dst[1]->SetBinContent(bin, proj->GetRMS());
									dst[0]->SetBinError(bin, sqrt(proj->GetRMS()));
									dst[1]->SetBinError(bin, sqrt(proj->GetRMS()));
								}
							}
						}
						else
						{
							dst[0]->SetBinContent(bin, 0.f);
							dst[0]->SetBinError(bin, 0.f);
							dst[1]->SetBinContent(bin, 0.f);
							dst[1]->SetBinError(bin, 0.f);
						}
						delete proj;
					}
					if (ii == 0)
					{
						dstIntegral = src->ProjectionX(config.nativeFitResolutions ? ParameterNamesNative[j] : ParameterNames[j], 0, nBins + 1);
						unsigned int rebin = 1;
						while (dstIntegral->GetMaximum() < 50 && rebin < sizeof(res_axis_bins) / sizeof(res_axis_bins[0]))
						{
							dstIntegral->Rebin(res_axis_bins[rebin - 1]/res_axis_bins[rebin]);
							rebin++;
						}

					}
				}
				if (ii == 0)
				{
					if (config.inputHistogramsOnly) dstIntegral = new TH1D;
					sprintf(fname, p ? "IntPull%s" : "IntRes%s", VSParameterNames[j]);
					sprintf(name, p ? "%s Pull" : "%s Resolution", config.nativeFitResolutions ? ParameterNamesNative[j] : ParameterNames[j]);
					dstIntegral->SetName(fname);
					dstIntegral->SetTitle(name);
				}
				pad->cd();

				int numColor = 0;
				float tmpMax = -1000.;
				float tmpMin = 1000.;
				
				for (int l = 0;l < 2;l++)
				{
					for (int k = 0;k < ConfigNumInputs;k++)
					{
						TH1F* e = dst[l];
						if (GetHist(e, tin, k, nNewInput) == NULL) continue;
						if (nNewInput && k == 0 && ii != 5)
						{
							if (p == 0) e->Scale(config.nativeFitResolutions ? ScaleNative[j] : Scale[j]);
							e->GetYaxis()->SetTitle(p ? AxisTitlesPull[j] : config.nativeFitResolutions ? AxisTitlesNative[j] : AxisTitles[j]);
							e->GetXaxis()->SetTitle(XAxisTitles[i]);
						}
						if (ii == 4) e->GetXaxis()->SetRangeUser(0.2, PT_MAX);
						else if (ii == 5) e->GetXaxis()->SetRange(1, 0);
						e->SetMinimum(-1111);
						e->SetMaximum(-1111);
						
						if (e->GetMaximum() > tmpMax) tmpMax = e->GetMaximum();
						if (e->GetMinimum() < tmpMin) tmpMin = e->GetMinimum();
					}
				}

				float tmpSpan;
				tmpSpan = tmpMax - tmpMin;
				tmpMax += tmpSpan * .02;
				tmpMin -= tmpSpan * .02;
				if (j == 2 && i < 3) tmpMax += tmpSpan * 0.13 * ConfigNumInputs;

				for (int k = 0;k < ConfigNumInputs;k++)
				{
					for (int l = 0;l < 2;l++)
					{
						TH1F* e = dst[l];
						if (!config.inputHistogramsOnly && k == 0)
						{
							sprintf(name, p ? "%s Pull" : "%s Resolution", config.nativeFitResolutions ? ParameterNamesNative[j] : ParameterNames[j]);
							e->SetTitle(name);
							e->SetStats(kFALSE);
							if (tout)
							{
								if (l == 0)
								{
									res2[j][i]->SetOption("colz");
									res2[j][i]->Write();
								}
								e->Write();
							}
						}
						else if (GetHist(e, tin, k, nNewInput) == NULL) continue;
						e->SetMaximum(tmpMax);
						e->SetMinimum(tmpMin);
						e->SetMarkerColor(kBlack);
						e->SetLineWidth(1);
						e->SetLineColor(colorNums[numColor++ % ColorCount]);
						e->SetLineStyle(ConfigDashedMarkers ? k + 1 : 1);
						SetAxisSize(e);
						if (j == 0) e->GetYaxis()->SetTitleOffset(1.5);
						else if (j < 3) e->GetYaxis()->SetTitleOffset(1.4);
						e->Draw(k || l ? "same" : "");
						if (j == 0)
						{
							GetName(fname, k);
							if (p) sprintf(name, "%s%s", fname, l ? "Mean Pull" : "Pull");
							else sprintf(name, "%s%s", fname, l ? "Mean Resolution" : "Resolution");
							leg->AddEntry(e, name, "l");
						}
					}
				}
				if (ii == 5) pad->SetLogx();
				can->cd();
				if (j == 4) ChangePadTitleSize(pad, 0.056);
			}
			leg->Draw();

			sprintf(fname, p ? "plots/pull_vs_%s.pdf" : "plots/res_vs_%s.pdf", VSParameterNames[ii]);
			can->Print(fname);
		}
	}
	
	//Process Integral Resolution Histogreams
	for (int p = 0;p < 2;p++)
	{
		TCanvas* can = p ? cpull[6] : cres[6];
		for (int i = 0;i < 5;i++)
		{
			TPad* pad = p ? ppull[6][i] : pres[6][i];
			TH1D* hist = p ? pullIntegral[i] : resIntegral[i];
			int numColor = 0;
			pad->cd();
			if (!config.inputHistogramsOnly)
			{
				TH1D* e = hist;
				e->GetEntries();
				e->Fit("gaus","sQ");
			}
			
			float tmpMax = 0;
			for (int k = 0;k < ConfigNumInputs;k++)
			{
				TH1D* e = hist;
				if (GetHist(e, tin, k, nNewInput) == NULL) continue;
				e->SetMaximum(-1111);
				if (e->GetMaximum() > tmpMax) tmpMax = e->GetMaximum();
			}
			
			for (int k = 0;k < ConfigNumInputs;k++)
			{
				TH1D* e = hist;
				if (GetHist(e, tin, k, nNewInput) == NULL) continue;
				e->SetMaximum(tmpMax * 1.02);
				e->SetMinimum(tmpMax * -0.02);
				e->SetLineColor(colorNums[numColor++ % ColorCount]);
				if (tout && !config.inputHistogramsOnly && k == 0) e->Write();
				e->Draw(k == 0 ? "" : "same");
			}
			can->cd();
		}
		can->Print(p ? "plots/pull_integral.pdf" : "plots/res_integral.pdf");
		if (!config.inputHistogramsOnly) for (int i = 0;i < 5;i++) delete (p ? pullIntegral : resIntegral)[i];
	}
	
	//Process Cluster Histograms
	{
		if (config.inputHistogramsOnly == 0)
		{
			for (int i = 4;i < 11;i++) clusters[i]->Sumw2(true);
			double totalVal = 0;
			for (int j = 0;j < clusters[3]->GetXaxis()->GetNbins() + 2;j++) totalVal += clusters[3]->GetBinContent(j);
			if (totalVal == 0.) totalVal = 1.;
			for (int i = 0;i < 4;i++)
			{
				double val = 0;
				for (int j = 0;j < clusters[i]->GetXaxis()->GetNbins() + 2;j++)
				{
					val += clusters[i]->GetBinContent(j);
					clusters[7 + i]->SetBinContent(j, val / totalVal);
				}
			}
			for (int i = 0;i < 4;i++)
			{
				clusters[7 + i]->SetMaximum(1.02);
				clusters[7 + i]->SetMinimum(-0.02);
			}
			
			for (int i = 0;i < 3;i++)
			{
				clusters[i + 4]->Divide(clusters[i], clusters[3], 1, 1, "B");
				clusters[i + 4]->SetMinimum(-0.02);
				clusters[i + 4]->SetMaximum(1.02);
			}
		}
		
		float tmpMax = 0, tmpMin = 0;
		for (int k = 0;k < ConfigNumInputs;k++)
		{
			TH1F* e = clusters[3];
			if (GetHist(e, tin, k, nNewInput) == NULL) continue;
			e->SetMinimum(-1111);
			e->SetMaximum(-1111);
			
			if (e->GetMaximum() > tmpMax) tmpMax = e->GetMaximum();
			if (e->GetMinimum() < tmpMin) tmpMin = e->GetMinimum();
		}
		for (int k = 0;k < ConfigNumInputs;k++)
		{
			for (int i = 0;i < 4;i++)
			{
				TH1F* e = clusters[i];
				if (GetHist(e, tin, k, nNewInput) == NULL) continue;
				e->SetMaximum(tmpMax * 1.02);
				e->SetMinimum(tmpMax * -0.02);
			}
		}
		
		for (int i = 0;i < 3;i++)
		{
			pclust[i]->cd();
			pclust[i]->SetLogx();
			int begin = i == 2 ?  7 : i == 1 ? 4 : 0;
			int end   = i == 2 ? 11 : i == 1 ? 7 : 4;
			int numColor = 0;
			for (int k = 0;k < ConfigNumInputs;k++)
			{
				for (int j = begin;j < end;j++)
				{
					TH1F* e = clusters[j];
					if (GetHist(e, tin, k, nNewInput) == NULL) continue;

					e->SetTitle(ClusterTitles[i]);
					e->GetYaxis()->SetTitle(i == 0 ? "Number of clusters" : "Fraction of clusters");
					e->GetXaxis()->SetTitle("p_{Tmc} [Gev/c]");
					if (tout && !config.inputHistogramsOnly && k == 0) e->Write();
					e->SetStats(kFALSE);
					e->SetMarkerColor(kBlack);
					e->SetLineWidth(1);
					e->SetLineColor(colorNums[numColor++ % ColorCount]);
					e->SetLineStyle(ConfigDashedMarkers ? j + 1 : 1);
					e->Draw(j == begin && k == 0 ? "" : "same");
					GetName(fname, k);
					sprintf(name, "%s%s", fname, ClustersNames[j - begin]);
					legendclust[i]->AddEntry(e, name, "l");
				}
			}
			legendclust[i]->Draw();
			cclust[i]->cd();
			cclust[i]->Print(i == 2 ? "plots/clusters_integral.pdf" : i == 1 ? "plots/clusters_relative.pdf" : "plots/clusters.pdf");
		}
	}
	
	//Process track histograms
	{
		float tmpMax = 0.;
		for (int k = 0;k < ConfigNumInputs;k++)
		{
			TH1F* e = tracks;
			if (GetHist(e, tin, k, nNewInput) == NULL) continue;
			e->SetMaximum(-1111);
			if (e->GetMaximum() > tmpMax) tmpMax = e->GetMaximum();
		}
		ptracks->cd();
		ptracks->SetLogx();
		for (int k = 0;k < ConfigNumInputs;k++)
		{
			TH1F* e = tracks;
			if (GetHist(e, tin, k, nNewInput) == NULL) continue;
			if (tout && !config.inputHistogramsOnly && k == 0) e->Write();
			e->SetMaximum(tmpMax * 1.02);
			e->SetMinimum(tmpMax * -0.02);
			e->SetStats(kFALSE);
			e->SetMarkerColor(kBlack);
			e->SetLineWidth(1);
			e->SetLineColor(colorNums[k % ColorCount]);
			e->GetYaxis()->SetTitle("a.u.");
			e->GetXaxis()->SetTitle("p_{Tmc} [Gev/c]");
			e->Draw(k == 0 ? "" : "same");
			GetName(fname, k);
			sprintf(name, "%sTrack Pt", fname);
			legendtracks->AddEntry(e, name, "l");
		}
		legendtracks->Draw();
		ctracks->cd();
		ctracks->Print("plots/tracks.pdf");
		tmpMax = 0.;
		for (int k = 0;k < ConfigNumInputs;k++)
		{
			TH1F* e = ncl;
			if (GetHist(e, tin, k, nNewInput) == NULL) continue;
			e->SetMaximum(-1111);
			if (e->GetMaximum() > tmpMax) tmpMax = e->GetMaximum();
		}
		pncl->cd();
		for (int k = 0;k < ConfigNumInputs;k++)
		{
			TH1F* e = ncl;
			if (GetHist(e, tin, k, nNewInput) == NULL) continue;
			if (tout && !config.inputHistogramsOnly && k == 0) e->Write();
			e->SetMaximum(tmpMax * 1.02);
			e->SetMinimum(tmpMax * -0.02);
			e->SetStats(kFALSE);
			e->SetMarkerColor(kBlack);
			e->SetLineWidth(1);
			e->SetLineColor(colorNums[k % ColorCount]);
			e->GetYaxis()->SetTitle("a.u.");
			e->GetXaxis()->SetTitle("NClusters");
			e->Draw(k == 0 ? "" : "same");
			GetName(fname, k);
			sprintf(name, "%sNClusters", fname);
			legendncl->AddEntry(e, name, "l");
		}
		legendncl->Draw();
		cncl->cd();
		cncl->Print("plots/nClusters.pdf");
	}

	if (tout)
	{
		tout->Close();
		delete tout;
	}
	for (unsigned int i = 0;i < config.compareInputs.size();i++)
	{
		tin[i]->Close();
		delete tin[i];
	}
	return(0);
}
