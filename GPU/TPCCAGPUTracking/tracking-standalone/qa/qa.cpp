#include "Rtypes.h"

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCASliceData.h"
#include "AliHLTTPCCAStandaloneFramework.h"
#include "AliHLTTPCCATrack.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCATrackerFramework.h"
#include "AliHLTTPCGMMergedTrack.h"
//#include "AliHLTTPCGMPhysicalTrackModel.h"
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
#include "TMath.h"
#include "TColor.h"
#include "TPaveText.h"
#include "TF1.h"
#include "TFile.h"
#include "TStyle.h"

#include "../cmodules/qconfig.h"

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

static TH1F* clusters[11]; //attached, fakeAttached, tracks, all, attachedRel, fakeAttachedRel, treaksRel, attachedInt, fakeAttachedInt, treaksInt, AllInt
static TCanvas* cclust[3];
static TPad* pclust[3];
static TLegend* legendclust[3];

#define DEBUG 0

#define SORT_NLABELS 1
#define REC_THRESHOLD 0.9f

bool MCComp(const AliHLTTPCClusterMCWeight& a, const AliHLTTPCClusterMCWeight& b) {return(a.fMCID > b.fMCID);}

#define Y_MAX 100
#define Y_MAX2 40
#define Z_MAX 100
#define PT_MIN 0.015
#define PT_MIN2 0.1
#define PT_MIN_PRIM 0.1
#define PT_MIN_CLUST 0.015
#define PT_MAX 20
#define ETA_MAX 1.5
#define ETA_MAX2 0.9

#define MIN_WEIGHT_CLS 40
#define FINDABLE_WEIGHT_CLS 70

static const int ColorCount = 12;
static Color_t colorNums[ColorCount];

static const char* EffTypes[4] = {"Rec", "Clone", "Fake", "All"};
static const char* FindableNames[2] = {"", "Findable"};
static const char* PrimNames[2] = {"Prim", "Sec"};
static const char* ParameterNames[5] = {"Y", "Z", "#Phi", "#lambda", "Relative p_{T}"};
static const char* VSParameterNames[6] = {"Y", "Z", "Phi", "Eta", "Pt", "Pt_log"};
static const char* EffNames[3] = {"Efficiency", "Clone Rate", "Fake Rate"};
static const char* EfficiencyTitles[4] = {"Efficiency (Primary Tracks, Findable)", "Efficiency (Secondary Tracks, Findable)", "Efficiency (Primary Tracks)", "Efficiency (Secondary Tracks)"};
static const double Scale[5] = {10., 10., 1000., 1000., 100.};
static const char* XAxisTitles[5] = {"y_{mc} [cm]", "z_{mc} [cm]", "#Phi_{mc} [rad]", "#eta_{mc}", "p_{Tmc} [Gev/c]"};
static const char* AxisTitles[5] = {"y-y_{mc} [mm] (Resolution)", "z-z_{mc} [mm] (Resolution)", "#phi-#phi_{mc} [mrad] (Resolution)", "#lambda-#lambda_{mc} [mrad] (Resolution)", "(p_{T} - p_{Tmc}) / p_{Tmc} [%] (Resolution)"};
static const char* ClustersNames[4] = {"Correctly attached clusters", "Fake attached clusters", "Clusters of reconstructed tracks", "All clusters"};
static const char* ClusterTitles[3] = {"Clusters Pt Distribution / Attachment", "Clusters Pt Distribution / Attachment (relative to all clusters)", "Clusters Pt Distribution / Attachment (integrated)"};
static const char* ClusterNamesShort[4] = {"Attached", "Fake", "FoundTracks", "All"};
static const char* ClusterTypes[3] = {"", "Ratio", "Integral"};
static int colorsHex[ColorCount] = {0xB03030, 0x00A000, 0x0000C0, 0x9400D3, 0x19BBBF, 0xF25900, 0x7F7F7F, 0xFFD700, 0x07F707, 0x07F7F7, 0xF08080, 0x000000};

static double legendSpacingString = 0.025;
static int ConfigDashedMarkers = 0;

static const float axes_min[5] = {-Y_MAX2, -Z_MAX, 0., -ETA_MAX, PT_MIN};
static const float axes_max[5] = {Y_MAX2, Z_MAX, 2. *  M_PI, ETA_MAX, PT_MAX};
static const int axis_bins[5] = {50, 50, 144, 30, 50};
static const float res_axes[5] = {1., 1., 0.03, 0.03, 0.2};

static void SetAxisSize(TH1F* e)
{
	e->GetYaxis()->SetTitleOffset(1.0);
	e->GetYaxis()->SetTitleSize(0.045);
	e->GetYaxis()->SetLabelSize(0.045);
	e->GetXaxis()->SetTitleSize(0.045);
	e->GetXaxis()->SetLabelSize(0.045);
}

static double* CreateLogAxis(int nbins, double xmin, double xmax)
{
	double logxmin = TMath::Log10(xmin);
	double logxmax = TMath::Log10(xmax);
	double binwidth = (logxmax-logxmin)/nbins;

	double *xbins =  new double[nbins+1];

	xbins[0] = xmin;
	for (int i=1;i<=nbins;i++)
	{
		xbins[i] = TMath::Power(10,logxmin+i*binwidth);
	}
	return xbins;
}

static void ChangePadTitleSize(TPad* p, double size)
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
				res2[i][j] = new TH2F(name, name, 100, -res_axes[i], res_axes[i], axis_bins[j], binsPt);
				delete[] binsPt;
			}
			else
			{
				res2[i][j] = new TH2F(name, name, 100, -res_axes[i], res_axes[i], axis_bins[j], axes_min[j], axes_max[j]);
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

	if (hlt.GetNMCInfo() == 0 || hlt.GetNMCLabels() == 0)
	{
		printf("Missing MC information, skipping QA\n");
		return;
	}

	//Assign Track MC Labels
	bool ompError = false;
#pragma omp parallel for
	for (int i = 0; i < merger.NOutputTracks(); i++)
	{
		if (ompError) continue;
		int nClusters = 0;
		const AliHLTTPCGMMergedTrack &track = merger.OutputTracks()[i];
		std::vector<AliHLTTPCClusterMCWeight> labels;
		for (int k = 0;k < track.NClusters();k++)
		{
			if (merger.ClusterRow()[track.FirstClusterRef() + k] < 0) continue;
			nClusters++;
			int hitId = merger.OutputClusterIds()[track.FirstClusterRef() + k];
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
			trackMCLabels[i] = -1;
			totalFakes++;
			continue;
		}
		std::sort(labels.data(), labels.data() + labels.size(), MCComp);
		
		AliHLTTPCClusterMCWeight maxLabel;
		AliHLTTPCClusterMCWeight cur = labels[0];
		if (SORT_NLABELS) cur.fWeight = 1;
		float sumweight = 0.f;
		int curcount = 1, maxcount = 0;
		if (DEBUG >= 2 && track.OK()) for (unsigned int k = 0;k < labels.size();k++) printf("\t%d %f\n", labels[k].fMCID, labels[k].fWeight);
		for (unsigned int k = 1;k <= labels.size();k++)
		{
			if (k == labels.size() || labels[k].fMCID != cur.fMCID)
			{
				sumweight += cur.fWeight;
				if (cur.fWeight > maxLabel.fWeight)
				{
					if (maxcount >= REC_THRESHOLD * nClusters) recTracks[maxLabel.fMCID]++;
					maxLabel = cur;
					maxcount = curcount;
				}
				if (k < labels.size())
				{
					cur = labels[k];
					if (SORT_NLABELS) cur.fWeight = 1;
					curcount = 1;
				}
			}
			else
			{
				cur.fWeight += SORT_NLABELS ? 1 : labels[k].fWeight;
				curcount++;
			}
		}
		
		if (track.OK())
		{
			for (int k = 0;k < track.NClusters();k++)
			{
				if (merger.ClusterRow()[track.FirstClusterRef() + k] < 0) continue;
				int hitId = merger.OutputClusterIds()[track.FirstClusterRef() + k];
				bool correct = false;
				for (int j = 0;j < 3;j++) if (hlt.GetMCLabels()[hitId].fClusterID[j].fMCID == maxLabel.fMCID) {correct=true;break;}
				if (correct) clusterParam[hitId].attached++;
				else clusterParam[hitId].fakeAttached++;
			}
		}

		if (maxcount < REC_THRESHOLD * nClusters)
		{
			fakeTracks[maxLabel.fMCID]++;
			maxLabel.fMCID = -2 - maxLabel.fMCID;
		}
		else
		{
			recTracks[maxLabel.fMCID]++;
			
			int& revLabel = trackMCLabelsReverse[maxLabel.fMCID];
			if (revLabel == -1 ||
				!merger.OutputTracks()[revLabel].OK() ||
				(merger.OutputTracks()[i].OK() && fabs(merger.OutputTracks()[i].GetParam().GetZ()) < fabs(merger.OutputTracks()[revLabel].GetParam().GetZ())))
			{
				revLabel = i;
			}
		}
		trackMCLabels[i] = maxLabel.fMCID;
		if (DEBUG && track.OK() && hlt.GetNMCInfo() > maxLabel.fMCID)
		{
			const AliHLTTPCCAMCInfo& mc = hlt.GetMCInfo()[maxLabel.fMCID >= 0 ? maxLabel.fMCID : (-maxLabel.fMCID - 2)];
			printf("Track %d label %d weight %f clusters %d (%f%% %f%%) Pt %f\n", i, maxLabel.fMCID >= 0 ? maxLabel.fMCID : (maxLabel.fMCID + 2), maxLabel.fWeight, nClusters, maxLabel.fWeight / sumweight, (float) maxcount / (float) nClusters, sqrt(mc.fPx * mc.fPx + mc.fPy * mc.fPy));
		}
	}
	if (ompError) return;
	
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
	
	//Compute MC Track Parameters for MC Tracks
	//Fill Efficiency Histograms
	for (int i = 0;i < hlt.GetNMCInfo();i++)
	{
		if ((mcTrackMin != -1 && i < mcTrackMin) || (mcTrackMax != -1 && i >= mcTrackMax)) continue;
		const AliHLTTPCCAMCInfo& info = hlt.GetMCInfo()[i];
		if (mcParam[i].nWeightCls == 0.) continue;
		float mcpt = TMath::Sqrt(info.fPx * info.fPx + info.fPy * info.fPy);
		float mcphi = TMath::Pi() + TMath::ATan2(-info.fPy,-info.fPx);
		float mctheta = info.fPz ==0 ? (TMath::Pi() / 2) : (TMath::ACos(info.fPz / TMath::Sqrt(info.fPx*info.fPx+info.fPy*info.fPy+info.fPz*info.fPz)));
		float mceta = -TMath::Log(TMath::Tan(0.5 * mctheta));
		mcParam[i].pt = mcpt; mcParam[i].phi = mcphi; mcParam[i].theta = mctheta; mcParam[i].eta = mceta;

		if (info.fPrim && info.fPrimDaughters) continue;
		if (mcParam[i].nWeightCls < MIN_WEIGHT_CLS) continue;
		int findable = mcParam[i].nWeightCls >= FINDABLE_WEIGHT_CLS;
		if (info.fPID < 0) continue;
		if (info.fCharge == 0.) continue;
		
		if (fabs(mceta) > ETA_MAX || mcpt < PT_MIN || mcpt > PT_MAX) continue;
		
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
					if (l < 4 && mcpt < 1. / config.qpt) continue;
					
					float pos = l == 0 ? info.fX : l == 1 ? info.fY : l == 2 ? mcphi : l == 3 ? mceta : mcpt;

					eff[j][k][!info.fPrim][l][0]->Fill(pos, val);
				}
			}
		}
	}
	
	//Fill Resolution Histograms
	AliHLTTPCGMPropagator prop;
	const float kRho = 1.025e-3;//0.9e-3;
	const float kRadLen = 29.532;//28.94;
	prop.SetMaxSinPhi( .999 );
	prop.SetMaterial( kRadLen, kRho );
	prop.SetPolynomialField( merger.Field() );
	prop.SetPolynomialFieldBz( merger.PolinomialFieldBz() );
	prop.SetUseMeanMomentum(kFALSE );
	prop.SetContinuousTracking( kFALSE );
	
	for (int i = 0; i < merger.NOutputTracks(); i++)
	{
		if (trackMCLabels[i] < 0) continue;
		const AliHLTTPCCAMCInfo& mc1 = hlt.GetMCInfo()[trackMCLabels[i]];
		const additionalMCParameters& mc2 = mcParam[trackMCLabels[i]];
		const AliHLTTPCGMMergedTrack& track = merger.OutputTracks()[i];
		
		if ((mcTrackMin != -1 && trackMCLabels[i] < mcTrackMin) || (mcTrackMax != -1 && trackMCLabels[i] >= mcTrackMax)) continue;
		
		if (!track.OK()) continue;
		if (fabs(mc2.eta) > ETA_MAX || mc2.pt < PT_MIN || mc2.pt > PT_MAX) continue;
		if (mc1.fCharge == 0.) continue;
		if (mc1.fPID < 0) continue;
		if (mc2.nWeightCls < MIN_WEIGHT_CLS) continue;
		
		float mclocal[4]; //Rotated x,y,Px,Py mc-coordinates - the MC data should be rotated since the track is propagated best along x
		float c = TMath::Cos(track.GetAlpha());
		float s = TMath::Sin(track.GetAlpha());
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

		float alpha = track.GetAlpha();		
		prop.SetTrack(&param, alpha);	
		bool inFlyDirection = 0;
		if (config.strict && (param.X() - mclocal[0]) * (param.X() - mclocal[0]) + (param.Y() - mclocal[1]) * (param.Y() - mclocal[1]) + (param.Z() - mc1.fZ) * (param.Z() - mc1.fZ) > 25) continue;
		
		if (prop.PropagateToXAlpha( mclocal[0], alpha, inFlyDirection ) ) continue;
		if (fabs(param.Y() - mclocal[1]) > (config.strict ? 1. : 4.) || fabs(param.Z() - mc1.fZ) > (config.strict ? 1. : 4.)) continue;
		
		float deltaY = param.GetY() - mclocal[1];
		float deltaZ = param.GetZ() - mc1.fZ;
		float deltaPhi = TMath::ASin(param.GetSinPhi()) - TMath::ATan2(mclocal[3], mclocal[2]);
		float deltaLambda = TMath::ATan(param.GetDzDs()) - TMath::ATan2(mc1.fPz, mc2.pt);
		float deltaPt = (fabs(1. / param.GetQPt()) - mc2.pt) / mc2.pt;
		
		float paramval[5] = {mclocal[1], mc1.fZ, mc2.phi, mc2.eta, mc2.pt};
		float resval[5] = {deltaY, deltaZ, deltaPhi, deltaLambda, deltaPt};
		
		for (int j = 0;j < 5;j++)
		{
			for (int k = 0;k < 5;k++)
			{
				if (k != 3 && fabs(mc2.eta) > ETA_MAX2) continue;
				if (k < 4 && mc2.pt < 1. / config.qpt) continue;
				res2[j][k]->Fill(resval[j], paramval[k]);
			}
		}
	}
	
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
}

int DrawQAHistograms()
{
	char name[1024], fname[1024];
	
	structConfigQA& config = configStandalone.configQA;
	int nNewInput = config.inputHistogramsOnly ? 0 : 1;
	int ConfigNumInputs = nNewInput + config.compareInputs.size();
	
	std::vector<TFile*> tin(config.compareInputs.size());
	for (unsigned int i = 0;i < config.compareInputs.size();i++)
	{
		tin[i] = new TFile(config.compareInputs[i]);
	}
	TFile* tout = NULL;
	if (config.output) tout = new TFile(config.output, "RECREATE");

	//Create Canvas / Pads for Efficiency Histograms
	for (int ii = 0;ii < 6;ii++)
	{
		int i = ii == 5 ? 4 : ii;
		sprintf(fname, "ceff_%d", ii);
		sprintf(name, "Efficiency versus %s", ParameterNames[i]);
		ceff[ii] = new TCanvas(fname,name,0,0,700,700.*2./3.);
		ceff[ii]->cd();
		float dy = 1. / 2.;
		peff[ii][0] = new TPad( "p0","",0.0,dy*0,0.5,dy*1); peff[ii][0]->Draw();peff[ii][0]->SetRightMargin(0.04);
		peff[ii][1] = new TPad( "p1","",0.5,dy*0,1.0,dy*1); peff[ii][1]->Draw();peff[ii][1]->SetRightMargin(0.04);
		peff[ii][2] = new TPad( "p2","",0.0,dy*1,0.5,dy*2-.001); peff[ii][2]->Draw();peff[ii][2]->SetRightMargin(0.04);
		peff[ii][3] = new TPad( "p3","",0.5,dy*1,1.0,dy*2-.001); peff[ii][3]->Draw();peff[ii][3]->SetRightMargin(0.04);
		legendeff[ii] = new TLegend(0.92 - legendSpacingString * 1.45, 0.83 - (0.93 - 0.83) / 2. * (float) ConfigNumInputs,0.98,0.849);
		legendeff[ii]->SetTextFont(72);
		legendeff[ii]->SetTextSize(0.016);
		legendeff[ii]->SetFillColor(0);
	}

	//Create Canvas / Pads for Resolution Histograms
	for (int ii = 0;ii < 7;ii++)
	{
		int i = ii == 5 ? 4 : ii;
		sprintf(fname, "cres_%d", ii);
		if (ii == 6) sprintf(name, "Integral Resolution");
		else sprintf(name, "Resolution versus %s", ParameterNames[i]);
		cres[ii] = new TCanvas(fname,name,0,0,700,700.*2./3.);
		cres[ii]->cd();
		gStyle->SetOptFit(1);

		float dy = 1. / 2.;
		pres[ii][3] = new TPad( "p0","",0.0,dy*0,0.5,dy*1); pres[ii][3]->Draw();pres[ii][3]->SetRightMargin(0.04);
		pres[ii][4] = new TPad( "p1","",0.5,dy*0,1.0,dy*1); pres[ii][4]->Draw();pres[ii][4]->SetRightMargin(0.04);
		pres[ii][0] = new TPad( "p2","",0.0,dy*1,1./3.,dy*2-.001); pres[ii][0]->Draw();pres[ii][0]->SetRightMargin(0.04);pres[ii][0]->SetLeftMargin(0.15);
		pres[ii][1] = new TPad( "p3","",1./3.,dy*1,2./3.,dy*2-.001); pres[ii][1]->Draw();pres[ii][1]->SetRightMargin(0.04);pres[ii][1]->SetLeftMargin(0.135);
		pres[ii][2] = new TPad( "p4","",2./3.,dy*1,1.0,dy*2-.001); pres[ii][2]->Draw();pres[ii][2]->SetRightMargin(0.06);pres[ii][2]->SetLeftMargin(0.135);
		if (ii < 6)
		{
			legendres[ii] = new TLegend(0.9 - legendSpacingString * 1.45, 0.93 - (0.93 - 0.87) / 2. * (float) ConfigNumInputs, 0.98, 0.949);
			legendres[ii]->SetTextFont(72);
			legendres[ii]->SetTextSize(0.016);
			legendres[ii]->SetFillColor(0);
		}
	}
	
	//Create Canvas for Cluster Histos
	for (int i = 0;i < 3;i++)
	{
		sprintf(fname, "cclust_%d", i);
		cclust[i] = new TCanvas(fname,ClusterTitles[i],0,0,700,700.*2./3.);
		cclust[i]->cd();
		pclust[i] = new TPad( "p0","",0.0,0.0,1.0,1.0);
		pclust[i]->Draw();
		legendclust[i] = new TLegend(i == 2 ? 0.1 : 0.65, i != 1 ? 0.8 : 0.45, i == 2 ? 0.35 : 0.9, i != 1 ? 0.9 : 0.55);
		legendclust[i]->SetTextFont(72);
		legendclust[i]->SetTextSize(0.016);
		legendclust[i]->SetFillColor(0);
	}
	
	printf("QA Stats: Eff: Tracks Prim %d (Eta %d, Pt %d) %f%% (%f%%) Sec %d (Eta %d, Pt %d) %f%% (%f%%) -  Res: Tracks %d (Eta %d, Pt %d)\n",
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
					else if ((e = dynamic_cast<TH1F*>(tin[k - nNewInput]->Get(e->GetName()))) == NULL)
					{
						printf("Missing histogram in input %s\n", config.compareInputs[k - nNewInput]);
						return(1);
					}

					e->SetMarkerColor(kBlack);
					e->SetLineWidth(1);
					e->SetLineColor(colorNums[(l == 2 ? (ConfigNumInputs * 2 + k) : (k * 2 + l)) % ColorCount]);
					e->SetLineStyle(ConfigDashedMarkers ? k + 1 : 1);
					SetAxisSize(e);
					e->Draw(k || l ? "same" : "");
					if (j == 0)
					{
						if (k || config.inputHistogramsOnly || config.name) sprintf(fname, "%s - ", config.inputHistogramsOnly || k ? (config.compareInputNames.size() > (unsigned) (k - nNewInput) ? config.compareInputNames[k - nNewInput] : config.compareInputs[k - nNewInput]) : config.name);
						else fname[0] = 0;
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
		sprintf(fname, "eff_vs_%s.pdf", VSParameterNames[ii]);
		ceff[ii]->Print(fname);
	}

	//Process / Draw Resolution Histograms
	TH1D* resIntegral[5] = {};
	for (int ii = 0;ii < 6;ii++)
	{
		int i = ii == 5 ? 4 : ii;
		for (int j = 0;j < 5;j++)
		{
			if (!config.inputHistogramsOnly && ii != 5)
			{
				TCanvas cfit;
				cfit.cd();
				TAxis* axis = res2[j][i]->GetYaxis();
				int nBins = axis->GetNbins();
				int integ = 1;
				for (int bin = 1;bin <= nBins;bin++)
				{
					int bin0 = std::max(bin - integ, 0);
					int bin1 = std::min(bin + integ, nBins);
					TH1D* proj = res2[j][i]->ProjectionX("proj", bin0, bin1);
					if (proj->GetEntries())
					{
						if (proj->GetEntries() < 20 || proj->GetRMS() < 0.00001)
						{
							res[j][i][0]->SetBinContent(bin, proj->GetRMS());
							res[j][i][0]->SetBinError(bin, sqrt(proj->GetRMS()));
							res[j][i][1]->SetBinContent(bin, proj->GetMean());
							res[j][i][1]->SetBinError(bin, sqrt(proj->GetRMS()));
						}
						else
						{
							proj->Fit("gaus", proj->GetMaximum() < 20 ? "sQl" : "sQ");
							TF1* fitFunc = proj->GetFunction("gaus");
							float sigma = fabs(fitFunc->GetParameter(2));
							if (sigma > 0.)
							{
								res[j][i][0]->SetBinContent(bin, fabs(fitFunc->GetParameter(2)));
								res[j][i][1]->SetBinContent(bin, fitFunc->GetParameter(1));
							}
							else
							{
								res[j][i][0]->SetBinContent(bin, 0);
								res[j][i][1]->SetBinContent(bin, 0);
							}
							res[j][i][0]->SetBinError(bin, fitFunc->GetParError(2));
							res[j][i][1]->SetBinError(bin, fitFunc->GetParError(1));

							bool fail = fabs(proj->GetMean() - res[j][i][1]->GetBinContent(bin)) > res_axes[j] || res[j][i][0]->GetBinError(bin) > 1 || res[j][i][1]->GetBinError(bin) > 1;
							if (fail)
							{
								res[j][i][0]->SetBinContent(bin, proj->GetMean());
								res[j][i][1]->SetBinContent(bin, proj->GetRMS());
								res[j][i][0]->SetBinError(bin, sqrt(proj->GetRMS()));
								res[j][i][1]->SetBinError(bin, sqrt(proj->GetRMS()));
							}
						}
					}
					else
					{
						res[j][i][0]->SetBinContent(bin, 0.);
						res[j][i][0]->SetBinError(bin, 0.);
						res[j][i][1]->SetBinContent(bin, 0.);
						res[j][i][1]->SetBinError(bin, 0.);
					}
					delete proj;
				}
				if (ii == 0)
				{
					resIntegral[j] = res2[j][0]->ProjectionX(ParameterNames[j], 0, nBins + 1);
				}
			}
			pres[ii][j]->cd();

			int numColor = 0;
			float tmpMax = -1000.;
			float tmpMin = 1000.;
			
			for (int l = 0;l < 2;l++)
			{
				for (int k = 0;k < ConfigNumInputs;k++)
				{
					TH1F* e = res[j][i][l];
					if ((config.inputHistogramsOnly || k) && (e = dynamic_cast<TH1F*>(tin[k - nNewInput]->Get(e->GetName()))) == NULL)
					{
						printf("Missing histogram in input %s\n", config.compareInputs[k - nNewInput]);
						return(1);
					}
					if (nNewInput && k == 0 && ii != 5)
					{
						e->Scale(Scale[j]);
						e->GetYaxis()->SetTitle(AxisTitles[j]);
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

			double tmpSpan;
			tmpSpan = tmpMax - tmpMin;
			tmpMax += tmpSpan * .02;
			tmpMin -= tmpSpan * .02;
			if (j == 2 && i < 3) tmpMax += tmpSpan * 0.13 * ConfigNumInputs;

			for (int k = 0;k < ConfigNumInputs;k++)
			{
				for (int l = 0;l < 2;l++)
				{
					TH1F* e = res[j][i][l];
					if (!config.inputHistogramsOnly && k == 0)
					{
						sprintf(name, "%s Resolution", ParameterNames[j]);
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
					else if ((e = dynamic_cast<TH1F*>(tin[k - nNewInput]->Get(e->GetName()))) == NULL)
					{
						printf("Missing histogram in input %s\n", config.compareInputs[k - nNewInput]);
						return(1);
					}
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
						if (k || config.inputHistogramsOnly || config.name) sprintf(fname, "%s - ", config.inputHistogramsOnly || k ? (config.compareInputNames.size() > (unsigned) (k - nNewInput) ? config.compareInputNames[k - nNewInput] : config.compareInputs[k - nNewInput]) : config.name);
						else fname[0] = 0;
						sprintf(name, "%s%s", fname, l ? "Mean Resolution" : "Resolution");
						legendres[ii]->AddEntry(e, name, "l");
					}
				}
			}
			if (ii == 5) pres[ii][j]->SetLogx();
			cres[ii]->cd();
		}
		ChangePadTitleSize(pres[ii][4], 0.056);
		legendres[ii]->Draw();

		sprintf(fname, "res_vs_%s.pdf", VSParameterNames[ii]);
		cres[ii]->Print(fname);
	}
	
	//Process Integral Resolution Histogreams
	if (config.inputHistogramsOnly)
	{
		std::remove("res_integral.pdf");
	}
	else
	{
		for (int i = 0;i < 5;i++)
		{
			pres[6][i]->cd();
			sprintf(fname, "IntRes%s", ParameterNames[i]);
			sprintf(name, "%s Resolution", ParameterNames[i]);
			resIntegral[i]->SetName(fname);
			resIntegral[i]->SetTitle(name);
			resIntegral[i]->GetEntries();
			resIntegral[i]->Fit("gaus","sQ");
			if (tout) resIntegral[i]->Write();
			resIntegral[i]->Draw();
			cres[6]->cd();
		}
		cres[6]->Print("res_integral.pdf");
		for (int j = 0;j < 5;j++) delete resIntegral[j];
	}
	
	//Process Cluster Histograms
	if (config.inputHistogramsOnly)
	{
		std::remove("clusters.pdf");
		std::remove("clusters_relative.pdf");
		std::remove("clusters_integral.pdf");
	}
	else
	{
		for (int i = 0;i < 11;i++) clusters[i]->Sumw2();
		
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
			clusters[7 + i]->SetMaximum(clusters[10]->GetMaximum() * 1.02);
			clusters[7 + i]->SetMinimum(clusters[10]->GetMaximum() * -0.02);
		}
		
		for (int i = 0;i < 3;i++)
		{
			clusters[i + 4]->Divide(clusters[i], clusters[3], 1, 1, "B");
			clusters[i + 4]->SetMinimum(-0.02);
			clusters[i + 4]->SetMaximum(1.02);
		}
		
		for (int i = 0;i < 4;i++)
		{
			clusters[i]->SetMaximum(clusters[3]->GetMaximum() * 1.02);
			clusters[i]->SetMinimum(clusters[3]->GetMaximum() * -0.02);
		}
		
		for (int i = 0;i < 3;i++)
		{
			pclust[i]->cd();
			pclust[i]->SetLogx();
			int begin = i == 2 ?  7 : i == 1 ? 4 : 0;
			int end   = i == 2 ? 11 : i == 1 ? 7 : 4;
			int numColor = 0;
			for (int j = begin;j < end;j++)
			{
				TH1F* e = clusters[j];
				e->SetTitle(ClusterTitles[i]);
				e->GetYaxis()->SetTitle(i == 0 ? "Number of clusters" : "Fraction of clusters");
				e->GetXaxis()->SetTitle("p_{Tmc} [Gev/c]");
				if (tout) e->Write();
				e->SetStats(kFALSE);
				e->SetMarkerColor(kBlack);
				e->SetLineWidth(1);
				e->SetLineColor(colorNums[numColor++ % ColorCount]);
				e->SetLineStyle(ConfigDashedMarkers ? j + 1 : 1);
				e->Draw(j == begin ? "" : "same");
				legendclust[i]->AddEntry(e, ClustersNames[j - begin], "l");
			}
			legendclust[i]->Draw();
			cclust[i]->cd();
			cclust[i]->Print(i == 2 ? "clusters_integral.pdf" : i == 1 ? "clusters_relative.pdf" : "clusters.pdf");
		}
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
