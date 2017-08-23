#include "Rtypes.h"

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCASliceData.h"
#include "AliHLTTPCCAStandaloneFramework.h"
#include "AliHLTTPCCATrack.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCATrackerFramework.h"
#include "AliHLTTPCGMMergedTrack.h"
#include "AliHLTTPCGMTrackLinearisation.h"
#include "include.h"
#include <algorithm>

#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TLegend.h"
#include "TMath.h"
#include "TColor.h"
#include "TPaveText.h"

struct additionalMCParameters
{
	float pt, phi, theta, eta;
};

std::vector<int> trackMCLabels;
std::vector<int> trackMCLabelsReverse;
std::vector<int> recTracks;
std::vector<int> fakeTracks;
std::vector<additionalMCParameters> mcParam;
int totalFakes = 0;

TH1F* eff[4][2][2][5][2]; //eff,clone,fake - findable - secondaries - y,z,phi,eta,pt,ptlog - work,result
TH1F* res[5][5][2]; //y,z,phi,lambda,pt,ptlog - see above - res,mean
TH2F* res2[5][5];
TCanvas *ceff[6];
TPad* peff[6][4];
TLegend* legendeff[6];
bool init = false;

#define SORT_NLABELS 1
#define REC_THRESHOLD 0.9f

bool MCComp(const AliHLTTPCClusterMCWeight& a, const AliHLTTPCClusterMCWeight& b) {return(a.fMCID > b.fMCID);}

#define Y_MIN -100
#define Y_MAX 100
#define Z_MIN -100
#define Z_MAX 100
#define PT_MIN 0.015
#define PT_MIN2 0.1
#define PT_MIN_PRIM 0.1
#define PT_MAX 20
#define ETA_MAX 1.5
#define ETA_MAX2 0.9

#define MIN_WEIGHT_CLS 40
#define FINDABLE_WEIGHT_CLS 70

const Int_t ColorCount = 12;
Color_t colorNums[ColorCount];

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

static void ChangePadTitleSize(TPad* p, Double_t size)
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

void RunQA()
{
	const Char_t* EffTypes[4] = {"rec", "clone", "fake", "all"};
	const Char_t* FindableNames[2] = {"all", "findables"};
	const Char_t* PrimNames[2] = {"primaries", "secondaries"};
	const Char_t* ParameterNames[5] = {"Y", "Z", "#Phi", "#lambda", "Relative p_{T}"};
	const Char_t* VSParameterNames[6] = {"Y", "Z", "Phi", "Eta", "Pt", "Pt_log"};
	const Char_t* ShortParameterNames[5] = {"", "", "phi", "eta", "pt"};
	const Char_t* EffNames[3] = {"Efficiency", "Clone Rate", "Fake Rate"};
	const Char_t* ShortEffNames[3] = {"RecEff", "Clone", "Fake"};
	const Char_t* EfficiencyTitles[4] = {"Efficiency (Primary Tracks, Findable)", "Efficiency (Secondary Tracks, Findable)", "Efficiency (Primary Tracks)", "Efficiency (Secondary Tracks)"};
	const Double_t Scale[5] = {10., 10., 1000., 1000., 100.};
	const Char_t* XAxisTitles[5] = {"y_{mc} [cm]", "z_{mc} [cm]", "#Phi_{mc} [rad]", "#eta_{mc}", "p_{Tmc} [Gev/c]"};
	const Char_t* AxisTitles[5] = {"y-y_{mc} [mm] (Resolution)", "z-z_{mc} [mm] (Resolution)", "#phi-#phi_{mc} [mrad] (Resolution)", "#lambda-#lambda_{mc} [mrad] (Resolution)", "(p_{T} - p_{Tmc}) / p_{Tmc} [%] (Resolution)"};
	Int_t colorsHex[ColorCount] = {0xB03030, 0x00A000, 0x0000C0, 0x9400D3, 0x19BBBF, 0xF25900, 0x7F7F7F, 0xFFD700, 0x07F707, 0x07F7F7, 0xF08080, 0x000000};

	double legendSpacingString = 0;
	int ConfigNumInputs = 1;
	int ConfigDashedMarkers = 0;

	char name[1024], fname[1024];

	//Create Histograms
	if (init == false)
	{
		for (Int_t i = 0;i < ColorCount;i++)
		{
			float f1 = (float) ((colorsHex[i] >> 16) & 0xFF) / (float) 0xFF;
			float f2 = (float) ((colorsHex[i] >> 8) & 0xFF) / (float) 0xFF;
			float f3 = (float) ((colorsHex[i] >> 0) & 0xFF) / (float) 0xFF;
			TColor* c = new TColor(10000 + i, f1, f2, f3);
			colorNums[i] = c->GetNumber();
		}
		
		const float axes_min[5] = {Y_MIN, Z_MIN, 0., -ETA_MAX, PT_MIN};
		const float axes_max[5] = {Y_MAX, Z_MAX, 2. *  M_PI, ETA_MAX, PT_MAX};
		const int axis_bins[5] = {50, 50, 144, 30, 50};
		const float res_axes[5] = {-1.,-1.,-0.03,-0.03,-0.2};
		
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
							sprintf(name, "%s_%s-%s_vs_%s%s", EffTypes[i], FindableNames[j], PrimNames[k], VSParameterNames[l], m ? "" : "_work");
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
		
		//Create Canvas / Pads for Efficiency Histograms
		for (int ii = 0;ii < 6;ii++)
		{
			Int_t i = ii == 5 ? 4 : ii;
			sprintf(fname, "ceff_%d", ii);
			sprintf(name, "Efficiency versus %s", ParameterNames[i]);
			ceff[ii] = new TCanvas(fname,name,0,0,700,700.*2./3.);
			Float_t dy=1./2.;
			peff[ii][0] = new TPad( "p0","",0.0,dy*0,0.5,dy*1); peff[ii][0]->Draw();peff[ii][0]->SetRightMargin(0.04);
			peff[ii][1] = new TPad( "p1","",0.5,dy*0,1.0,dy*1); peff[ii][1]->Draw();peff[ii][1]->SetRightMargin(0.04);
			peff[ii][2] = new TPad( "p2","",0.0,dy*1,0.5,dy*2-.001); peff[ii][2]->Draw();peff[ii][2]->SetRightMargin(0.04);
			peff[ii][3] = new TPad( "p3","",0.5,dy*1,1.0,dy*2-.001); peff[ii][3]->Draw();peff[ii][3]->SetRightMargin(0.04);
			legendeff[ii] = new TLegend(0.92 - legendSpacingString * 1.45, 0.93 - (0.93 - 0.83) / 2. * (float) ConfigNumInputs,0.98,0.949);
			legendeff[ii]->SetTextFont(72);
			legendeff[ii]->SetTextSize(0.016);
			legendeff[ii]->SetFillColor(0);
		}
		
		//Create Resolution Histograms
		for (int i = 0;i < 5;i++)
		{
			for (int j = 0;j < 5;j++)
			{
				sprintf(name, "res_%s_vs_%s", VSParameterNames[i], VSParameterNames[j]);
				sprintf(fname, "meanres_%s_vs_%s", VSParameterNames[i], VSParameterNames[j]);
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
				strcat(name, "_work");
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
	}
	
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
	totalFakes = 0;

	//Assign Track MC Labels
	for (int i = 0; i < merger.NOutputTracks(); i++)
	{
		int nClusters = 0;
		const AliHLTTPCGMMergedTrack &track = merger.OutputTracks()[i];
		std::vector<AliHLTTPCClusterMCWeight> labels;
		for (int k = 0;k < track.NClusters();k++)
		{
			if (merger.ClusterRowType()[track.FirstClusterRef() + k] < 0) continue;
			nClusters++;
			int hitId = merger.OutputClusterIds()[track.FirstClusterRef() + k];
			if (hitId > hlt.GetNMCLabels()) {printf("Invalid hit id\n");return;}
			for (int j = 0;j < 3;j++)
			{
				if (hlt.GetMCLabels()[hitId].fClusterID[j].fMCID >= hlt.GetNMCInfo()) {printf("Invalid label\n");return;}
				if (hlt.GetMCLabels()[hitId].fClusterID[j].fMCID >= 0) labels.push_back(hlt.GetMCLabels()[hitId].fClusterID[j]);
			}
		}
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
		//for (unsigned int k = 0;k < labels.size();k++) printf("\t%d %f\n", labels[k].fMCID, labels[k].fWeight);
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
		if (0 && track.OK() && hlt.GetNMCInfo() > maxLabel.fMCID)
		{
			const AliHLTTPCCAMCInfo& mc = hlt.GetMCInfo()[maxLabel.fMCID];
			printf("Track %d label %d weight %f (%f%% %f%%) Pt %f\n", i, maxLabel.fMCID, maxLabel.fWeight, maxLabel.fWeight / sumweight, (float) maxcount / (float) nClusters, sqrt(mc.fPx * mc.fPx + mc.fPy * mc.fPy));
		}
	}
	
	//Compute MC Track Parameters for MC Tracks
	//Fill Efficiency Histograms
	for (int i = 0;i < hlt.GetNMCInfo();i++)
	{
		const AliHLTTPCCAMCInfo& info = hlt.GetMCInfo()[i];
		if (info.fNWeightCls == 0.) continue;
		if (info.fPrim && info.fPrimDaughters) continue;
		if (info.fNWeightCls < MIN_WEIGHT_CLS) continue;
		int findable = info.fNWeightCls >= FINDABLE_WEIGHT_CLS;
		if (info.fPID < 0) continue;
		if (info.fCharge == 0.) continue;
		
		float mcpt = TMath::Sqrt(info.fPx * info.fPx + info.fPy * info.fPy);
		float mcphi = TMath::Pi() + TMath::ATan2(-info.fPy,-info.fPx);
		float mctheta = info.fPz ==0 ? (TMath::Pi() / 2) : (TMath::ACos(info.fPz / TMath::Sqrt(info.fPx*info.fPx+info.fPy*info.fPy+info.fPz*info.fPz)));
		float mceta = -TMath::Log(TMath::Tan(0.5 * mctheta));
		
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
					if (l != 3 && fabs(mceta) > ETA_MAX) continue;
					if (l < 4 && mcpt < PT_MIN2) continue;
					
					float pos = l == 0 ? info.fX : l == 1 ? info.fY : l == 2 ? mcphi : l == 3 ? mceta : mcpt;

					eff[j][k][!info.fPrim][l][0]->Fill(pos, val);
				}
			}
		}
		
		mcParam[i].pt = mcpt; mcParam[i].phi = mcphi; mcParam[i].theta = mctheta; mcParam[i].eta = mceta;
	}
	
	//Fill Resolution Histograms
	for (int i = 0; i < merger.NOutputTracks(); i++)
	{
		if (trackMCLabels[i] < 0) continue;
		const AliHLTTPCCAMCInfo& mc1 = hlt.GetMCInfo()[trackMCLabels[i]];
		const additionalMCParameters& mc2 = mcParam[trackMCLabels[i]];
		const AliHLTTPCGMMergedTrack& track = merger.OutputTracks()[i];
		
		if (!track.OK()) continue;
		if (fabs(mc2.eta) > ETA_MAX || mc2.pt < PT_MIN || mc2.pt > PT_MAX) continue;
		if (mc1.fCharge == 0.) continue;
		if (mc1.fPID < 0) continue;
		if (mc1.fNWeightCls < MIN_WEIGHT_CLS) continue;
		
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
		AliHLTTPCGMTrackLinearisation t0(param);
		AliHLTTPCGMTrackParam::AliHLTTPCGMTrackFitParam par;
		int N = 0;
		float alpha = track.GetAlpha();
		float dL = 0., ex1i = 0., trDzDs2 = t0.DzDs() * t0.DzDs();
		const float kRho = 1.025e-3;//0.9e-3;
		const float kRadLen = 29.532;//28.94;
		const float kRhoOverRadLen = kRho / kRadLen;
		param.CalculateFitParameters( par, kRhoOverRadLen, kRho, false );
		param.PropagateTrack(merger.PolinomialFieldBz(), mclocal[0], mclocal[1], mc1.fZ, track.GetAlpha(), 0, merger.SliceParam(), N, alpha, 0.999, false, false, par, t0, dL, ex1i, trDzDs2);
		
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
				res2[j][k]->Fill(paramval[j], resval[k]);
			}
		}
	}
	
	//Process / Draw Efficiency Histograms
	for (int ii = 0;ii < 6;ii++)
	{
		Int_t i = ii == 5 ? 4 : ii;
		for (int j = 0;j < 4;j++)
		{
			peff[ii][j]->cd();
			for (int l = 0;l < 3;l++)
			{
				int k = 0;
				if (ii != 5)
				{
					if (l == 0)
					{
						eff[0][j / 2][j % 2][i][1]->Divide(eff[l][j / 2][j % 2][i][0], eff[3][j / 2][j % 2][i][0], 1, 1, "B");
						eff[3][j / 2][j % 2][i][1]->Reset(); //Sum up rec + clone + fake for clone/fake rate
						eff[3][j / 2][j % 2][i][1]->Add(eff[0][j / 2][j % 2][i][0]);
						eff[3][j / 2][j % 2][i][1]->Add(eff[1][j / 2][j % 2][i][0]);
						eff[3][j / 2][j % 2][i][1]->Add(eff[2][j / 2][j % 2][i][0]);
					}
					else
					{
						eff[l][j / 2][j % 2][i][1]->Divide(eff[l][j / 2][j % 2][i][0], eff[3][j / 2][j % 2][i][1], 1, 1, "B");
					}
				}
				
				TH1F* e = eff[l][j / 2][j % 2][i][1];
				e->SetTitle(EfficiencyTitles[j]);
				e->SetStats(kFALSE);
				e->SetMarkerColor(kBlack);
				e->SetLineWidth(1);
				e->SetLineColor(colorNums[(l == 2 ? (ConfigNumInputs * 2 + k) : (k * 2 + l)) % ColorCount]);
				e->SetLineStyle(ConfigDashedMarkers ? k + 1 : 1);
				e->SetMaximum(1.02);
				e->SetMinimum(-0.02);
				SetAxisSize(e);
				e->GetYaxis()->SetTitle("(Efficiency)");
				e->GetXaxis()->SetTitle(XAxisTitles[i]);
				e->Draw(k || l ? "same" : "");
				if (init == false && j == 0)
				{
					sprintf(name, "%s - %s", "HLT", EffNames[l]);
					legendeff[ii]->AddEntry(e, name, "l");
				}
				if (ii == 5) peff[ii][j]->SetLogx();
			}
			ceff[ii]->cd();
			ChangePadTitleSize(peff[ii][j], 0.056);				
		}
		legendeff[ii]->Draw();
		sprintf(fname, "eff_vs_%s.pdf", VSParameterNames[ii]);
		ceff[ii]->Print(fname);
	}

	init = true;
}
