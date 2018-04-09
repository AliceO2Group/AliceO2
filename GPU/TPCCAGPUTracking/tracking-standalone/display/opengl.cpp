#include "AliHLTTPCCADef.h"

#include <GL/glew.h>
#ifdef R__WIN32
#include <windows.h>
#include <winbase.h>
#include <windowsx.h>

HDC hDC = NULL;                                       // Private GDI Device Context
HGLRC hRC = NULL;                                     // Permanent Rendering Context
HWND hWnd = NULL;                                     // Holds Our Window Handle
HINSTANCE hInstance;                                  // Holds The Instance Of The Application
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM); // Declaration For WndProc

bool active = TRUE;     // Window Active Flag Set To TRUE By Default
bool fullscreen = TRUE; // Fullscreen Flag Set To Fullscreen Mode By Default

HANDLE semLockDisplay = NULL;

POINT mouseCursorPos;

volatile int mouseReset = false;
#else
#include "bitmapfile.h"
#include <GL/glx.h> // This includes the necessary X headers
#include <pthread.h>

Display *g_pDisplay = NULL;
Window g_window;
GLuint g_textureID = 0;

float g_fSpinX = 0.0f;
float g_fSpinY = 0.0f;
int g_nLastMousePositX = 0;
int g_nLastMousePositY = 0;
bool g_bMousing = false;

pthread_mutex_t semLockDisplay = PTHREAD_MUTEX_INITIALIZER;
#endif
#include <vector>
#include <array>
#include <tuple>
#include <GL/gl.h>
#include <GL/glu.h>

#include "AliHLTTPCCASliceData.h"
#include "AliHLTTPCCAStandaloneFramework.h"
#include "AliHLTTPCCATrack.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCATrackerFramework.h"
#include "AliHLTTPCGMMergedTrack.h"
#include "AliHLTTPCGMPropagator.h"
#include "include.h"
#include "../cmodules/timer.h"

#define fgkNSlices 36
#ifndef BUILD_QA
bool SuppressHit(int iHit) {return false;}
#endif
volatile static int needUpdate = 0;
void ShowNextEvent() {needUpdate = 1;}
#define GL_SCALE_FACTOR 50.f
int screenshot_scale = 1;

const int init_width = 1024, init_height = 768;

GLuint vbo_id;
typedef std::tuple<GLsizei, GLsizei, int> vboList;
struct GLvertex {GLfloat x, y, z; GLvertex(GLfloat a, GLfloat b, GLfloat c) : x(a), y(b), z(c) {}};
std::vector<GLvertex> vertexBuffer[fgkNSlices];
std::vector<GLint> vertexBufferStart[fgkNSlices];
std::vector<GLsizei> vertexBufferCount[fgkNSlices];
int drawCalls = 0;
inline void drawVertices(const vboList& v, const GLenum t)
{
	auto first = std::get<0>(v);
	auto count = std::get<1>(v);
	auto iSlice = std::get<2>(v);
	if (count == 0) return;
	drawCalls += count;
	glMultiDrawArrays(t, vertexBufferStart[iSlice].data() + first, vertexBufferCount[iSlice].data() + first, count);
	
	/*fprintf(stderr, "Draw start %d count %d: %d (size %lld)\n", (vertexBufferStart.data() + v.first)[0], (vertexBufferCount.data() + v.first)[0], v.second, (long long int) vertexBuffer.size());
	for (int k = vertexBufferStart.data()[v.first];k < vertexBufferStart.data()[v.first + v.second - 1] + vertexBufferCount.data()[v.first + v.second - 1];k++)
	{
		printf("Vertex %f %f %f\n", vertexBuffer[k].x, vertexBuffer[k].y, vertexBuffer[k].z);
	}*/
}
inline void insertVertexList(int iSlice, size_t first, size_t last)
{
	if (first == last) return;
	vertexBufferStart[iSlice].emplace_back(first);
	vertexBufferCount[iSlice].emplace_back(last - first);
}

bool keys[256] = {false}; // Array Used For The Keyboard Routine
bool keysShift[256] = {false}; //Shift held when key down

bool separateGlobalTracks = 0;
#define SEPERATE_GLOBAL_TRACKS_MAXID 5
#define TRACK_TYPE_ID_LIMIT 100
#define SEPERATE_GLOBAL_TRACKS_DISTINGUISH_TYPES 6
bool reorderFinalTracks = 0;

float rotateX = 0, rotateY = 0;
float mouseDnX, mouseDnY;
float mouseMvX, mouseMvY;
bool mouseDn = false;
bool mouseDnR = false;
int mouseWheel = 0;

volatile int buttonPressed = 0;
volatile int sendKey = 0;

GLfloat currentMatrice[16];

int drawClusters = true;
int drawLinks = false;
int drawSeeds = false;
int drawInitLinks = false;
int drawTracklets = false;
int drawTracks = false;
int drawGlobalTracks = false;
int drawFinal = false;

int drawSlice = -1;
int drawRelatedSlices = 0;
int drawGrid = 0;
int excludeClusters = 0;
int projectxy = 0;

int markClusters = 0;
int hideRejectedClusters = 1;
int hideUnmatchedClusters = 0;
int hideRejectedTracks = 1;

int propagateTracks = 0;
int colorCollisions = 0;
std::vector<std::array<int,37>> collisionClusters;
int nCollisions = 1;
int showCollision = -1;
void SetCollisionFirstCluster(unsigned int collision, int slice, int cluster)
{
	nCollisions = collision + 1;
	collisionClusters.resize(nCollisions);
	collisionClusters[collision][slice] = cluster;
}

float Xadd = 0;
float Zadd = 0;

float4 *globalPos = NULL;
int maxClusters = 0;
int currentClusters = 0;

volatile int displayEventNr = 0;
int currentEventNr = -1;

int glDLrecent = 0;
int updateDLList = 0;

float pointSize = 2.0;
float lineWidth = 1.5;

int animate = 0;

volatile int resetScene = 0;

inline void SetColorClusters() { if (colorCollisions) return; glColor3f(0, 0.7, 1.0); }
inline void SetColorInitLinks() { glColor3f(0.42, 0.4, 0.1); }
inline void SetColorLinks() { glColor3f(0.8, 0.2, 0.2); }
inline void SetColorSeeds() { glColor3f(0.8, 0.1, 0.85); }
inline void SetColorTracklets() { glColor3f(1, 1, 1); }
inline void SetColorTracks()
{
	if (separateGlobalTracks) glColor3f(1., 1., 0.15);
	else glColor3f(0.4, 1, 0);
}
inline void SetColorGlobalTracks()
{
	if (separateGlobalTracks) glColor3f(1., 0.15, 0.15);
	else glColor3f(1.0, 0.4, 0);
}
inline void SetColorFinal() { if (colorCollisions) return; glColor3f(0, 0.7, 0.2); }
inline void SetColorGrid() { glColor3f(0.7, 0.7, 0.0); }
inline void SetColorMarked() { glColor3f(1.0, 0.0, 0.0); }
inline void SetCollisionColor(int col)
{
	int red = (col * 2) % 5;
	int blue =  (2 + col * 3) % 7;
	int green = (4 + col * 5) % 6;
	if (red == 0 && blue == 0 && green == 0) red = 4;
	glColor3f(red / 4., green / 5., blue / 6.);
}	

void ReSizeGLScene(GLsizei width, GLsizei height) // Resize And Initialize The GL Window
{
	if (height == 0) // Prevent A Divide By Zero By
	{
		height = 1; // Making Height Equal One
	}

	static int init = 1;
	GLfloat tmp[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, tmp);

	glViewport(0, 0, width, height); // Reset The Current Viewport

	glMatrixMode(GL_PROJECTION); // Select The Projection Matrix
	glLoadIdentity();            // Reset The Projection Matrix

	// Calculate The Aspect Ratio Of The Window
	gluPerspective(45.0f, (GLfloat) width / (GLfloat) height, 0.1f, 1000.0f);

	glMatrixMode(GL_MODELVIEW); // Select The Modelview Matrix
	glLoadIdentity();           // Reset The Modelview Matrix

	if (init)
	{
		glTranslatef(0, 0, -16);
		init = 0;
	}
	else
	{
		glLoadMatrixf(tmp);
	}

	glGetFloatv(GL_MODELVIEW_MATRIX, currentMatrice);
}

int InitGL() // All Setup For OpenGL Goes Here
{
	glewInit();
	glGenBuffers(1, &vbo_id);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_id);
	glShadeModel(GL_SMOOTH);                           // Enable Smooth Shading
	glClearColor(0.0f, 0.0f, 0.0f, 0.5f);              // Black Background
	glClearDepth(1.0f);                                // Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);                           // Enables Depth Testing
	glDepthFunc(GL_LEQUAL);                            // The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST); // Really Nice Perspective Calculations
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	ReSizeGLScene(init_width, init_height);
	return (true);                                     // Initialization Went OK
}

inline void drawPointLinestrip(int iSlice, int cid, int id, int id_limit = TRACK_TYPE_ID_LIMIT)
{
	vertexBuffer[iSlice].emplace_back(globalPos[cid].x, globalPos[cid].y, projectxy ? 0 : globalPos[cid].z);
	if (globalPos[cid].w < id_limit) globalPos[cid].w = id;
}

vboList DrawClusters(AliHLTTPCCATracker &tracker, int select, unsigned int iCol)
{
	int iSlice = tracker.Param().ISlice();
	size_t startCount = vertexBufferStart[iSlice].size();
	size_t startCountInner = vertexBuffer[iSlice].size();
	for (int i = 0; i < tracker.Param().NRows(); i++)
	{
		const AliHLTTPCCARow &row = tracker.Data().Row(i);
		for (int j = 0; j < row.NHits(); j++)
		{
			const int cidInSlice = tracker.Data().ClusterDataIndex(row, j);
			const int cid = tracker.ClusterData()->Id(cidInSlice);
			if (hideUnmatchedClusters && SuppressHit(cid)) continue;
			if (nCollisions > 1)
			{
				unsigned int k = 0;
				while (k < collisionClusters.size() && collisionClusters[k][tracker.Param().ISlice()] < cidInSlice) k++;
				if (k != iCol) continue;
			}
			bool draw = globalPos[cid].w == select;
			if (markClusters)
			{
				const short flags = tracker.ClusterData()->Flags(tracker.Data().ClusterDataIndex(row, j));
				const bool match = flags & markClusters;
				draw = (select == 8) ? (match) : (draw && !match);
			}
			if (draw)
			{
				vertexBuffer[iSlice].emplace_back(globalPos[cid].x, globalPos[cid].y, projectxy ? 0 : globalPos[cid].z);
			}
		}
	}
	insertVertexList(tracker.Param().ISlice(), startCountInner, vertexBuffer[iSlice].size());
	return vboList(startCount, vertexBufferStart[iSlice].size() - startCount, iSlice);
}

vboList DrawLinks(AliHLTTPCCATracker &tracker, int id, bool dodown = false)
{
	int iSlice = tracker.Param().ISlice();
	size_t startCount = vertexBufferStart[iSlice].size();
	size_t startCountInner = vertexBuffer[iSlice].size();
	for (int i = 0; i < tracker.Param().NRows(); i++)
	{
		const AliHLTTPCCARow &row = tracker.Data().Row(i);

		if (i < tracker.Param().NRows() - 2)
		{
			const AliHLTTPCCARow &rowUp = tracker.Data().Row(i + 2);
			for (int j = 0; j < row.NHits(); j++)
			{
				if (tracker.Data().HitLinkUpData(row, j) != CALINK_INVAL)
				{
					const int cid1 = tracker.ClusterData()->Id(tracker.Data().ClusterDataIndex(row, j));
					const int cid2 = tracker.ClusterData()->Id(tracker.Data().ClusterDataIndex(rowUp, tracker.Data().HitLinkUpData(row, j)));
					drawPointLinestrip(iSlice, cid1, id);
					drawPointLinestrip(iSlice, cid2, id);
				}
			}
		}

		if (dodown && i >= 2)
		{
			const AliHLTTPCCARow &rowDown = tracker.Data().Row(i - 2);
			for (int j = 0; j < row.NHits(); j++)
			{
				if (tracker.Data().HitLinkDownData(row, j) != CALINK_INVAL)
				{
					const int cid1 = tracker.ClusterData()->Id(tracker.Data().ClusterDataIndex(row, j));
					const int cid2 = tracker.ClusterData()->Id(tracker.Data().ClusterDataIndex(rowDown, tracker.Data().HitLinkDownData(row, j)));
					drawPointLinestrip(iSlice, cid1, id);
					drawPointLinestrip(iSlice, cid2, id);
				}
				}
		}
	}
	insertVertexList(tracker.Param().ISlice(), startCountInner, vertexBuffer[iSlice].size());
	return vboList(startCount, vertexBufferStart[iSlice].size() - startCount, iSlice);
}

vboList DrawSeeds(AliHLTTPCCATracker &tracker)
{
	int iSlice = tracker.Param().ISlice();
	size_t startCount = vertexBufferStart[iSlice].size();
	for (int i = 0; i < *tracker.NTracklets(); i++)
	{
		const AliHLTTPCCAHitId &hit = tracker.TrackletStartHit(i);
		size_t startCountInner = vertexBuffer[iSlice].size();
		int ir = hit.RowIndex();
		calink ih = hit.HitIndex();
		do
		{
			const AliHLTTPCCARow &row = tracker.Data().Row(ir);
			const int cid = tracker.ClusterData()->Id(tracker.Data().ClusterDataIndex(row, ih));
			drawPointLinestrip(iSlice, cid, 3);
			ir += 2;
			ih = tracker.Data().HitLinkUpData(row, ih);
		} while (ih != CALINK_INVAL);
		insertVertexList(tracker.Param().ISlice(), startCountInner, vertexBuffer[iSlice].size());
	}
	return vboList(startCount, vertexBufferStart[iSlice].size() - startCount, iSlice);
}

vboList DrawTracklets(AliHLTTPCCATracker &tracker)
{
	int iSlice = tracker.Param().ISlice();
	size_t startCount = vertexBufferStart[iSlice].size();
	for (int i = 0; i < *tracker.NTracklets(); i++)
	{
		const AliHLTTPCCATracklet &tracklet = tracker.Tracklet(i);
		if (tracklet.NHits() == 0) continue;
		size_t startCountInner = vertexBuffer[iSlice].size();
		float4 oldpos;
		for (int j = tracklet.FirstRow(); j <= tracklet.LastRow(); j++)
		{
#ifdef EXTERN_ROW_HITS
			const calink rowHit = tracker.TrackletRowHits()[j * *tracker.NTracklets() + i];
#else
			const calink rowHit = tracklet.RowHit(j);
#endif
			if (rowHit != CALINK_INVAL)
			{
				const AliHLTTPCCARow &row = tracker.Data().Row(j);
				const int cid = tracker.ClusterData()->Id(tracker.Data().ClusterDataIndex(row, rowHit));
				/*if (j != tracklet.FirstRow())
				{
					float dist = (oldpos.x - globalPos[cid].x) * (oldpos.x - globalPos[cid].x) + (oldpos.y - globalPos[cid].y) * (oldpos.y - globalPos[cid].y) + (oldpos.z - globalPos[cid].z) * (oldpos.z - globalPos[cid].z);
				}*/
				oldpos = globalPos[cid];
				drawPointLinestrip(iSlice, cid, 4);
			}
		}
		insertVertexList(tracker.Param().ISlice(), startCountInner, vertexBuffer[iSlice].size());
	}
	return vboList(startCount, vertexBufferStart[iSlice].size() - startCount, iSlice);
}

vboList DrawTracks(AliHLTTPCCATracker &tracker, int global)
{
	int iSlice = tracker.Param().ISlice();
	size_t startCount = vertexBufferStart[iSlice].size();
	for (int i = (global ? tracker.CommonMemory()->fNLocalTracks : 0); i < (global ? *tracker.NTracks() : tracker.CommonMemory()->fNLocalTracks); i++)
	{
		AliHLTTPCCATrack &track = tracker.Tracks()[i];
		size_t startCountInner = vertexBuffer[iSlice].size();
		for (int j = 0; j < track.NHits(); j++)
		{
			const AliHLTTPCCAHitId &hit = tracker.TrackHits()[track.FirstHitID() + j];
			const AliHLTTPCCARow &row = tracker.Data().Row(hit.RowIndex());
			const int cid = tracker.ClusterData()->Id(tracker.Data().ClusterDataIndex(row, hit.HitIndex()));
			drawPointLinestrip(iSlice, cid, 5 + global);
		}
		insertVertexList(tracker.Param().ISlice(), startCountInner, vertexBuffer[iSlice].size());
	}
	return vboList(startCount, vertexBufferStart[iSlice].size() - startCount, iSlice);
}

vboList DrawFinal(AliHLTTPCCAStandaloneFramework &hlt, int iSlice, unsigned int iCol)
{
	size_t startCount = vertexBufferStart[iSlice].size();
	static AliHLTTPCGMPropagator prop;
	static bool initProp = true;
	if (initProp)
	{
		initProp = false;
		const float kRho = 1.025e-3;//0.9e-3;
		const float kRadLen = 29.532;//28.94;
		prop.SetMaxSinPhi( .999 );
		prop.SetMaterial( kRadLen, kRho );
		prop.SetPolynomialField( hlt.Merger().pField() );		
		prop.SetToyMCEventsFlag( hlt.Merger().SliceParam().ToyMCEventsFlag());
	}

	const AliHLTTPCGMMerger &merger = hlt.Merger();
	for (int i = 0; i < merger.NOutputTracks(); i++)
	{
		const AliHLTTPCGMMergedTrack &track = merger.OutputTracks()[i];
		if (track.NClusters() == 0) continue;
		int *clusterused = NULL;
		int bestk = 0;
		if (hideRejectedTracks && !track.OK()) continue;
		if (merger.Clusters()[track.FirstClusterRef() + track.NClusters() - 1].fSlice != iSlice) continue;
#ifdef BUILD_QA
		if (nCollisions > 1)
		{
			int label = GetMCLabel(i);
			if (label < -1) label = -label - 2;
			if (label != -1)
			{
				unsigned int k = 0;
				while (k < collisionClusters.size() && collisionClusters[k][36] < label) k++;
				if (k != iCol) continue;
			}
		}
#else
		if (iCol != 0) continue;
#endif
		size_t startCountInner = vertexBuffer[iSlice].size();
		
		if (reorderFinalTracks)
		{
			clusterused = new int[track.NClusters()];
			for (int j = 0; j < track.NClusters(); j++)
				clusterused[j] = 0;

			float smallest = 1e20;
			for (int k = 0; k < track.NClusters(); k++)
			{
				if (hideRejectedClusters && (merger.Clusters()[track.FirstClusterRef() + k].fState & AliHLTTPCGMMergedTrackHit::flagReject)) continue;
				int cid = merger.Clusters()[track.FirstClusterRef() + k].fId;
				float dist = globalPos[cid].x * globalPos[cid].x + globalPos[cid].y * globalPos[cid].y + globalPos[cid].z * globalPos[cid].z;
				if (dist < smallest)
				{
					smallest = dist;
					bestk = k;
				}
			}
		}
		else
		{
			while (hideRejectedClusters && (merger.Clusters()[track.FirstClusterRef() + bestk].fState & AliHLTTPCGMMergedTrackHit::flagReject)) bestk++;
		}

		int lastcid = merger.Clusters()[track.FirstClusterRef() + bestk].fId;
		if (reorderFinalTracks) clusterused[bestk] = 1;

		bool linestarted = (globalPos[lastcid].w < SEPERATE_GLOBAL_TRACKS_DISTINGUISH_TYPES);
		if (!separateGlobalTracks || linestarted)
		{
			drawPointLinestrip(iSlice, lastcid, 7, SEPERATE_GLOBAL_TRACKS_MAXID);
		}

		for (int j = (reorderFinalTracks ? 1 : (bestk + 1)); j < track.NClusters(); j++)
		{
			int bestcid = 0;
			if (reorderFinalTracks)
			{
				bestk = 0;
				float bestdist = 1e20;
				for (int k = 0; k < track.NClusters(); k++)
				{
					if (clusterused[k]) continue;
					if (hideRejectedClusters && (merger.Clusters()[track.FirstClusterRef() + k].fState & AliHLTTPCGMMergedTrackHit::flagReject)) continue;
					int cid = merger.Clusters()[track.FirstClusterRef() + k].fId;
					float dist = (globalPos[cid].x - globalPos[lastcid].x) * (globalPos[cid].x - globalPos[lastcid].x) +
					             (globalPos[cid].y - globalPos[lastcid].y) * (globalPos[cid].y - globalPos[lastcid].y) +
					             (globalPos[cid].z - globalPos[lastcid].z) * (globalPos[cid].z - globalPos[lastcid].z);
					if (dist < bestdist)
					{
						bestdist = dist;
						bestcid = cid;
						bestk = k;
					}
				}
				if (bestdist > 1e19) continue;
			}
			else
			{
				if (hideRejectedClusters && (merger.Clusters()[track.FirstClusterRef() + j].fState & AliHLTTPCGMMergedTrackHit::flagReject)) continue;
				bestcid = merger.Clusters()[track.FirstClusterRef() + j].fId;
			}
			if (separateGlobalTracks && !linestarted && globalPos[bestcid].w < SEPERATE_GLOBAL_TRACKS_DISTINGUISH_TYPES)
			{
				drawPointLinestrip(iSlice, lastcid, 7, SEPERATE_GLOBAL_TRACKS_MAXID);
				linestarted = true;
			}
			if (!separateGlobalTracks || linestarted)
			{
				drawPointLinestrip(iSlice, bestcid, 7, SEPERATE_GLOBAL_TRACKS_MAXID);
			}
			if (separateGlobalTracks && linestarted && !(globalPos[bestcid].w < SEPERATE_GLOBAL_TRACKS_DISTINGUISH_TYPES)) linestarted = false;
			if (reorderFinalTracks) clusterused[bestk] = 1;
			lastcid = bestcid;
		}
		if (propagateTracks)
		{
			AliHLTTPCGMTrackParam param = track.GetParam();
			float alpha = track.GetAlpha();
			prop.SetTrack(&param, alpha);
			bool inFlyDirection = 0;
			auto cl = merger.Clusters()[track.FirstClusterRef() + track.NClusters() - 1];
			alpha = hlt.Param().Alpha(cl.fSlice);
			float x = cl.fX - 1.;
			if (cl.fState & AliHLTTPCGMMergedTrackHit::flagReject) x = 0;
			while (x > 1.)
			{
				if (prop.PropagateToXAlpha( x, alpha, inFlyDirection ) ) break;
				if (fabs(param.SinPhi()) > 0.9) break;
				float4 ptr;
				hlt.Tracker().CPUTracker(cl.fSlice).Param().Slice2Global(param.X() + Xadd, param.Y(), param.Z(), &ptr.x, &ptr.y, &ptr.z);
				vertexBuffer[iSlice].emplace_back(ptr.x / GL_SCALE_FACTOR, ptr.y / GL_SCALE_FACTOR, projectxy ? 0 : (ptr.z + param.ZOffset()) / GL_SCALE_FACTOR);
				x -= 1;
			}

		}
		insertVertexList(iSlice, startCountInner, vertexBuffer[iSlice].size());
		if (reorderFinalTracks) delete[] clusterused;
	}
	return vboList(startCount, vertexBufferStart[iSlice].size() - startCount, iSlice);
}

vboList DrawGrid(AliHLTTPCCATracker &tracker)
{
	int iSlice = tracker.Param().ISlice();
	size_t startCount = vertexBufferStart[iSlice].size();
	size_t startCountInner = vertexBuffer[iSlice].size();
	for (int i = 0; i < tracker.Param().NRows(); i++)
	{
		const AliHLTTPCCARow &row = tracker.Data().Row(i);
		for (int j = 0; j <= (signed) row.Grid().Ny(); j++)
		{
			float z1 = row.Grid().ZMin();
			float z2 = row.Grid().ZMax();
			float x = row.X() + Xadd;
			float y = row.Grid().YMin() + (float) j / row.Grid().StepYInv();
			float zz1, zz2, yy1, yy2, xx1, xx2;
			tracker.Param().Slice2Global(x, y, z1, &xx1, &yy1, &zz1);
			tracker.Param().Slice2Global(x, y, z2, &xx2, &yy2, &zz2);
			if (zz1 >= 0)
			{
				zz1 += Zadd;
				zz2 += Zadd;
			}
			else
			{
				zz1 -= Zadd;
				zz2 -= Zadd;
			}
			vertexBuffer[iSlice].emplace_back(xx1 / GL_SCALE_FACTOR, yy1 / GL_SCALE_FACTOR, zz1 / GL_SCALE_FACTOR);
			vertexBuffer[iSlice].emplace_back(xx2 / GL_SCALE_FACTOR, yy2 / GL_SCALE_FACTOR, zz2 / GL_SCALE_FACTOR);
		}
		for (int j = 0; j <= (signed) row.Grid().Nz(); j++)
		{
			float y1 = row.Grid().YMin();
			float y2 = row.Grid().YMax();
			float x = row.X() + Xadd;
			float z = row.Grid().ZMin() + (float) j / row.Grid().StepZInv();
			float zz1, zz2, yy1, yy2, xx1, xx2;
			tracker.Param().Slice2Global(x, y1, z, &xx1, &yy1, &zz1);
			tracker.Param().Slice2Global(x, y2, z, &xx2, &yy2, &zz2);
			if (zz1 >= 0)
			{
				zz1 += Zadd;
				zz2 += Zadd;
			}
			else
			{
				zz1 -= Zadd;
				zz2 -= Zadd;
			}
			vertexBuffer[iSlice].emplace_back(xx1 / GL_SCALE_FACTOR, yy1 / GL_SCALE_FACTOR, zz1 / GL_SCALE_FACTOR);
			vertexBuffer[iSlice].emplace_back(xx2 / GL_SCALE_FACTOR, yy2 / GL_SCALE_FACTOR, zz2 / GL_SCALE_FACTOR);
		}
	}
	insertVertexList(tracker.Param().ISlice(), startCountInner, vertexBuffer[iSlice].size());
	return vboList(startCount, vertexBufferStart[iSlice].size() - startCount, iSlice);
}

int DrawGLScene(bool doAnimation = false) // Here's Where We Do All The Drawing
{
	static float fpsscale = 1;

	static int framesDone = 0, framesDoneFPS = 0;
	static HighResTimer timerFPS, timerDisplay, timerDraw;
	bool showTimer = false;

	constexpr const int N_POINTS_TYPE = 9;
	constexpr const int N_LINES_TYPE = 6;
	static vboList glDLlines[fgkNSlices][N_LINES_TYPE];
	static std::vector<vboList> glDLfinal[fgkNSlices];
	static std::vector<vboList> GLpoints[fgkNSlices][N_POINTS_TYPE];
	static vboList glDLgrid[fgkNSlices];

	AliHLTTPCCAStandaloneFramework &hlt = AliHLTTPCCAStandaloneFramework::Instance();
	
	//Initialize
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear Screen And Depth Buffer
	glLoadIdentity();                                   // Reset The Current Modelview Matrix

	int mouseWheelTmp = mouseWheel;
	mouseWheel = 0;

	//Calculate rotation / translation scaling factors
	float scalefactor = keys[16] ? 0.2 : 1.0;
	float rotatescalefactor = 1;

	float sqrdist = sqrt(sqrt(currentMatrice[12] * currentMatrice[12] + currentMatrice[13] * currentMatrice[13] + currentMatrice[14] * currentMatrice[14]) / GL_SCALE_FACTOR) * 0.8;
	if (sqrdist < 0.2) sqrdist = 0.2;
	if (sqrdist > 5) sqrdist = 5;
	scalefactor *= sqrdist;
	if (drawSlice != -1)
	{
		scalefactor /= 5;
		rotatescalefactor = 3;
	}

	//Perform new rotation / translation
	if (doAnimation)
	{
		float moveY = scalefactor * -0.14 * 0.25;
		float moveX = scalefactor * -0.14;
		static int nFrame = 0;
		nFrame++;
		float moveZ = 0;
		if (nFrame > 570)
		{
			moveZ = scalefactor * 1.;
		}
		else if (nFrame > 510)
		{
			moveZ = scalefactor * 1.f * (nFrame - 510) / 60.f;
		}
		glTranslatef(moveX, moveY, moveZ);
	}
	else
	{
		float moveZ = scalefactor * ((float) mouseWheelTmp / 150 + (float) (keys['W'] - keys['S']) * 0.2 * fpsscale);
		float moveX = scalefactor * ((float) (keys['A'] - keys['D']) * 0.2 * fpsscale);
		glTranslatef(moveX, 0, moveZ);
	}

	if (doAnimation)
	{
		glRotatef(scalefactor * rotatescalefactor * -0.5, 0, 1, 0);
		glRotatef(scalefactor * rotatescalefactor * 0.5 * 0.25, 1, 0, 0);
		glRotatef(scalefactor * 0.2, 0, 0, 1);
	}
	else if (mouseDnR && mouseDn)
	{
		glTranslatef(0, 0, -scalefactor * ((float) mouseMvY - (float) mouseDnY) / 4);
		glRotatef(scalefactor * ((float) mouseMvX - (float) mouseDnX), 0, 0, 1);
	}
	else if (mouseDnR)
	{
		glTranslatef(-scalefactor * 0.5 * ((float) mouseDnX - (float) mouseMvX) / 4, -scalefactor * 0.5 * ((float) mouseMvY - (float) mouseDnY) / 4, 0);
	}
	else if (mouseDn)
	{
		glRotatef(scalefactor * rotatescalefactor * ((float) mouseMvX - (float) mouseDnX), 0, 1, 0);
		glRotatef(scalefactor * rotatescalefactor * ((float) mouseMvY - (float) mouseDnY), 1, 0, 0);
	}
	if (keys['E'] ^ keys['F'])
	{
		glRotatef(scalefactor * fpsscale * 2, 0, 0, keys['E'] - keys['F']);
	}

	if (mouseDn || mouseDnR)
	{
#ifdef R__WIN32a
		mouseReset = true;
		SetCursorPos(mouseCursorPos.x, mouseCursorPos.y);
#else
		mouseDnX = mouseMvX;
		mouseDnY = mouseMvY;
#endif
	}

	//Apply standard translation / rotation
	glMultMatrixf(currentMatrice);

	//Graphichs Options
	lineWidth += (float) (keys['+']*keysShift['+'] - keys['-']*keysShift['-']) * fpsscale * 0.05;
	if (lineWidth < 0.01) lineWidth = 0.01;
	pointSize += (float) (keys['+']*(!keysShift['+']) - keys['-']*(!keysShift['-'])) * fpsscale * 0.05;
	if (pointSize < 0.01) pointSize = 0.01;
	
	//Reset position
	if (resetScene)
	{
		glLoadIdentity();
		glTranslatef(0, 0, -16);

		timerFPS.ResetStart();
		framesDone = framesDoneFPS = 0;

		pointSize = 2.0;
		drawSlice = -1;
		
		Xadd = 0;
		Zadd = 0;

		resetScene = 0;
		updateDLList = true;
	}

	//Store position
	glGetFloatv(GL_MODELVIEW_MATRIX, currentMatrice);

//Make sure event gets not overwritten during display
#ifdef R__WIN32
	WaitForSingleObject(semLockDisplay, INFINITE);
#else
	pthread_mutex_lock(&semLockDisplay);
#endif

	//Open GL Default Values
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glPointSize(pointSize);
	glLineWidth(lineWidth);

	//Extract global cluster information
	if (updateDLList || displayEventNr != currentEventNr)
	{
		showTimer = true;
		timerDraw.ResetStart();
		currentClusters = 0;
		for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
		{
			currentClusters += hlt.Tracker().CPUTracker(iSlice).NHitsTotal();
		}

		if (maxClusters < currentClusters)
		{
			if (globalPos) delete[] globalPos;
			maxClusters = currentClusters;
			globalPos = new float4[maxClusters];
		}

		for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
		{
			const AliHLTTPCCAClusterData &cdata = hlt.ClusterData(iSlice);
			for (int i = 0; i < cdata.NumberOfClusters(); i++)
			{
				const int cid = cdata.Id(i);
				if (cid >= maxClusters)
				{
					printf("Cluster Buffer Size exceeded (id %d max %d)\n", cid, maxClusters);
					exit(1);
				}
				float4 *ptr = &globalPos[cid];
				hlt.Tracker().CPUTracker(iSlice).Param().Slice2Global(cdata.X(i) + Xadd, cdata.Y(i), cdata.Z(i), &ptr->x, &ptr->y, &ptr->z);
				if (ptr->z >= 0)
				{
					ptr->z += Zadd;
					ptr->z += Zadd;
				}
				else
				{
					ptr->z -= Zadd;
					ptr->z -= Zadd;
				}

				ptr->x /= GL_SCALE_FACTOR;
				ptr->y /= GL_SCALE_FACTOR;
				ptr->z /= GL_SCALE_FACTOR;
				ptr->w = 1;
			}
		}

		currentEventNr = displayEventNr;

		timerFPS.ResetStart();
		framesDone = framesDoneFPS = 0;
		glDLrecent = 0;
		updateDLList = 0;
	}

	//Prepare Event
	if (!glDLrecent)
	{
		for (int i = 0;i < fgkNSlices;i++)
		{
			vertexBuffer[i].clear();
			vertexBufferStart[i].clear();
			vertexBufferCount[i].clear();
		}

		for (int i = 0; i < currentClusters; i++) globalPos[i].w = 0;

		for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
		{
			for (int i = 0;i < N_POINTS_TYPE;i++) GLpoints[iSlice][i].resize(nCollisions);
			glDLfinal[iSlice].resize(nCollisions);
		}
#pragma omp parallel
		{
#pragma omp for
			for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
			{
				AliHLTTPCCATracker &tracker = hlt.Tracker().CPUTracker(iSlice);
				if (drawInitLinks)
				{
					char *tmpMem;
					if (tracker.fLinkTmpMemory == NULL)
					{
						printf("Need to set TRACKER_KEEP_TEMPDATA for visualizing PreLinks!\n");
						continue;
					}
					tmpMem = tracker.Data().Memory();
					tracker.SetGPUSliceDataMemory((void *) tracker.fLinkTmpMemory, tracker.Data().Rows());
					tracker.SetPointersSliceData(tracker.ClusterData());
					glDLlines[iSlice][0] = DrawLinks(tracker, 1, true);
					tracker.SetGPUSliceDataMemory(tmpMem, tracker.Data().Rows());
					tracker.SetPointersSliceData(tracker.ClusterData());
				}
			}
#pragma omp barrier
#pragma omp for
			for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
			{
				AliHLTTPCCATracker &tracker = hlt.Tracker().CPUTracker(iSlice);

				glDLlines[iSlice][1] = DrawLinks(tracker, 2);
				glDLlines[iSlice][2] = DrawSeeds(tracker);
				glDLlines[iSlice][3] = DrawTracklets(tracker);
				glDLlines[iSlice][4] = DrawTracks(tracker, 0);
				glDLlines[iSlice][5] = DrawTracks(tracker, 1);
				glDLgrid[iSlice] = DrawGrid(tracker);

				for (int iCol = 0;iCol < nCollisions;iCol++)
				{
					glDLfinal[iSlice][iCol] = DrawFinal(hlt, iSlice, iCol);
				}
			}
#pragma omp barrier
#pragma omp for
			for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
			{
				AliHLTTPCCATracker &tracker = hlt.Tracker().CPUTracker(iSlice);
				for (int i = 0; i < N_POINTS_TYPE; i++)
				{
					for (int iCol = 0;iCol < nCollisions;iCol++)
					{
						GLpoints[iSlice][i][iCol] = DrawClusters(tracker, i, iCol);
					}
				}
			}
		}

		int errCode;
		if ((errCode = glGetError()) != GL_NO_ERROR)
		{
			printf("Error creating OpenGL display list: %s\n", gluErrorString(errCode));
			resetScene = 1;
		}
		else
		{
			glDLrecent = 1;
			size_t totalVertizes = 0;
			for (int i = 0;i < fgkNSlices;i++) totalVertizes += vertexBuffer[i].size();
			vertexBuffer[0].reserve(totalVertizes); //ATTENTION, this only reserves but not initializes, I.e. we use the vector as raw ptr for now...
			size_t totalYet = vertexBuffer[0].size();
			for (int i = 1;i < fgkNSlices;i++)
			{
				for (unsigned int j = 0;j < vertexBufferStart[i].size();j++)
				{
					vertexBufferStart[i][j] += totalYet;
				}
				memcpy(&vertexBuffer[0][totalYet], &vertexBuffer[i][0], vertexBuffer[i].size() * sizeof(vertexBuffer[i][0]));
				totalYet += vertexBuffer[i].size();
				vertexBuffer[i].clear();
			}
			glBufferData(GL_ARRAY_BUFFER, totalVertizes * sizeof(vertexBuffer[0][0]), vertexBuffer[0].data(), GL_STATIC_DRAW);
			vertexBuffer[0].clear();
		}
	}
	if (showTimer)
	{
		printf("Draw time: %d ms\n", (int) (timerDraw.GetCurrentElapsedTime() * 1000000.));
	}

	framesDone++;
	framesDoneFPS++;
	double time = timerFPS.GetCurrentElapsedTime();
	if (time > 1.)
	{
		float fps = (double) framesDoneFPS / time;
		printf("FPS: %f (%d frames, %d calls, Slice: %d, 1:Clusters %d, 2:Prelinks %d, 3:Links %d, 4:Seeds %d, 5:Tracklets %d, 6:Tracks %d, 7:GTracks %d, 8:Merger %d, C:Mark %X)\n",
			fps, framesDone, drawCalls, drawSlice, drawClusters, drawInitLinks, drawLinks, drawSeeds, drawTracklets, drawTracks, drawGlobalTracks, drawFinal, (int) markClusters);
		fpsscale = 60 / fps;
		timerFPS.ResetStart();
		framesDoneFPS = 0;
	}

	//Draw Event
	if (glDLrecent)
	{
		drawCalls = 0;
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3, GL_FLOAT, 0, 0);
		for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
		{
			if (drawSlice != -1)
			{
				if (!drawRelatedSlices && drawSlice != iSlice) continue;
				if (drawRelatedSlices && (drawSlice % 9) != (iSlice % 9)) continue;
			}

			if (drawGrid)
			{
				SetColorGrid();
				drawVertices(glDLgrid[iSlice], GL_LINES);
			}

			if (drawClusters)
			{
				SetColorClusters();
				for (int iCol = 0;iCol < nCollisions;iCol++)
				{
					if (showCollision != -1) iCol = showCollision;
					if (colorCollisions) SetCollisionColor(iCol);
					drawVertices(GLpoints[iSlice][0][iCol], GL_POINTS);

					if (drawInitLinks)
					{
						if (excludeClusters) goto skip1;
						SetColorInitLinks();
					}
					drawVertices(GLpoints[iSlice][1][iCol], GL_POINTS);

					if (drawLinks)
					{
						if (excludeClusters) goto skip1;
						SetColorLinks();
					}
					else
					{
						SetColorClusters();
					}
					drawVertices(GLpoints[iSlice][2][iCol], GL_POINTS);

					if (drawSeeds)
					{
						if (excludeClusters) goto skip1;
						SetColorSeeds();
					}
					drawVertices(GLpoints[iSlice][3][iCol], GL_POINTS);

				skip1:
					SetColorClusters();
					if (drawTracklets)
					{
						if (excludeClusters) goto skip2;
						SetColorTracklets();
					}
					drawVertices(GLpoints[iSlice][4][iCol], GL_POINTS);

					if (drawTracks)
					{
						if (excludeClusters) goto skip2;
						SetColorTracks();
					}
					drawVertices(GLpoints[iSlice][5][iCol], GL_POINTS);

				skip2:
					if (drawGlobalTracks)
					{
						if (excludeClusters) goto skip3;
						SetColorGlobalTracks();
					}
					else
					{
						SetColorClusters();
					}
					drawVertices(GLpoints[iSlice][6][iCol], GL_POINTS);

					if (drawFinal)
					{
						if (excludeClusters) goto skip3;
						SetColorFinal();
					}
					drawVertices(GLpoints[iSlice][7][iCol], GL_POINTS);
				skip3:;
					if (showCollision != -1) break;
				}
			}

			if (!excludeClusters)
			{
				if (drawInitLinks)
				{
					SetColorInitLinks();
					drawVertices(glDLlines[iSlice][0], GL_LINES);
				}
				if (drawLinks)
				{
					SetColorLinks();
					drawVertices(glDLlines[iSlice][1], GL_LINES);
				}
				if (drawSeeds)
				{
					SetColorSeeds();
					drawVertices(glDLlines[iSlice][2], GL_LINE_STRIP);
				}
				if (drawTracklets)
				{
					SetColorTracklets();
					drawVertices(glDLlines[iSlice][3], GL_LINE_STRIP);
				}
				if (drawTracks)
				{
					SetColorTracks();
					drawVertices(glDLlines[iSlice][4], GL_LINE_STRIP);
				}
				if (drawGlobalTracks)
				{
					SetColorGlobalTracks();
					drawVertices(glDLlines[iSlice][5], GL_LINE_STRIP);
				}
				if (drawFinal)
				{
					SetColorFinal();
					for (int iCol = 0;iCol < nCollisions;iCol++)
					{
						if (showCollision != -1) iCol = showCollision;
						if (colorCollisions) SetCollisionColor(iCol);
						if (!drawClusters) drawVertices(GLpoints[iSlice][7][iCol], GL_POINTS);

						drawVertices(glDLfinal[iSlice][iCol], GL_LINE_STRIP);
						if (showCollision != -1) break;
					}
				}
				if (markClusters)
				{
					SetColorMarked();
					for (int iCol = 0;iCol < nCollisions;iCol++)
					{
						if (showCollision != -1) iCol = showCollision;
						drawVertices(GLpoints[iSlice][8][iCol], GL_POINTS);
						if (showCollision != -1) break;
					}
				}
			}
		}
		glDisableClientState(GL_VERTEX_ARRAY);
	}

//Free event
#ifdef R__WIN32
	ReleaseSemaphore(semLockDisplay, 1, NULL);
#else
	pthread_mutex_unlock(&semLockDisplay);
#endif

	return true; // Keep Going
}

void DoScreenshot(char *filename, int SCALE_X, unsigned char **mixBuffer = NULL, float mixFactor = 0.)
{

//#define SCALE_X 3
#define SCALE_Y SCALE_X

	if (mixFactor < 0.f) mixFactor = 0.f;
	if (mixFactor > 1.f) mixFactor = 1.f;

	float tmpPointSize = pointSize;
	float tmpLineWidth = lineWidth;
	pointSize *= (float) (SCALE_X + SCALE_Y) / 2.;
	lineWidth *= (float) (SCALE_X + SCALE_Y) / 2.;

	GLint view[4], viewold[4];
	glGetIntegerv(GL_VIEWPORT, viewold);
	glGetIntegerv(GL_VIEWPORT, view);
	view[2] *= SCALE_X;
	view[3] *= SCALE_Y;
	unsigned char *pixels = (unsigned char *) malloc(4 * view[2] * view[3]);

	if (SCALE_X != 1 || SCALE_Y != 1)
	{
		memset(pixels, 0, 4 * view[2] * view[3]);
		unsigned char *pixels2 = (unsigned char *) malloc(4 * view[2] * view[3]);
		for (int i = 0; i < SCALE_X; i++)
		{
			for (int j = 0; j < SCALE_Y; j++)
			{
				glViewport(-i * viewold[2], -j * viewold[3], view[2], view[3]);

				DrawGLScene();
				glPixelStorei(GL_PACK_ALIGNMENT, 1);
				glReadBuffer(GL_BACK);
				glReadPixels(0, 0, view[2], view[3], GL_RGBA, GL_UNSIGNED_BYTE, pixels2);
				for (int k = 0; k < viewold[2]; k++)
				{
					for (int l = 0; l < viewold[3]; l++)
					{
						pixels[((j * viewold[3] + l) * view[2] + i * viewold[2] + k) * 4] = pixels2[(l * view[2] + k) * 4 + 2];
						pixels[((j * viewold[3] + l) * view[2] + i * viewold[2] + k) * 4 + 1] = pixels2[(l * view[2] + k) * 4 + 1];
						pixels[((j * viewold[3] + l) * view[2] + i * viewold[2] + k) * 4 + 2] = pixels2[(l * view[2] + k) * 4];
						pixels[((j * viewold[3] + l) * view[2] + i * viewold[2] + k) * 4 + 3] = 0;
					}
				}
			}
		}
		free(pixels2);
	}
	else
	{
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadBuffer(GL_BACK);
		glReadPixels(0, 0, view[2], view[3], GL_BGRA, GL_UNSIGNED_BYTE, pixels);
	}

	if (mixBuffer)
	{
		if (*mixBuffer == NULL)
		{
			*mixBuffer = (unsigned char *) malloc(4 * view[2] * view[3]);
			memcpy(*mixBuffer, pixels, 4 * view[2] * view[3]);
		}
		else
		{
			for (int i = 0; i < 4 * view[2] * view[3]; i++)
			{
				pixels[i] = (*mixBuffer)[i] = (mixFactor * pixels[i] + (1.f - mixFactor) * (*mixBuffer)[i]);
			}
		}
	}

	if (filename)
	{
		FILE *fp = fopen(filename, "w+b");

		BITMAPFILEHEADER bmpFH;
		BITMAPINFOHEADER bmpIH;
		memset(&bmpFH, 0, sizeof(bmpFH));
		memset(&bmpIH, 0, sizeof(bmpIH));

		bmpFH.bfType = 19778; //"BM"
		bmpFH.bfSize = sizeof(bmpFH) + sizeof(bmpIH) + 4 * view[2] * view[3];
		bmpFH.bfOffBits = sizeof(bmpFH) + sizeof(bmpIH);

		bmpIH.biSize = sizeof(bmpIH);
		bmpIH.biWidth = view[2];
		bmpIH.biHeight = view[3];
		bmpIH.biPlanes = 1;
		bmpIH.biBitCount = 32;
		bmpIH.biCompression = BI_RGB;
		bmpIH.biSizeImage = view[2] * view[3] * 4;
		bmpIH.biXPelsPerMeter = 5670;
		bmpIH.biYPelsPerMeter = 5670;

		fwrite(&bmpFH, 1, sizeof(bmpFH), fp);
		fwrite(&bmpIH, 1, sizeof(bmpIH), fp);
		fwrite(pixels, view[2] * view[3], 4, fp);
		fclose(fp);
	}
	free(pixels);

	glViewport(0, 0, viewold[2], viewold[3]);
	pointSize = tmpPointSize;
	lineWidth = tmpLineWidth;
	DrawGLScene();
}

void PrintHelp()
{
	printf("[n]/[SPACE]\tNext event\n");
	printf("[q]/[ESC]\tQuit\n");
	printf("[r]\t\tReset Display Settings\n");
	printf("[l]\t\tDraw single slice (next slice)\n");
	printf("[k]\t\tDraw single slice (previous slice)\n");
	printf("[J]\t\tDraw related slices (same plane in phi)\n");
	printf("[z]/[U]\t\tShow splitting of TPC in slices by extruding volume, [U] resets\n");
	printf("[y]\t\tStart Animation\n");
	printf("[g]\t\tDraw Grid\n");
	printf("[i]\t\tProject onto XY-plane\n");
	printf("[x]\t\tExclude Clusters used in the tracking steps enabled for visualization ([1]-[8])\n");
	printf("[<]\t\tExclude rejected tracks\n");
	printf("[c]\t\tMark flagged clusters (splitPad = 0x1, splitTime = 0x2, edge = 0x4, singlePad = 0x8, rejectDistance = 0x10, rejectErr = 0x20\n");
	printf("[C]\t\tColorcode clusters of different collisions\n");
	printf("[v]\t\tHide rejected clusters from tracks\n");
	printf("[b]\t\tHide all clusters not belonging or related to matched tracks\n");
	printf("[1]\t\tShow Clusters\n");
	printf("[2]\t\tShow Links that were removed\n");
	printf("[3]\t\tShow Links that remained in Neighbors Cleaner\n");
	printf("[4]\t\tShow Seeds (Start Hits)\n");
	printf("[5]\t\tShow Tracklets\n");
	printf("[6]\t\tShow Tracks (after Tracklet Selector)\n");
	printf("[7]\t\tShow Global Track Segments\n");
	printf("[8]\t\tShow Final Merged Tracks (after Track Merger)\n");
	printf("[j]\t\tShow global tracks as additional segments of final tracks\n");
	printf("[m]\t\tReorder clusters of merged tracks before showing them geometrically\n");
	printf("[t]\t\tTake Screenshot\n");
	printf("[o]\t\tSave current camera position\n");
	printf("[p]\t\tRestore camera position\n");
	printf("[h]\t\tPrint Help\n");
	printf("[w]/[s]/[a]/[d]\tZoom / Move Left Right\n");
	printf("[e]/[f]\t\tRotate\n");
	printf("[+]/[-]\t\tMake points thicker / fainter (Hold SHIFT for lines)\n");
	printf("[MOUSE1]\tLook around\n");
	printf("[MOUSE2]\tShift camera\n");
	printf("[MOUSE1+2]\tZoom / Rotate\n");
	printf("[SHIFT]\tSlow Zoom / Move / Rotate\n");
}

void HandleKeyRelease(int wParam)
{
	keys[wParam] = false;
	
	if (wParam >= 'A' && wParam <= 'Z')
	{
		if (keysShift[wParam]) wParam &= ~(int) ('a' ^ 'A');
		else wParam |= (int) ('a' ^ 'A');
	}

	if (wParam == 13 || wParam == 'n') buttonPressed = 1;
	else if (wParam == 'q') buttonPressed = 2;
	else if (wParam == 'r') resetScene = 1;

	else if (wParam == 'l')
	{
		if (drawSlice >= (drawRelatedSlices ? (fgkNSlices / 4 - 1) : (fgkNSlices - 1)))
			drawSlice = -1;
		else
			drawSlice++;
	}
	else if (wParam == 'k')
	{
		if (drawSlice <= -1)
			drawSlice = drawRelatedSlices ? (fgkNSlices / 4 - 1) : (fgkNSlices - 1);
		else
			drawSlice--;
	}
	else if (wParam == 'L')
	{
		if (showCollision >= nCollisions - 1)
			showCollision = -1;
		else
			showCollision++;
	}
	else if (wParam == 'K')
	{
		if (showCollision <= -1)
			showCollision = nCollisions - 1;
		else
			showCollision--;
	}
	else if (wParam == 'J')
	{
		drawRelatedSlices ^= 1;
	}
	else if (wParam == 'j')
	{
		separateGlobalTracks ^= 1;
		updateDLList = true;
	}
	else if (wParam == 'm')
	{
		reorderFinalTracks ^= 1;
		updateDLList = true;
	}
	else if (wParam == 'c')
	{
		if (markClusters == 0) markClusters = 1;
		else if (markClusters >= 8) markClusters = 0;
		else markClusters <<= 1;
		updateDLList = true;
	}
	else if (wParam == 'C')
	{
		colorCollisions ^= 1;
	}
	else if (wParam == 'P')
	{
		propagateTracks ^= 1;
		updateDLList = true;
	}
	else if (wParam == 'v')
	{
		hideRejectedClusters ^= 1;
		updateDLList = true;
	}
	else if (wParam == 'b')
	{
		hideUnmatchedClusters ^= 1;
		updateDLList = true;
	}
	else if (wParam == 'i')
	{
		updateDLList = true;
		projectxy ^= 1;
	}
	else if (wParam == 'z')
	{
		updateDLList = true;
		Xadd += 60;
		Zadd += 60;
	}
	else if (wParam == 'u')
	{
		updateDLList = true;
		Xadd -= 60;
		Zadd -= 60;
		if (Zadd < 0 || Xadd < 0) Zadd = Xadd = 0;
	}
	else if (wParam == 'y')
	{
		animate = 1;
		printf("Starting Animation\n");
	}

	else if (wParam == 'g') drawGrid ^= 1;
	else if (wParam == 'x') excludeClusters ^= 1;
	else if (wParam == '<')
	{
		hideRejectedTracks ^= 1;
		updateDLList = true;
	}

	else if (wParam == '1')
		drawClusters ^= 1;
	else if (wParam == '2')
	{
		drawInitLinks ^= 1;
		updateDLList = true;
	}
	else if (wParam == '3')
		drawLinks ^= 1;
	else if (wParam == '4')
		drawSeeds ^= 1;
	else if (wParam == '5')
		drawTracklets ^= 1;
	else if (wParam == '6')
		drawTracks ^= 1;
	else if (wParam == '7')
		drawGlobalTracks ^= 1;
	else if (wParam == '8')
		drawFinal ^= 1;
	else if (wParam == 't')
	{
		printf("Taking Screenshot\n");
		static int nScreenshot = 1;
		char fname[32];
		sprintf(fname, "screenshot%d.bmp", nScreenshot++);
		DoScreenshot(fname, screenshot_scale);
	}
	else if (wParam == 'o')
	{
		GLfloat tmp[16];
		glGetFloatv(GL_MODELVIEW_MATRIX, tmp);
		FILE *ftmp = fopen("glpos.tmp", "w+b");
		if (ftmp)
		{
			int retval = fwrite(&tmp[0], sizeof(GLfloat), 16, ftmp);
			if (retval != 16)
				printf("Error writing position to file\n");
			else
				printf("Position stored to file\n");
			fclose(ftmp);
		}
		else
		{
			printf("Error opening file\n");
		}
	}
	else if (wParam == 'p')
	{
		GLfloat tmp[16];
		FILE *ftmp = fopen("glpos.tmp", "rb");
		if (ftmp)
		{
			int retval = fread(&tmp[0], sizeof(GLfloat), 16, ftmp);
			if (retval == 16)
			{
				glLoadMatrixf(tmp);
				glGetFloatv(GL_MODELVIEW_MATRIX, currentMatrice);
				printf("Position read from file\n");
			}
			else
			{
				printf("Error reading position from file\n");
			}
			fclose(ftmp);
		}
		else
		{
			printf("Error opening file\n");
		}
	}
	else if (wParam == 'h')
	{
		PrintHelp();
	}
}

#ifdef R__WIN32

void KillGLWindow() // Properly Kill The Window
{
	if (fullscreen) // Are We In Fullscreen Mode?
	{
		ChangeDisplaySettings(NULL, 0); // If So Switch Back To The Desktop
		ShowCursor(TRUE);               // Show Mouse Pointer
	}

	if (hRC) // Do We Have A Rendering Context?
	{
		if (!wglMakeCurrent(NULL, NULL)) // Are We Able To Release The DC And RC Contexts?
		{
			MessageBox(NULL, "Release Of DC And RC Failed.", "SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
		}

		if (!wglDeleteContext(hRC)) // Are We Able To Delete The RC?
		{
			MessageBox(NULL, "Release Rendering Context Failed.", "SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
		}
		hRC = NULL; // Set RC To NULL
	}

	if (hDC && !ReleaseDC(hWnd, hDC)) // Are We Able To Release The DC
	{
		MessageBox(NULL, "Release Device Context Failed.", "SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
		hDC = NULL; // Set DC To NULL
	}

	if (hWnd && !DestroyWindow(hWnd)) // Are We Able To Destroy The Window?
	{
		MessageBox(NULL, "Could Not Release hWnd.", "SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
		hWnd = NULL; // Set hWnd To NULL
	}

	if (!UnregisterClass("OpenGL", hInstance)) // Are We Able To Unregister Class
	{
		MessageBox(NULL, "Could Not Unregister Class.", "SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
		hInstance = NULL; // Set hInstance To NULL
	}
}

BOOL CreateGLWindow(char *title, int width, int height, int bits, bool fullscreenflag)
{
	GLuint PixelFormat;                // Holds The Results After Searching For A Match
	WNDCLASS wc;                       // Windows Class Structure
	DWORD dwExStyle;                   // Window Extended Style
	DWORD dwStyle;                     // Window Style
	RECT WindowRect;                   // Grabs Rectangle Upper Left / Lower Right Values
	WindowRect.left = (long) 0;        // Set Left Value To 0
	WindowRect.right = (long) width;   // Set Right Value To Requested Width
	WindowRect.top = (long) 0;         // Set Top Value To 0
	WindowRect.bottom = (long) height; // Set Bottom Value To Requested Height

	fullscreen = fullscreenflag; // Set The Global Fullscreen Flag

	hInstance = GetModuleHandle(NULL);             // Grab An Instance For Our Window
	wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC; // Redraw On Size, And Own DC For Window.
	wc.lpfnWndProc = (WNDPROC) WndProc;            // WndProc Handles Messages
	wc.cbClsExtra = 0;                             // No Extra Window Data
	wc.cbWndExtra = 0;                             // No Extra Window Data
	wc.hInstance = hInstance;                      // Set The Instance
	wc.hIcon = LoadIcon(NULL, IDI_WINLOGO);        // Load The Default Icon
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);      // Load The Arrow Pointer
	wc.hbrBackground = NULL;                       // No Background Required For GL
	wc.lpszMenuName = NULL;                        // We Don't Want A Menu
	wc.lpszClassName = "OpenGL";                   // Set The Class Name

	if (!RegisterClass(&wc)) // Attempt To Register The Window Class
	{
		MessageBox(NULL, "Failed To Register The Window Class.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE; // Return FALSE
	}

	if (fullscreen) // Attempt Fullscreen Mode?
	{
		DEVMODE dmScreenSettings;                               // Device Mode
		memset(&dmScreenSettings, 0, sizeof(dmScreenSettings)); // Makes Sure Memory's Cleared
		dmScreenSettings.dmSize = sizeof(dmScreenSettings);     // Size Of The Devmode Structure
		dmScreenSettings.dmPelsWidth = width;                   // Selected Screen Width
		dmScreenSettings.dmPelsHeight = height;                 // Selected Screen Height
		dmScreenSettings.dmBitsPerPel = bits;                   // Selected Bits Per Pixel
		dmScreenSettings.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;

		if (ChangeDisplaySettings(&dmScreenSettings, CDS_FULLSCREEN) != DISP_CHANGE_SUCCESSFUL)
		{
			printf("The Requested Fullscreen Mode Is Not Supported By Your Video Card.\n");
			return(FALSE);
		}

		dwExStyle = WS_EX_APPWINDOW;
		dwStyle = WS_POPUP;
		ShowCursor(FALSE);
	}
	else
	{
		dwExStyle = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE; // Window Extended Style
		dwStyle = WS_OVERLAPPEDWINDOW;                  // Windows Style
	}

	AdjustWindowRectEx(&WindowRect, dwStyle, FALSE, dwExStyle); // Adjust Window To True Requested Size

	// Create The Window
	if (!(hWnd = CreateWindowEx(dwExStyle, "OpenGL", title, dwStyle | WS_CLIPSIBLINGS | WS_CLIPCHILDREN, 0, 0, WindowRect.right - WindowRect.left, WindowRect.bottom - WindowRect.top, NULL, NULL, hInstance, NULL)))
	{
		KillGLWindow();
		MessageBox(NULL, "Window Creation Error.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE;
	}

	static PIXELFORMATDESCRIPTOR pfd = // pfd Tells Windows How We Want Things To Be
	    {
	        sizeof(PIXELFORMATDESCRIPTOR), // Size Of This Pixel Format Descriptor
	        1,                             // Version Number
	        PFD_DRAW_TO_WINDOW |           // Format Must Support Window
	            PFD_SUPPORT_OPENGL |       // Format Must Support OpenGL
	            PFD_DOUBLEBUFFER,          // Must Support Double Buffering
	        PFD_TYPE_RGBA,                 // Request An RGBA Format
	        (unsigned char) bits,          // Select Our Color Depth
	        0,
	        0, 0, 0, 0, 0,  // Color Bits Ignored
	        0,              // No Alpha Buffer
	        0,              // Shift Bit Ignored
	        0,              // No Accumulation Buffer
	        0, 0, 0, 0,     // Accumulation Bits Ignored
	        16,             // 16Bit Z-Buffer (Depth Buffer)
	        0,              // No Stencil Buffer
	        0,              // No Auxiliary Buffer
	        PFD_MAIN_PLANE, // Main Drawing Layer
	        0,              // Reserved
	        0, 0, 0         // Layer Masks Ignored
	    };

	if (!(hDC = GetDC(hWnd))) // Did We Get A Device Context?
	{
		KillGLWindow(); // Reset The Display
		MessageBox(NULL, "Can't Create A GL Device Context.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE; // Return FALSE
	}

	if (!(PixelFormat = ChoosePixelFormat(hDC, &pfd))) // Did Windows Find A Matching Pixel Format?
	{
		KillGLWindow(); // Reset The Display
		MessageBox(NULL, "Can't Find A Suitable PixelFormat.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE; // Return FALSE
	}

	if (!SetPixelFormat(hDC, PixelFormat, &pfd)) // Are We Able To Set The Pixel Format?
	{
		KillGLWindow(); // Reset The Display
		MessageBox(NULL, "Can't Set The PixelFormat.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE; // Return FALSE
	}

	if (!(hRC = wglCreateContext(hDC))) // Are We Able To Get A Rendering Context?
	{
		KillGLWindow(); // Reset The Display
		MessageBox(NULL, "Can't Create A GL Rendering Context.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE; // Return FALSE
	}

	if (!wglMakeCurrent(hDC, hRC)) // Try To Activate The Rendering Context
	{
		KillGLWindow(); // Reset The Display
		MessageBox(NULL, "Can't Activate The GL Rendering Context.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE; // Return FALSE
	}

	ShowWindow(hWnd, SW_SHOW);    // Show The Window
	SetForegroundWindow(hWnd);    // Slightly Higher Priority
	SetFocus(hWnd);               // Sets Keyboard Focus To The Window
	ReSizeGLScene(width, height); // Set Up Our Perspective GL Screen

	if (!InitGL()) // Initialize Our Newly Created GL Window
	{
		KillGLWindow(); // Reset The Display
		printf("Initialization Failed.\n");
		return FALSE; // Return FALSE
	}

	return TRUE; // Success
}

int GetKey(int key)
{
	if (key == 107 || key == 187) return('+');
	if (key == 109 || key == 189) return('-');
	if (key >= 'a' && key <= 'z') key += 'A' - 'a';
	
	return(key);
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg) // Check For Windows Messages
	{
		case WM_ACTIVATE: // Watch For Window Activate Message
		{
			if (!HIWORD(wParam)) // Check Minimization State
			{
				active = TRUE; // Program Is Active
			}
			else
			{
				active = FALSE; // Program Is No Longer Active
			}

			return 0; // Return To The Message Loop
		}

		case WM_SYSCOMMAND: // Intercept System Commands
		{
			switch (wParam) // Check System Calls
			{
			case SC_SCREENSAVE:   // Screensaver Trying To Start?
			case SC_MONITORPOWER: // Monitor Trying To Enter Powersave?
				return 0;         // Prevent From Happening
			}
			break; // Exit
		}

		case WM_CLOSE: // Did We Receive A Close Message?
		{
			PostQuitMessage(0); // Send A Quit Message
			return 0;           // Jump Back
		}

		case WM_KEYDOWN: // Is A Key Being Held Down?
		{
			wParam = GetKey(wParam);
			keys[wParam] = TRUE; // If So, Mark It As TRUE
			keysShift[wParam] = keys[16];
			return 0;            // Jump Back
		}

		case WM_KEYUP: // Has A Key Been Released?
		{
			wParam = GetKey(wParam);
			HandleKeyRelease(wParam);
			keysShift[wParam] = false;

			printf("Key: %d\n", wParam);
			return 0; // Jump Back
		}

		case WM_SIZE: // Resize The OpenGL Window
		{
			ReSizeGLScene(LOWORD(lParam), HIWORD(lParam)); // LoWord=Width, HiWord=Height
			return 0;                                      // Jump Back
		}

		case WM_LBUTTONDOWN:
		{
			mouseDnX = GET_X_LPARAM(lParam);
			mouseDnY = GET_Y_LPARAM(lParam);
			mouseDn = true;
			GetCursorPos(&mouseCursorPos);
			return 0;
		}

		case WM_LBUTTONUP:
		{
			mouseDn = false;
			return 0;
		}

		case WM_RBUTTONDOWN:
		{
			mouseDnX = GET_X_LPARAM(lParam);
			mouseDnY = GET_Y_LPARAM(lParam);
			mouseDnR = true;
			GetCursorPos(&mouseCursorPos);
			return 0;
		}

		case WM_RBUTTONUP:
		{
			mouseDnR = false;
			return 0;
		}

		case WM_MOUSEMOVE:
		{
			if (mouseReset)
			{
				mouseDnX = GET_X_LPARAM(lParam);
				mouseDnY = GET_Y_LPARAM(lParam);
				mouseReset = 0;
			}
			mouseMvX = GET_X_LPARAM(lParam);
			mouseMvY = GET_Y_LPARAM(lParam);
			return 0;
		}

		case WM_MOUSEWHEEL:
		{
			mouseWheel += GET_WHEEL_DELTA_WPARAM(wParam);
			return 0;
		}
	}

	// Pass All Unhandled Messages To DefWindowProc
	return DefWindowProc(hWnd, uMsg, wParam, lParam);
}

DWORD WINAPI OpenGLMain(LPVOID tmp)
{
	MSG msg;           // Windows Message Structure
	BOOL done = FALSE; // Bool Variable To Exit Loop

	// Ask The User Which Screen Mode They Prefer
	fullscreen = FALSE; // Windowed Mode

	// Create Our OpenGL Window
	if (!CreateGLWindow("Alice HLT TPC CA Event Display", init_width, init_height, 32, fullscreen))
	{
		return 0; // Quit If Window Was Not Created
	}

	while (!done) // Loop That Runs While done=FALSE
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) // Is There A Message Waiting?
		{
			if (msg.message == WM_QUIT) // Have We Received A Quit Message?
			{
				done = TRUE; // If So done=TRUE
			}
			else // If Not, Deal With Window Messages
			{
				TranslateMessage(&msg); // Translate The Message
				DispatchMessage(&msg);  // Dispatch The Message
			}
		}
		else // If There Are No Messages
		{
			// Draw The Scene.  Watch For ESC Key And Quit Messages From DrawGLScene()
			if (active) // Program Active?
			{
				if (keys[VK_ESCAPE]) // Was ESC Pressed?
				{
					done = TRUE; // ESC Signalled A Quit
				}
				else // Not Time To Quit, Update Screen
				{
					if (animate)
					{
						static int nFrame = 0;

						DrawGLScene(true);
						char filename[16];
						sprintf(filename, "video%05d.bmp", nFrame++);
						unsigned char *mixBuffer = NULL;
						drawClusters = nFrame < 240;
						drawSeeds = nFrame >= 90 && nFrame < 210;
						drawTracklets = nFrame >= 210 && nFrame < 300;
						pointSize = nFrame >= 90 ? 1.0 : 2.0;
						drawTracks = nFrame >= 300 && nFrame < 390;
						drawFinal = nFrame >= 390;
						drawGlobalTracks = nFrame >= 480;
						DoScreenshot(NULL, 1, &mixBuffer);

						drawClusters = nFrame < 210;
						drawSeeds = nFrame > 60 && nFrame < 180;
						drawTracklets = nFrame >= 180 && nFrame < 270;
						pointSize = nFrame > 60 ? 1.0 : 2.0;
						drawTracks = nFrame > 270 && nFrame < 360;
						drawFinal = nFrame > 360;
						drawGlobalTracks = nFrame > 450;
						DoScreenshot(filename, 1, &mixBuffer, (float) (nFrame % 30) / 30.f);

						free(mixBuffer);
						printf("Wrote video frame %s\n\n", filename);
					}
					else
					{
						DrawGLScene(); // Draw The Scene
					}
					SwapBuffers(hDC); // Swap Buffers (Double Buffering)
				}
			}
		}
	}

	// Shutdown
	KillGLWindow();      // Kill The Window
	return (msg.wParam); // Exit The Program
}

#else

int GetKey(int key)
{
	if (key == 65453 || key == 45) return('-');
	if (key == 65451 || key == 43) return('+');
	if (key == 65505) return(16); //Shift
	if (key == 65307) return('Q'); //ESC
	if (key == 32) return(13); //Space
	if (key > 255) return(0);
	
	if (key >= 'a' && key <= 'z') key += 'A' - 'a';
	
	return(key);
}

void *OpenGLMain(void *ptr)
{
	XSetWindowAttributes windowAttributes;
	XVisualInfo *visualInfo = NULL;
	XEvent event;
	Colormap colorMap;
	GLXContext glxContext;
	int errorBase;
	int eventBase;

	// Open a connection to the X server
	g_pDisplay = XOpenDisplay(NULL);

	if (g_pDisplay == NULL)
	{
		fprintf(stderr, "glxsimple: %s\n", "could not open display");
		exit(1);
	}

	// Make sure OpenGL's GLX extension supported
	if (!glXQueryExtension(g_pDisplay, &errorBase, &eventBase))
	{
		fprintf(stderr, "glxsimple: %s\n", "X server has no OpenGL GLX extension");
		exit(1);
	}

	// Find an appropriate visual

	int doubleBufferVisual[] =
	    {
	        GLX_RGBA,           // Needs to support OpenGL
	        GLX_DEPTH_SIZE, 16, // Needs to support a 16 bit depth buffer
	        GLX_DOUBLEBUFFER,   // Needs to support double-buffering
	        None                // end of list
	    };

	// Try for the double-bufferd visual first
	visualInfo = glXChooseVisual(g_pDisplay, DefaultScreen(g_pDisplay), doubleBufferVisual);
	if (visualInfo == NULL)
	{
		fprintf(stderr, "glxsimple: %s\n", "no RGB visual with depth buffer");
		exit(1);
	}

	// Create an OpenGL rendering context
	glxContext = glXCreateContext(g_pDisplay,
	                              visualInfo,
	                              NULL,     // No sharing of display lists
	                              GL_TRUE); // Direct rendering if possible

	if (glxContext == NULL)
	{
		fprintf(stderr, "glxsimple: %s\n", "could not create rendering context");
		exit(1);
	}
	
	Window win = RootWindow(g_pDisplay, visualInfo->screen);

	// Create an X colormap since we're probably not using the default visual
	colorMap = XCreateColormap(g_pDisplay,
	                           win,
	                           visualInfo->visual,
	                           AllocNone);

	windowAttributes.colormap = colorMap;
	windowAttributes.border_pixel = 0;
	windowAttributes.event_mask = ExposureMask |
	                              VisibilityChangeMask |
	                              KeyPressMask |
	                              KeyReleaseMask |
	                              ButtonPressMask |
	                              ButtonReleaseMask |
	                              PointerMotionMask |
	                              StructureNotifyMask |
	                              SubstructureNotifyMask |
	                              FocusChangeMask;

	// Create an X window with the selected visual
	g_window = XCreateWindow(g_pDisplay,
	                         win,
	                         50, 50,     // x/y position of top-left outside corner of the window
	                         init_width, init_height, // Width and height of window
	                         0,        // Border width
	                         visualInfo->depth,
	                         InputOutput,
	                         visualInfo->visual,
	                         CWBorderPixel | CWColormap | CWEventMask,
	                         &windowAttributes);

	XSetStandardProperties(g_pDisplay,
	                       g_window,
	                       "AliHLTTPCCA Online Event Display",
	                       "AliHLTTPCCA Online Event Display",
	                       None,
	                       NULL,
	                       0,
	                       NULL);

	// Bind the rendering context to the window
	glXMakeCurrent(g_pDisplay, g_window, glxContext);

	// Request the X window to be displayed on the screen
	XMapWindow(g_pDisplay, g_window);
	
	XEvent xev;
	Atom wm_state  =  XInternAtom(g_pDisplay, "_NET_WM_STATE", False);
	Atom max_horz  =  XInternAtom(g_pDisplay, "_NET_WM_STATE_MAXIMIZED_HORZ", False);
	Atom max_vert  =  XInternAtom(g_pDisplay, "_NET_WM_STATE_MAXIMIZED_VERT", False);
	memset(&xev, 0, sizeof(xev));
	xev.type = ClientMessage;
	xev.xclient.window = g_window;
	xev.xclient.message_type = wm_state;
	xev.xclient.format = 32;
	xev.xclient.data.l[0] = 1; //_NET_WM_STATE_ADD
	xev.xclient.data.l[1] = max_horz;
	xev.xclient.data.l[2] = max_vert;
	XSendEvent(g_pDisplay, DefaultRootWindow(g_pDisplay), False, SubstructureNotifyMask, &xev);
	
	Atom WM_DELETE_WINDOW = XInternAtom(g_pDisplay, "WM_DELETE_WINDOW", False); 
    XSetWMProtocols(g_pDisplay, g_window, &WM_DELETE_WINDOW, 1);

	// Init OpenGL...
	InitGL();

	// Enter the render loop and don't forget to dispatch X events as they occur.
	
	XMapWindow(g_pDisplay, g_window);
	XFlush(g_pDisplay);
	int x11_fd = ConnectionNumber(g_pDisplay);

	while (1)
	{
		int num_ready_fds;
		struct timeval tv;
		fd_set in_fds;
		int waitCount = 0;
		do
		{
			FD_ZERO(&in_fds);
			FD_SET(x11_fd, &in_fds);
			tv.tv_usec = 10000;
			tv.tv_sec = 0;
			num_ready_fds = XPending(g_pDisplay) || select(x11_fd + 1, &in_fds, NULL, NULL, &tv);
			if (num_ready_fds < 0)
			{
				printf("Error\n");
			}
			else if (num_ready_fds > 0) needUpdate = 0;
			if (buttonPressed == 2) break;
			if (sendKey) needUpdate = 1;
			if (waitCount++ != 100) needUpdate = 1;
		} while (!(num_ready_fds || needUpdate));
		
		do
		{
			//XNextEvent(g_pDisplay, &event);
			if (needUpdate)
			{
				needUpdate = 0;
				event.type = Expose;
			}
			else
			{
				XNextEvent(g_pDisplay, &event);
			}
			if (buttonPressed == 2) break;
			
			switch (event.type)
			{
				case ButtonPress:
				{
					if (event.xbutton.button == 1)
					{
						mouseDn = true;
					}
					if (event.xbutton.button != 1)
					{
						mouseDnR = true;
					}
					mouseDnX = event.xmotion.x;
					mouseDnY = event.xmotion.y;
				}
				break;

				case ButtonRelease:
				{
					if (event.xbutton.button == 1)
					{
						mouseDn = false;
					}
					if (event.xbutton.button != 1)
					{
						mouseDnR = false;
					}
				}
				break;

				case KeyPress:
				{
					KeySym sym = XLookupKeysym(&event.xkey, 0);
					int wParam = GetKey(sym);
					//fprintf(stderr, "KeyPress event %d --> %d (%c) -> %d (%c), %d\n", event.xkey.keycode, (int) sym, (char) (sym > 27 ? sym : ' '), wParam, (char) wParam, (int) keys[16]);
					keys[wParam] = true;
					keysShift[wParam] = keys[16];
				}
				break;

				case KeyRelease:
				{
					KeySym sym = XLookupKeysym(&event.xkey, 0);
					int wParam = GetKey(sym);
					//fprintf(stderr, "KeyRelease event %d -> %d (%c) -> %d (%c), %d\n", event.xkey.keycode, (int) sym, (char) (sym > 27 ? sym : ' '), wParam, (char) wParam, (int) keysShift[wParam]);
					HandleKeyRelease(wParam);
					keysShift[wParam] = false;
					//if (updateDLList) printf("Need update!!\n");
				}
				break;

				case MotionNotify:
				{
					mouseMvX = event.xmotion.x;
					mouseMvY = event.xmotion.y;
				}
				break;

				case Expose:
				{
				}
				break;

				case ConfigureNotify:
				{
					glViewport(0, 0, event.xconfigure.width, event.xconfigure.height);
					ReSizeGLScene(event.xconfigure.width, event.xconfigure.height);
				}
				break;
				
				case ClientMessage:
				{
					buttonPressed = 2;
				}
				break;
			}
			
			if (sendKey)
			{
				//fprintf(stderr, "sendKey %d '%c'\n", sendKey, (char) sendKey);
				if (sendKey >= 'a' && sendKey <= 'z') sendKey ^= 'a' ^ 'A';
				HandleKeyRelease(sendKey);
				sendKey = 0;
			}
		} while (XPending(g_pDisplay)); // Loop to compress events
		if (buttonPressed == 2) break;

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		DrawGLScene();
		glXSwapBuffers(g_pDisplay, g_window); // Buffer swap does implicit glFlush
	}
	return(NULL);
}

#endif
