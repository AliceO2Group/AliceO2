#include "AliHLTTPCCADef.h"

#ifdef R__WIN32
#include <winbase.h>
#include <windows.h> // Header File For Windows
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
bool g_bDoubleBuffered = GL_TRUE;
GLuint g_textureID = 0;

float g_fSpinX = 0.0f;
float g_fSpinY = 0.0f;
int g_nLastMousePositX = 0;
int g_nLastMousePositY = 0;
bool g_bMousing = false;

int screenshot_scale = 1;

pthread_mutex_t semLockDisplay = PTHREAD_MUTEX_INITIALIZER;
#endif
#include <GL/gl.h>  // Header File For The OpenGL32 Library
#include <GL/glu.h> // Header File For The GLu32 Library

#include "AliHLTTPCCASliceData.h"
#include "AliHLTTPCCAStandaloneFramework.h"
#include "AliHLTTPCCATrack.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCATrackerFramework.h"
#include "AliHLTTPCGMMergedTrack.h"
#include "include.h"
#include "../cmodules/timer.h"

#define fgkNSlices 36

bool keys[256]; // Array Used For The Keyboard Routine

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
int drawGrid = 0;
int excludeClusters = 0;
int projectxy = 0;

int markClusters = 0;
int hideRejectedClusters = 1;
int hideUnmatchedClusters = 0;
int hideRejectedTracks = 1;

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

inline void SetColorClusters() { glColor3f(0, 0.7, 1.0); }
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
inline void SetColorFinal() { glColor3f(0, 0.7, 0.2); }
inline void SetColorGrid() { glColor3f(0.7, 0.7, 0.0); }
inline void SetColorMarked() { glColor3f(1.0, 0.0, 0.0); }

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
	gluPerspective(45.0f, (GLfloat) width / (GLfloat) height, 0.1f, 100.0f);

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
	glShadeModel(GL_SMOOTH);                           // Enable Smooth Shading
	glClearColor(0.0f, 0.0f, 0.0f, 0.5f);              // Black Background
	glClearDepth(1.0f);                                // Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);                           // Enables Depth Testing
	glDepthFunc(GL_LEQUAL);                            // The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST); // Really Nice Perspective Calculations
	return (true);                                     // Initialization Went OK
}

inline void drawPointLinestrip(int cid, int id, int id_limit = TRACK_TYPE_ID_LIMIT)
{
	glVertex3f(globalPos[cid].x, globalPos[cid].y, projectxy ? 0 : globalPos[cid].z);
	if (globalPos[cid].w < id_limit) globalPos[cid].w = id;
}

void DrawClusters(AliHLTTPCCATracker &tracker, int select)
{
	glBegin(GL_POINTS);
	for (int i = 0; i < tracker.Param().NRows(); i++)
	{
		const AliHLTTPCCARow &row = tracker.Data().Row(i);
		for (int j = 0; j < row.NHits(); j++)
		{
			const int cid = tracker.ClusterData()->Id(tracker.Data().ClusterDataIndex(row, j));
			if (hideUnmatchedClusters && SuppressHit(cid)) continue;
			bool draw = globalPos[cid].w == select;
			if (markClusters)
			{
				const short flags = tracker.ClusterData()->Flags(tracker.Data().ClusterDataIndex(row, j));
				const bool match = flags & markClusters;
				draw = (select == 8) ? (match) : (draw && !match);
			}
			if (draw)
			{
				glVertex3f(globalPos[cid].x, globalPos[cid].y, projectxy ? 0 : globalPos[cid].z);
			}
		}
	}
	glEnd();
}

void DrawLinks(AliHLTTPCCATracker &tracker, int id, bool dodown = false)
{
	glBegin(GL_LINES);
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
					drawPointLinestrip(cid1, id);
					drawPointLinestrip(cid2, id);
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
					drawPointLinestrip(cid1, id);
					drawPointLinestrip(cid2, id);
				}
			}
		}
	}
	glEnd();
}

void DrawSeeds(AliHLTTPCCATracker &tracker)
{
	for (int i = 0; i < *tracker.NTracklets(); i++)
	{
		const AliHLTTPCCAHitId &hit = tracker.TrackletStartHit(i);
		glBegin(GL_LINE_STRIP);
		int ir = hit.RowIndex();
		calink ih = hit.HitIndex();
		do
		{
			const AliHLTTPCCARow &row = tracker.Data().Row(ir);
			const int cid = tracker.ClusterData()->Id(tracker.Data().ClusterDataIndex(row, ih));
			drawPointLinestrip(cid, 3);
			ir += 2;
			ih = tracker.Data().HitLinkUpData(row, ih);
		} while (ih != CALINK_INVAL);
		glEnd();
	}
}

void DrawTracklets(AliHLTTPCCATracker &tracker)
{
	for (int i = 0; i < *tracker.NTracklets(); i++)
	{
		const AliHLTTPCCATracklet &tracklet = tracker.Tracklet(i);
		if (tracklet.NHits() == 0) continue;
		glBegin(GL_LINE_STRIP);
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
				drawPointLinestrip(cid, 4);
			}
		}
		glEnd();
	}
}

void DrawTracks(AliHLTTPCCATracker &tracker, int global)
{
	for (int i = (global ? tracker.CommonMemory()->fNLocalTracks : 0); i < (global ? *tracker.NTracks() : tracker.CommonMemory()->fNLocalTracks); i++)
	{
		AliHLTTPCCATrack &track = tracker.Tracks()[i];
		glBegin(GL_LINE_STRIP);
		for (int j = 0; j < track.NHits(); j++)
		{
			const AliHLTTPCCAHitId &hit = tracker.TrackHits()[track.FirstHitID() + j];
			const AliHLTTPCCARow &row = tracker.Data().Row(hit.RowIndex());
			const int cid = tracker.ClusterData()->Id(tracker.Data().ClusterDataIndex(row, hit.HitIndex()));
			drawPointLinestrip(cid, 5 + global);
		}
		glEnd();
	}
}

void DrawFinal(AliHLTTPCCAStandaloneFramework &hlt)
{
	const AliHLTTPCGMMerger &merger = hlt.Merger();
	for (int i = 0; i < merger.NOutputTracks(); i++)
	{
		const AliHLTTPCGMMergedTrack &track = merger.OutputTracks()[i];
		if (track.NClusters() == 0) continue;
		int *clusterused = NULL;
		int bestk = 0;
		if (hideRejectedTracks && !track.OK()) continue;
		glBegin(GL_LINE_STRIP);
		
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
			drawPointLinestrip(lastcid, 7, SEPERATE_GLOBAL_TRACKS_MAXID);
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
				drawPointLinestrip(lastcid, 7, SEPERATE_GLOBAL_TRACKS_MAXID);
				linestarted = true;
			}
			if (!separateGlobalTracks || linestarted)
			{
				drawPointLinestrip(bestcid, 7, SEPERATE_GLOBAL_TRACKS_MAXID);
			}
			if (separateGlobalTracks && linestarted && !(globalPos[bestcid].w < SEPERATE_GLOBAL_TRACKS_DISTINGUISH_TYPES)) linestarted = false;
			if (reorderFinalTracks) clusterused[bestk] = 1;
			lastcid = bestcid;
		}
		glEnd();
		if (reorderFinalTracks) delete[] clusterused;
	}
}

void DrawGrid(AliHLTTPCCATracker &tracker)
{
	glBegin(GL_LINES);
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
			glVertex3f(xx1 / 50, yy1 / 50, zz1 / 50);
			glVertex3f(xx2 / 50, yy2 / 50, zz2 / 50);
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
			glVertex3f(xx1 / 50, yy1 / 50, zz1 / 50);
			glVertex3f(xx2 / 50, yy2 / 50, zz2 / 50);
		}
	}
	glEnd();
}

int DrawGLScene(bool doAnimation = false) // Here's Where We Do All The Drawing
{
	static float fpsscale = 1;

	static int framesDone = 0, framesDoneFPS = 0;
	static HighResTimer timerFPS, timerDisplay;

	constexpr const int N_POINTS_TYPE = 9;
	constexpr const int N_LINES_TYPE = 6;
	static GLuint glDLlines[fgkNSlices][N_LINES_TYPE];
	static GLuint glDLlinesFinal;
	static GLuint glDLpoints[fgkNSlices][N_POINTS_TYPE];
	static GLuint glDLgrid[fgkNSlices];
	static int glDLcreated = 0;

	AliHLTTPCCAStandaloneFramework &hlt = AliHLTTPCCAStandaloneFramework::Instance();

	//Initialize
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear Screen And Depth Buffer
	glLoadIdentity();                                   // Reset The Current Modelview Matrix

	int mouseWheelTmp = mouseWheel;
	mouseWheel = 0;

	//Calculate rotation / translation scaling factors
	float scalefactor = keys[16] ? 0.2 : 1.0;
	float rotatescalefactor = 1;

	float sqrdist = sqrt(sqrt(currentMatrice[12] * currentMatrice[12] + currentMatrice[13] * currentMatrice[13] + currentMatrice[14] * currentMatrice[14]) / 50) * 0.8;
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
	pointSize += (float) (keys[107] - keys[109] + keys[187] - keys[189]) * fpsscale * 0.05;
	if (pointSize <= 0.01) pointSize = 0.01;

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

				ptr->x /= 50;
				ptr->y /= 50;
				ptr->z /= 50;
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
		for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
		{
			if (glDLcreated)
			{
				for (int i = 0; i < N_LINES_TYPE; i++)
					glDeleteLists(glDLlines[iSlice][i], 1);
				for (int i = 0; i < N_POINTS_TYPE; i++)
					glDeleteLists(glDLpoints[iSlice][i], 1);
				glDeleteLists(glDLgrid[iSlice], 1);
			}
			else
			{
				for (int i = 0; i < N_LINES_TYPE; i++)
					glDLlines[iSlice][i] = glGenLists(1);
				for (int i = 0; i < N_POINTS_TYPE; i++)
					glDLpoints[iSlice][i] = glGenLists(1);
				glDLgrid[iSlice] = glGenLists(1);
			}
		}
		if (glDLcreated)
		{
			glDeleteLists(glDLlinesFinal, 1);
		}
		else
		{
			glDLlinesFinal = glGenLists(1);
		}

		for (int i = 0; i < currentClusters; i++)
			globalPos[i].w = 0;

		for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
		{
			AliHLTTPCCATracker &tracker = hlt.Tracker().CPUTracker(iSlice);
			glNewList(glDLlines[iSlice][0], GL_COMPILE);
			if (drawInitLinks && iSlice == 0)
			{
				char *tmpMem[fgkNSlices];
				for (int i = 0; i < fgkNSlices; i++)
				{
					if (tracker.fLinkTmpMemory == NULL)
					{
						printf("Need to set TRACKER_KEEP_TEMPDATA for visualizing PreLinks!\n");
						break;
					}
					AliHLTTPCCATracker &tracker = hlt.Tracker().CPUTracker(i);
					tmpMem[i] = tracker.Data().Memory();
					tracker.SetGPUSliceDataMemory((void *) tracker.fLinkTmpMemory, tracker.Data().Rows());
					tracker.SetPointersSliceData(tracker.ClusterData());
					DrawLinks(tracker, 1, true);
					tracker.SetGPUSliceDataMemory(tmpMem[i], tracker.Data().Rows());
					tracker.SetPointersSliceData(tracker.ClusterData());
				}
			}
			glEndList();

			glNewList(glDLlines[iSlice][1], GL_COMPILE);
			DrawLinks(tracker, 2);
			glEndList();

			glNewList(glDLlines[iSlice][2], GL_COMPILE);
			DrawSeeds(tracker);
			glEndList();

			glNewList(glDLlines[iSlice][3], GL_COMPILE);
			DrawTracklets(tracker);
			glEndList();

			glNewList(glDLlines[iSlice][4], GL_COMPILE);
			DrawTracks(tracker, 0);
			glEndList();

			glNewList(glDLlines[iSlice][5], GL_COMPILE);
			DrawTracks(tracker, 1);
			glEndList();

			glNewList(glDLgrid[iSlice], GL_COMPILE);
			DrawGrid(tracker);
			glEndList();
		}

		glNewList(glDLlinesFinal, GL_COMPILE);
		DrawFinal(hlt);
		glEndList();

		for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
		{
			AliHLTTPCCATracker &tracker = hlt.Tracker().CPUTracker(iSlice);
			for (int i = 0; i < N_POINTS_TYPE; i++)
			{
				glNewList(glDLpoints[iSlice][i], GL_COMPILE);
				DrawClusters(tracker, i);
				glEndList();
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
		}
	}

	framesDone++;
	framesDoneFPS++;
	double time = timerFPS.GetCurrentElapsedTime();
	if (time > 1.)
	{
		float fps = (double) framesDoneFPS / time;
		printf("FPS: %f (%d frames, Slice: %d, 1:Clusters %d, 2:Prelinks %d, 3:Links %d, 4:Seeds %d, 5:Tracklets %d, 6:Tracks %d, 7:GTracks %d, 8:Merger %d, C:Mark %X)\n",
			fps, framesDone, drawSlice, drawClusters, drawInitLinks, drawLinks, drawSeeds, drawTracklets, drawTracks, drawGlobalTracks, drawFinal, (int) markClusters);
		fpsscale = 60 / fps;
		timerFPS.ResetStart();
		framesDoneFPS = 0;
	}

	//Draw Event
	if (glDLrecent)
	{
		for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
		{
			if (drawSlice != -1 && drawSlice != iSlice) continue;

			if (drawGrid)
			{
				SetColorGrid();
				glCallList(glDLgrid[iSlice]);
			}

			if (drawClusters)
			{
				SetColorClusters();
				glCallList(glDLpoints[iSlice][0]);

				if (drawInitLinks)
				{
					if (excludeClusters) goto skip1;
					SetColorInitLinks();
				}
				glCallList(glDLpoints[iSlice][1]);

				if (drawLinks)
				{
					if (excludeClusters) goto skip1;
					SetColorLinks();
				}
				else
				{
					SetColorClusters();
				}
				glCallList(glDLpoints[iSlice][2]);

				if (drawSeeds)
				{
					if (excludeClusters) goto skip1;
					SetColorSeeds();
				}
				glCallList(glDLpoints[iSlice][3]);

			skip1:
				glColor3f(0, 0.7, 1.0);
				if (drawTracklets)
				{
					if (excludeClusters) goto skip2;
					SetColorTracklets();
				}
				glCallList(glDLpoints[iSlice][4]);

				if (drawTracks)
				{
					if (excludeClusters) goto skip2;
					SetColorTracks();
				}
				glCallList(glDLpoints[iSlice][5]);

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
				glCallList(glDLpoints[iSlice][6]);

				if (drawFinal)
				{
					if (excludeClusters) goto skip3;
					SetColorFinal();
				}
				glCallList(glDLpoints[iSlice][7]);
			skip3:;
			}

			if (!excludeClusters)
			{
				if (drawInitLinks) {
					SetColorInitLinks();
					glCallList(glDLlines[iSlice][0]);
				}
				if (drawLinks) {
					SetColorLinks();
					glCallList(glDLlines[iSlice][1]);
				}
				if (drawSeeds) {
					SetColorSeeds();
					glCallList(glDLlines[iSlice][2]);
				}
				if (drawTracklets) {
					SetColorTracklets();
					glCallList(glDLlines[iSlice][3]);
				}
				if (drawTracks) {
					SetColorTracks();
					glCallList(glDLlines[iSlice][4]);
				}
				if (drawGlobalTracks) {
					SetColorGlobalTracks();
					glCallList(glDLlines[iSlice][5]);
				}
			}
		}
		if (!excludeClusters && drawFinal) {
			SetColorFinal();
			if (!drawClusters)
			{
				for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
				{
					glCallList(glDLpoints[iSlice][7]);
				}
			}
			glCallList(glDLlinesFinal);
		}
		if (!excludeClusters && markClusters)
		{
			SetColorMarked();
			for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
			{
				glCallList(glDLpoints[iSlice][8]);
			}
		}
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
	printf("[N]/[SPACE]\tNext event\n");
	printf("[Q]/[ESC]\tQuit\n");
	printf("[R]\t\tReset Display Settings\n");
	printf("[L]\t\tDraw single slice (next slice)\n");
	printf("[K]\t\tDraw single slice (previous slice)\n");
	printf("[Z]/[U]\t\tShow splitting of TPC in slices by extruding volume, [U] resets\n");
	printf("[Y]\t\tStart Animation\n");
	printf("[G]\t\tDraw Grid\n");
	printf("[I]\t\tProject onto XY-plane\n");
	printf("[X]\t\tExclude Clusters used in the tracking steps enabled for visualization ([1]-[8])\n");
	printf("[<]\t\tExclude rejected tracks\n");
	printf("[C]\t\tMark flagged clusters (splitPad = 0x1, splitTime = 0x2, edge = 0x4, singlePad = 0x8, rejectDistance = 0x10, rejectErr = 0x20\n");
	printf("[V]\t\tHide rejected clusters from tracks\n");
	printf("[B]\t\tHide all clusters not belonging or related to matched tracks\n");
	printf("[1]\t\tShow Clusters\n");
	printf("[2]\t\tShow Links that were removed\n");
	printf("[3]\t\tShow Links that remained in Neighbors Cleaner\n");
	printf("[4]\t\tShow Seeds (Start Hits)\n");
	printf("[5]\t\tShow Tracklets\n");
	printf("[6]\t\tShow Tracks (after Tracklet Selector)\n");
	printf("[7]\t\tShow Global Track Segments\n");
	printf("[8]\t\tShow Final Merged Tracks (after Track Merger)\n");
	printf("[J]\t\tShow global tracks as additional segments of final tracks\n");
	printf("[M]\t\tReorder clusters of merged tracks before showing them geometrically\n");
	printf("[T]\t\tTake Screenshot\n");
	printf("[O]\t\tSave current camera position\n");
	printf("[P]\t\tRestore camera position\n");
	printf("[H]\t\tPrint Help\n");
	printf("[W]/[S]/[A]/[D]\tZoom / Move Left Right\n");
	printf("[E]/[F]\t\tRotate\n");
	printf("[+]/[-]\t\tMake points thicker / fainter\n");
	printf("[MOUSE1]\tLook around\n");
	printf("[MOUSE2]\tShift camera\n");
	printf("[MOUSE1+2]\tZoom / Rotate\n");
	printf("[SHIFT]\tSlow Zoom / Move / Rotate\n");
}

void HandleKeyRelease(int wParam)
{
	keys[wParam] = false;

	if (wParam == 13 || wParam == 'N') buttonPressed = 1;
	else if (wParam == 'Q') buttonPressed = 2;
	else if (wParam == 'R') resetScene = 1;

	else if (wParam == 'L')
	{
		if (drawSlice == fgkNSlices - 1)
			drawSlice = -1;
		else
			drawSlice++;
	}
	else if (wParam == 'K')
	{
		if (drawSlice == -1)
			drawSlice = fgkNSlices - 1;
		else
			drawSlice--;
	}
	else if (wParam == 'J')
	{
		separateGlobalTracks ^= 1;
		updateDLList = true;
	}
	else if (wParam == 'M')
	{
		reorderFinalTracks ^= 1;
		updateDLList = true;
	}
	else if (wParam == 'C')
	{
		if (markClusters == 0) markClusters = 1;
		else if (markClusters >= 8) markClusters = 0;
		else markClusters <<= 1;
		updateDLList = true;
	}
	else if (wParam == 'V')
	{
		hideRejectedClusters ^= 1;
		updateDLList = true;
	}
	else if (wParam == 'B')
	{
		hideUnmatchedClusters ^= 1;
		updateDLList = true;
	}
	else if (wParam == 'I')
	{
		updateDLList = true;
		projectxy ^= 1;
	}
	else if (wParam == 'Z')
	{
		updateDLList = true;
		Xadd += 60;
		Zadd += 60;
	}
	else if (wParam == 'U')
	{
		updateDLList = true;
		Xadd -= 60;
		Zadd -= 60;
		if (Zadd < 0 || Xadd < 0) Zadd = Xadd = 0;
	}
	else if (wParam == 'Y')
	{
		animate = 1;
		printf("Starting Animation\n");
	}

	else if (wParam == 'G') drawGrid ^= 1;
	else if (wParam == 'X') excludeClusters ^= 1;
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
	else if (wParam == 'T')
	{
		printf("Taking Screenshot\n");
		DoScreenshot("screenshot.bmp", screenshot_scale);
	}
	else if (wParam == 'O')
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
	else if (wParam == 'P')
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
	else if (wParam == 'H')
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

/*	This Code Creates Our OpenGL Window.  Parameters Are:					*
 *	title			- Title To Appear At The Top Of The Window				*
 *	width			- Width Of The GL Window Or Fullscreen Mode				*
 *	height			- Height Of The GL Window Or Fullscreen Mode			*
 *	bits			- Number Of Bits To Use For Color (8/16/24/32)			*
 *	fullscreenflag	- Use Fullscreen Mode (TRUE) Or Windowed Mode (FALSE)	*/

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

		// Try To Set Selected Mode And Get Results.  NOTE: CDS_FULLSCREEN Gets Rid Of Start Bar.
		if (ChangeDisplaySettings(&dmScreenSettings, CDS_FULLSCREEN) != DISP_CHANGE_SUCCESSFUL)
		{
			// If The Mode Fails, Offer Two Options.  Quit Or Use Windowed Mode.
			if (MessageBox(NULL, "The Requested Fullscreen Mode Is Not Supported By\nYour Video Card. Use Windowed Mode Instead?", "NeHe GL", MB_YESNO | MB_ICONEXCLAMATION) == IDYES)
			{
				fullscreen = FALSE; // Windowed Mode Selected.  Fullscreen = FALSE
			}
			else
			{
				// Pop Up A Message Box Letting User Know The Program Is Closing.
				MessageBox(NULL, "Program Will Now Close.", "ERROR", MB_OK | MB_ICONSTOP);
				return FALSE; // Return FALSE
			}
		}
	}

	if (fullscreen) // Are We Still In Fullscreen Mode?
	{
		dwExStyle = WS_EX_APPWINDOW; // Window Extended Style
		dwStyle = WS_POPUP;          // Windows Style
		ShowCursor(FALSE);           // Hide Mouse Pointer
	}
	else
	{
		dwExStyle = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE; // Window Extended Style
		dwStyle = WS_OVERLAPPEDWINDOW;                  // Windows Style
	}

	AdjustWindowRectEx(&WindowRect, dwStyle, FALSE, dwExStyle); // Adjust Window To True Requested Size

	// Create The Window
	if (!(hWnd = CreateWindowEx(dwExStyle,            // Extended Style For The Window
	                            "OpenGL",             // Class Name
	                            title,                // Window Title
	                            dwStyle |             // Defined Window Style
	                                WS_CLIPSIBLINGS | // Required Window Style
	                                WS_CLIPCHILDREN,  // Required Window Style
	                            0,
	                            0,                                  // Window Position
	                            WindowRect.right - WindowRect.left, // Calculate Window Width
	                            WindowRect.bottom - WindowRect.top, // Calculate Window Height
	                            NULL,                               // No Parent Window
	                            NULL,                               // No Menu
	                            hInstance,                          // Instance
	                            NULL)))                             // Dont Pass Anything To WM_CREATE
	{
		KillGLWindow(); // Reset The Display
		MessageBox(NULL, "Window Creation Error.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE; // Return FALSE
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
		MessageBox(NULL, "Initialization Failed.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
		return FALSE; // Return FALSE
	}

	return TRUE; // Success
}

LRESULT CALLBACK WndProc(HWND hWnd,     // Handle For This Window
                         UINT uMsg,     // Message For This Window
                         WPARAM wParam, // Additional Message Information
                         LPARAM lParam) // Additional Message Information
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
		keys[wParam] = TRUE; // If So, Mark It As TRUE
		return 0;            // Jump Back
	}

	case WM_KEYUP: // Has A Key Been Released?
	{
		HandleKeyRelease(wParam);

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
		//ShowCursor(false);
		return 0;
	}

	case WM_LBUTTONUP:
	{
		mouseDn = false;
		//ShowCursor(true);
		return 0;
	}

	case WM_RBUTTONDOWN:
	{
		mouseDnX = GET_X_LPARAM(lParam);
		mouseDnY = GET_Y_LPARAM(lParam);
		mouseDnR = true;
		GetCursorPos(&mouseCursorPos);
		//ShowCursor(false);
		return 0;
	}

	case WM_RBUTTONUP:
	{
		mouseDnR = false;
		//ShowCursor(true);
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
	if (!CreateGLWindow("Alice HLT TPC CA Event Display", 1280, 1080, 32, fullscreen))
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

void render(void);
void init(void);

int GetKey(int key)
{
	if (key == 65453 || key == 45) return(109); //+
	if (key == 65451 || key == 43) return(107); //-
	if (key == 65505) return(16); //Shift
	if (key == 65307) return('Q'); //ESC
	if (key == 32) return(13); //Space
	if (key > 255) return(0);
	
	if (key >= 'a' && key <= 'z') key += 'A' - 'a';
	
	return(key);
}

volatile static int needUpdate = 0;

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

	int singleBufferVisual[] =
	    {
	        GLX_RGBA,           // Needs to support OpenGL
	        GLX_DEPTH_SIZE, 16, // Needs to support a 16 bit depth buffer
	        None                // end of list
	    };

	// Try for the double-bufferd visual first
	visualInfo = glXChooseVisual(g_pDisplay, DefaultScreen(g_pDisplay), doubleBufferVisual);

	if (visualInfo == NULL)
	{
		// If we can't find a double-bufferd visual, try for a single-buffered visual...
		visualInfo = glXChooseVisual(g_pDisplay, DefaultScreen(g_pDisplay), singleBufferVisual);

		if (visualInfo == NULL)
		{
			fprintf(stderr, "glxsimple: %s\n", "no RGB visual with depth buffer");
			exit(1);
		}

		g_bDoubleBuffered = false;
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
	                         1024, 1024, // Width and height of window
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
	init();

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
					//fprintf(stderr, "KeyPress event %d --> %d (%c) -> %d\n", event.xkey.keycode, (int) sym, (char) (sym > 27 ? sym : ' '), wParam);
					keys[wParam] = true;
				}
				break;

				case KeyRelease:
				{
					KeySym sym = XLookupKeysym(&event.xkey, 0);
					int wParam = GetKey(sym);
					//fprintf(stderr, "KeyRelease event %d -> %d (%c) -> %d\n", event.xkey.keycode, (int) sym, (char) (sym > 27 ? sym : ' '), wParam);
					HandleKeyRelease(wParam);
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

		render();
	}
	return(NULL);
}

void ShowNextEvent()
{
	needUpdate = 1;
}

//-----------------------------------------------------------------------------
// Name: init()
// Desc: Init OpenGL context for rendering
//-----------------------------------------------------------------------------
void init(void)
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glEnable(GL_TEXTURE_2D);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0f, 640.0f / 480.0f, 0.1f, 100.0f);

	ReSizeGLScene(1024, 768);
}

//-----------------------------------------------------------------------------
// Name: getBitmapImageData()
// Desc: Simply image loader for 24 bit BMP files.
//-----------------------------------------------------------------------------

void render(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	DrawGLScene();

	if (g_bDoubleBuffered)
		glXSwapBuffers(g_pDisplay, g_window); // Buffer swap does implicit glFlush
	else
		glFlush(); // Explicit flush for single buffered case
}

#endif
