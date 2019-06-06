#define SNIPPET_WIDTH 8
#define SNIPPET_HEIGHT 8

#define LOCAL_PADDING 1

#define SNIPPET_WIDTH_PADDED (SNIPPET_WIDTH+2*LOCAL_PADDING)
#define SNIPPET_HEIGHT_PADDED (SNIPPET_HEIGHT+2*LOCAL_PADDING)

#define SNIPPET_SIZE (SNIPPET_WIDTH * SNIPPET_HEIGHT)
#define SNIPPET_SIZE_PADDED (SNIPPET_WIDTH_PADDED * SNIPPET_HEIGHT_PADDED)


uint myIdx(uint lx, uint ly)
{
    return SNIPPET_WIDTH_PADDED * (ly + LOCAL_PADDING) + lx + LOCAL_PADDING;
}

uint2 paddingToInnerPos(paddingpos)
{
    uint rot  = localid / (SNIPPET_WIDTH_PADDED-1);
    uint lpos = localid % (SNIPPET_WIDTH_PADDED-1);

    int2 parents[3];
    switch (lpos)
    {
    case 0:
        for (uint i = 0; i < 3; i++) 
        {
            parents[i].x = 1;
            parents[i].y = 1;
        }
        break;

    case 1:
        parents[0].x = 0;
        parents[0].y = 1;

        parents[1].x = 1;
        parents[1].y = 1;

        parents[2].x = 1;
        parents[2].y = 1;
        break;

    case SNIPPET_WIDTH_PADDED-2: // fallthrough
        parents[0].x = -1;
        parents[0].y = 1;

        parents[1].x = 0;
        parents[1].y = 1;

        parents[2].x = 0;
        parents[2].y = 1;
        break;

    default:
        parents[0].x = -1;
        parents[0].y = 1;

        parents[1].x = 0;
        parents[1].y = 1;

        parents[2].x = 1;
        parents[2].y = 1;
    }

    switch (rot)
    {
    case 0:
        parents[0].x += paddingPos;
        parents[1].x += paddingPos;
        parents[2].x += paddingPos;
        break;
    case 1:
        parents[0].y = parents[0].x + paddingPos;
        parents[0].x = SNIPPET_WIDTH_PADDED -2;
        parents[1].y = parents[1].x + paddingPos;
        parents[1].x = SNIPPET_WIDTH_PADDED -2;
        parents[2].y = parents[2].x + paddingPos;
        parents[2].x = SNIPPET_WIDTH_PADDED -2;
        break;
    case 2:
        parents[0].x = paddingPos - parents[0].x;
        // FIXME: Use padding here?
        parents[0].y = SNIPPET_HEIGHT - 2;
        parents[1].x = paddingPos - parents[1].x;
        parents[1].y = SNIPPET_HEIGHT - 2;
        parents[2].x = paddingPos - parents[2].x;
        parents[2].y = SNIPPET_HEIGHT - 2;
        break;
    case 3:
        parents[0].y = SNIPPET_HEIGHT_PADDED - (parents[0].x + paddingPos);
        parents[0].x = PADDING;
        parents[1].y = SNIPPET_HEIGHT_PADDED - (parents[1].x + paddingPos);
        parents[1].x = PADDING;
        parents[2].y = SNIPPET_HEIGHT_PADDED - (parents[2].x + paddingPos);
        parents[2].x = PADDING;
    }

    // TODO load parents
    // TODO if any parents could be peaks: load padding
}

void fillPadding(
                  uint  localid,
        local charge_t *snippet)
{
    if (localid >= (SNIPPET_SIZE_PADDED - SNIPPET_SIZE))
    {
        return;
    }



    
}


kernel
void findPeaks(
        global const charge_t *chargemap,
        global       peak_t   *peakmap)
{
    local charge_t snippet[SNIPPET_SIZE_PADDED];

    uint lx = get_local_id(0);
    uint ly = get_local_id(1);
    uint ll = get_local_linear_id();
    uint gid = get_global_id(0);

    charge_t q = chargemap[gid];

    bool belowThreshold = (q <= PEAK_THRESHOLD);
    if (work_group_all(belowThreshold))
    {
        return;
    }

    snippet[myIdx(lx, ly)] = q;
    fillPadding(ll, chargemap, snippet);


}
