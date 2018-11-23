#ifndef ALITPCCOMMONTRANSFORM3D_H
#define ALITPCCOMMONTRANSFORM3D_H

namespace ali_tpc_common {

class Transform3D
{
public:
	Transform3D() = default;
	Transform3D(float* v) {for (int i = 0;i < 12;i++) m[i] = v[i];}
private:
	float m[12] = {0.f};
};

}

#endif
