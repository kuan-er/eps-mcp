"""Float32 reconstruction of EPS ComputeDelta, including angle and segment inputs.

Not a fluid simulation. Use original-DLL hash validation for published datasets.
"""
import numpy as np
F=np.float32
DEFAULTS=dict(SprayOkDistanceMeters=.3,EllipseAxisA=.95,EllipseAxisB=.9,Bifocality=0.,CrossStretch=1.,
    SoftEdgeRadiusSquared=1.4,CoreUniformRadiusSquared=.7,CoreEdgeDrop=.12,OuterEdgeExponent=2.,TailWeight=.22,
    NearEllipseMinScale=.45,MaxIncidenceDeg=85.,IncidenceDropK=.65,FluxScale=10.,MaxDeltaPerHit=18,
    PaintBackFaces=True,UseSegmentDistance=False,CameraHalfWidthMeters=.15,CameraHalfHeightMeters=.15,
    SegmentHalfDeltaRightMeters=0.,SegmentHalfDeltaUpMeters=0.,BaseFanHalfWidthMeters=.15,BaseFanHalfHeightMeters=.15)

def parameters(overrides=None):
    p=DEFAULTS.copy();p.update(overrides or {})
    return {k:(v if isinstance(v,bool) else F(v)) for k,v in p.items()}

def compute(p,ndcx,ndcy,distance,cosinc=1.,cosforward=1.,count=1):
    p=parameters(p);x=F(ndcx);y=F(ndcy);d=np.maximum(F(.001),F(distance));cnt=F(count)
    ratio=np.clip(d/p['SprayOkDistanceMeters'],F(.05),F(4))
    near=np.maximum(p['NearEllipseMinScale'],ratio)
    aa=np.maximum(F(.0001),p['EllipseAxisA']*near);bb=np.maximum(F(.0001),p['EllipseAxisB']*near)
    if p['UseSegmentDistance']:
        x=x*np.maximum(F(1e-6),p['CameraHalfWidthMeters']);y=y*np.maximum(F(1e-6),p['CameraHalfHeightMeters'])
        dx=p['SegmentHalfDeltaRightMeters'];dy=p['SegmentHalfDeltaUpMeters'];vx=F(2)*dx;vy=F(2)*dy;ll=vx*vx+vy*vy
        if ll>F(1e-12):
            t=np.clip(((x-(-dx))*vx+(y-(-dy))*vy)/ll,F(0),F(1));qx=-dx+t*vx;qy=-dy+t*vy
        else:qx=F(0);qy=F(0)
        x=(x-qx)/np.maximum(F(1e-6),p['BaseFanHalfWidthMeters']);y=(y-qy)/np.maximum(F(1e-6),p['BaseFanHalfHeightMeters'])
    u=x/aa/np.clip(p['CrossStretch'],F(.4),F(2.5));v=y/bb
    r=u*u+v*v;bif=np.clip(p['Bifocality'],F(0),F(1));shift=F(.65)*bif
    lo=v-shift;hi=v+shift;r=r+(np.minimum(u*u+lo*lo,u*u+hi*hi)-r)*bif
    core=np.clip(p['CoreUniformRadiusSquared'],F(0),F(.999));drop=np.clip(p['CoreEdgeDrop'],F(0),F(.95))
    t=np.clip((r-core)/np.maximum(F(.0001),F(1)-core),F(0),F(1));inside=F(1)-drop*(t*t*(F(3)-F(2)*t))
    o=np.clip((r-F(1))/np.maximum(F(.0001),p['SoftEdgeRadiusSquared']-F(1)),F(0),F(1))
    outside=(F(1)-drop)*np.power(F(1)-o,np.maximum(F(.1),p['OuterEdgeExponent']))
    outside*=F(1)-(F(1)-np.clip(p['TailWeight'],F(0),F(1)))*o*o
    shape=np.where(r<=core,F(1),np.where(r<=F(1),inside,outside))
    ci=np.abs(F(cosinc)) if p['PaintBackFaces'] else np.maximum(F(0),F(cosinc))
    angle=np.arccos(np.clip(ci,F(0),F(1)))*F(57.295780181884766)
    aw=np.clip(F(1)-p['IncidenceDropK']*(angle/F(90)),F(0),F(1))
    dw=F(1)/np.maximum(F(.25),ratio*ratio)
    fw=F(.8)+F(.2)*np.clip((F(cosforward)+F(1))*F(.5),F(0),F(1))
    hw=F(.85)+F(.35)*np.log(F(1)+cnt)
    raw=p['FluxScale']*shape*aw*F(1)*dw*fw*hw
    valid=(cnt>0)&(d<=p['SprayOkDistanceMeters'])&(r<p['SoftEdgeRadiusSquared'])&(ci>0)&(angle<=p['MaxIncidenceDeg'])
    return np.where(valid,np.clip(np.rint(raw),F(0),p['MaxDeltaPerHit']),F(0)).astype(np.int32)

def sample(spec):
    p=parameters(spec['parameters']);n=spec.get('n',300);span=F(spec.get('span_mm',300));half=span*F(.5);step=span/F(n)
    coords=-half+(np.arange(n,dtype=np.float32)+F(.5))*step;x,y=np.meshgrid(coords/F(1000),coords/F(1000))
    xx=x/p['CameraHalfWidthMeters'];yy=y/p['CameraHalfHeightMeters'];axial=F(F(spec.get('distance_mm',200))/F(1000))
    if spec.get('mode','plane')=='plane':
        # Analytic single-hit plane; includes shader quantization, not rasterization/UV aggregation.
        dist=np.sqrt(x*x+y*y+axial*axial);ci=axial/dist;cf=F(1)
        dist=np.rint(dist*F(1000))*F(.001);ci=np.rint(ci*F(1e6))*F(1e-6)
        xx=np.rint(xx*F(1e6))*F(1e-6);yy=np.rint(yy*F(1e6))*F(1e-6)
    else:dist=axial;ci=F(spec.get('cosinc',1));cf=F(spec.get('cosforward',1))
    a=compute(p,xx,yy,dist,ci,cf,spec.get('count',1))
    if spec.get('clip_camera',True):a=np.where((np.abs(x)<=p['CameraHalfWidthMeters'])&(np.abs(y)<=p['CameraHalfHeightMeters']),a,0).astype(np.int32)
    return a

def map_gun(features,settings):
    w=F(max(float(features['SprayWidth'])*.001,.01));cone=bool(features['ConeSpray'])
    h=w if cone else np.maximum(F(.001),np.minimum(w,F(.06)))
    p=DEFAULTS.copy();p.update({k:v for k,v in settings.items() if k in p})
    p.update(SprayOkDistanceMeters=max(settings.get('MaxSprayDistanceMm',300)*.001,.01),
             MaxIncidenceDeg=min(settings.get('MaxIncidenceDeg',85),85),
             EllipseAxisA=F(.95) if cone else np.maximum(F(.12),F(.95)*np.clip(h/w,F(.01),F(1))),
             EllipseAxisB=F(.9 if cone else .95),CameraHalfWidthMeters=w*F(.5),CameraHalfHeightMeters=h*F(.5),
             BaseFanHalfWidthMeters=w*F(.5),BaseFanHalfHeightMeters=h*F(.5),UseSegmentDistance=False)
    return {k:v if isinstance(v,bool) else int(v) if k=='MaxDeltaPerHit' else float(v) for k,v in p.items()}
