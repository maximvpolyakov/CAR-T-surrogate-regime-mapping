#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct Params {
    double D_C=0.014, D_T=5e-5, D_E=0.001, D_S=0.0045, D_A=0.02, D_H=1.5;
    double rho_C=5.0, r_T=0.05, delta_CT=0.30, gamma=30.0;
    double d_C=0.03, k_e0=0.03, K_C=0.4166666666666667, K_T=1.0;
    double sigma_S=0.56, lambda_S=0.56;
    double sigma_A=2.0, sigma_M=0.30, lambda_A=0.75;
    double eta=2.0, lambda_H=0.43;
    double chi_0=0.1, p_r=0.14, d_E=0.04, d_TA=0.01, d_TB=0.01;
    double mu_0=0.05, mu_1=0.089, nu=0.024, xi=0.24;
    double alpha_A=2.0, beta_A=3.0, beta_H=10.0, K_H=0.5;
};

struct Grid3D {
    int nx=40, ny=40, nz=40;
    double Lx=1.0, Ly=1.0, Lz=1.0;
    double dx=0.0, dy=0.0, dz=0.0;
    void finalize(){
        if(nx<2||ny<2||nz<2) throw std::runtime_error("grid dimensions must be >=2");
        dx=Lx/double(nx-1); dy=Ly/double(ny-1); dz=Lz/double(nz-1);
    }
    std::size_t size() const {return std::size_t(nx)*std::size_t(ny)*std::size_t(nz);}
};

static inline int clampi(int v,int lo,int hi){return std::max(lo,std::min(hi,v));}

struct Field {
    Grid3D g;
    std::vector<double> v;
    Field()=default;
    explicit Field(const Grid3D& gg):g(gg),v(gg.size(),0.0){}
    std::size_t idx(int i,int j,int k) const {
        return (std::size_t(k)*std::size_t(g.ny)+std::size_t(j))*std::size_t(g.nx)+std::size_t(i);
    }
    double get(int i,int j,int k) const {
        i=clampi(i,0,g.nx-1); j=clampi(j,0,g.ny-1); k=clampi(k,0,g.nz-1);
        return v[idx(i,j,k)];
    }
    void set(int i,int j,int k,double x){v[idx(i,j,k)]=x;}
    void clamp_nonneg(){
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(std::ptrdiff_t q=0;q<(std::ptrdiff_t)v.size();++q){
            if(v[(std::size_t)q]<0.0) v[(std::size_t)q]=0.0;
        }
    }
};

struct State {
    Field C,E,TA,TB,S,A,H;
    explicit State(const Grid3D& g):C(g),E(g),TA(g),TB(g),S(g),A(g),H(g){}
    void clamp_nonneg(){C.clamp_nonneg();E.clamp_nonneg();TA.clamp_nonneg();TB.clamp_nonneg();S.clamp_nonneg();A.clamp_nonneg();H.clamp_nonneg();}
};

struct Deriv {
    Field dC,dE,dTA,dTB,dS,dA,dH;
    explicit Deriv(const Grid3D& g):dC(g),dE(g),dTA(g),dTB(g),dS(g),dA(g),dH(g){}
};

static std::map<std::string,std::string> parse_args(int argc,char** argv){
    std::map<std::string,std::string> kv;
    for(int i=1;i<argc;++i){
        std::string s=argv[i]; if(s.rfind("--",0)==0) s=s.substr(2);
        auto p=s.find('='); if(p==std::string::npos) kv[s]="1"; else kv[s.substr(0,p)]=s.substr(p+1);
    }
    return kv;
}

template<class T> static T num(const std::map<std::string,std::string>& kv,const std::string& key,T def){
    auto it=kv.find(key); if(it==kv.end()) return def; std::istringstream ss(it->second); T x{}; ss>>x;
    if(!ss) throw std::runtime_error("cannot parse "+key); return x;
}
static std::string strv(const std::map<std::string,std::string>& kv,const std::string& key,const std::string& def){
    auto it=kv.find(key); return it==kv.end()?def:it->second;
}

static inline double chi_C(const Params&p,double A){return p.chi_0/(1.0+p.alpha_A*A);}
static inline double k_e(const Params&p,double A,double H){return p.k_e0*(1.0+p.beta_A*A+p.beta_H*H);}
static inline double mu_AB(const Params&p,double C){return p.mu_0+p.nu*C;}
static inline double mu_BA(const Params&p,double C){return p.mu_1*std::exp(-p.xi*C);}
static inline double fH(const Params&p,double H){return H/(p.K_H+H+1e-30);}

static Field chemotaxis(const State&s,const Params&p){
    const auto& g=s.C.g; Field out(g);
    const double idx=1.0/g.dx, idy=1.0/g.dy, idz=1.0/g.dz;
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(int k=0;k<g.nz;++k) for(int j=0;j<g.ny;++j) for(int i=0;i<g.nx;++i){
        auto fx=[&](int ia,int ib){
            double ch=0.5*(chi_C(p,s.A.get(ia,j,k))+chi_C(p,s.A.get(ib,j,k)));
            double cf=0.5*(s.C.get(ia,j,k)+s.C.get(ib,j,k));
            return ch*cf*(s.S.get(ib,j,k)-s.S.get(ia,j,k))*idx;
        };
        auto fy=[&](int ja,int jb){
            double ch=0.5*(chi_C(p,s.A.get(i,ja,k))+chi_C(p,s.A.get(i,jb,k)));
            double cf=0.5*(s.C.get(i,ja,k)+s.C.get(i,jb,k));
            return ch*cf*(s.S.get(i,jb,k)-s.S.get(i,ja,k))*idy;
        };
        auto fz=[&](int ka,int kb){
            double ch=0.5*(chi_C(p,s.A.get(i,j,ka))+chi_C(p,s.A.get(i,j,kb)));
            double cf=0.5*(s.C.get(i,j,ka)+s.C.get(i,j,kb));
            return ch*cf*(s.S.get(i,j,kb)-s.S.get(i,j,ka))*idz;
        };
        double div=(fx(i,i+1)-fx(i-1,i))*idx+(fy(j,j+1)-fy(j-1,j))*idy+(fz(k,k+1)-fz(k-1,k))*idz;
        out.set(i,j,k,-div);
    }
    return out;
}

static void explicit_rhs(const State&s,const Params&p,Deriv&o){
    const auto& g=s.C.g; Field chem=chemotaxis(s,p);
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(int k=0;k<g.nz;++k) for(int j=0;j<g.ny;++j) for(int i=0;i<g.nx;++i){
        double C=s.C.get(i,j,k), E=s.E.get(i,j,k), TA=s.TA.get(i,j,k), TB=s.TB.get(i,j,k);
        double S=s.S.get(i,j,k), A=s.A.get(i,j,k), H=s.H.get(i,j,k);
        double ke=k_e(p,A,H), muab=mu_AB(p,C), muba=mu_BA(p,C), pre=p.p_r/(1.0+A);
        double logistic=p.r_T*(1.0-(TA+TB));
        o.dC.set(i,j,k,p.rho_C*C*TA/(p.K_C+C+1e-30)-p.delta_CT*C*TA-ke*C+pre*E-p.d_C*C+chem.get(i,j,k));
        o.dE.set(i,j,k,ke*C-pre*E-p.d_E*E);
        o.dTA.set(i,j,k,logistic*TA-p.gamma*C*TA-muab*TA+muba*TB-p.d_TA*TA);
        o.dTB.set(i,j,k,logistic*TB+muab*TA-muba*TB-p.d_TB*TB);
        o.dS.set(i,j,k,p.sigma_S*TA-p.lambda_S*S);
        o.dA.set(i,j,k,p.sigma_A*(TA+TB)+p.sigma_M*fH(p,H)-p.lambda_A*A);
        o.dH.set(i,j,k,p.eta*(TA+TB)-p.lambda_H*H);
    }
}

static void add_scaled(State&s,const Deriv&d,double a){
    std::size_t n=s.C.v.size();
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(std::ptrdiff_t qq=0;qq<(std::ptrdiff_t)n;++qq){
        auto q=(std::size_t)qq;
        s.C.v[q]+=a*d.dC.v[q]; s.E.v[q]+=a*d.dE.v[q]; s.TA.v[q]+=a*d.dTA.v[q]; s.TB.v[q]+=a*d.dTB.v[q];
        s.S.v[q]+=a*d.dS.v[q]; s.A.v[q]+=a*d.dA.v[q]; s.H.v[q]+=a*d.dH.v[q];
    }
}
static State copy_add(const State&b,const Deriv&d,double a){
    State x(b.C.g); x.C.v=b.C.v;x.E.v=b.E.v;x.TA.v=b.TA.v;x.TB.v=b.TB.v;x.S.v=b.S.v;x.A.v=b.A.v;x.H.v=b.H.v; add_scaled(x,d,a); return x;
}
static void rk2(State&s,const Params&p,double dt,int substeps){
    if(substeps<1) substeps=1; double h=dt/double(substeps); Deriv k1(s.C.g),k2(s.C.g);
    for(int m=0;m<substeps;++m){
        explicit_rhs(s,p,k1); State tmp=copy_add(s,k1,h); tmp.clamp_nonneg(); explicit_rhs(tmp,p,k2);
        add_scaled(s,k1,0.5*h); add_scaled(s,k2,0.5*h); s.clamp_nonneg();
    }
}

static inline std::size_t id3(const Grid3D&g,int i,int j,int k){return (std::size_t(k)*g.ny+j)*g.nx+i;}
static inline double sample(const Grid3D&g,const std::vector<double>&v,int i,int j,int k){
    i=clampi(i,0,g.nx-1);j=clampi(j,0,g.ny-1);k=clampi(k,0,g.nz-1);return v[id3(g,i,j,k)];
}
static void implicit_diff(Field&u,const Field&rhs,double D,double dt,int iters,double tol){
    if(D<=0||dt<=0){u.v=rhs.v;return;} const auto&g=u.g;
    double ax=dt*D/(g.dx*g.dx), ay=dt*D/(g.dy*g.dy), az=dt*D/(g.dz*g.dz), den=1.0+2.0*(ax+ay+az);
    std::vector<double> cur=rhs.v,nxt=cur;
    for(int m=0;m<iters;++m){
        double md=0.0;
        #ifdef _OPENMP
        #pragma omp parallel
        {
            double local_md=0.0;
            #pragma omp for
            for(int k=0;k<g.nz;++k){
                for(int j=0;j<g.ny;++j){
                    for(int i=0;i<g.nx;++i){
                        auto id=id3(g,i,j,k);
                        double nv=(rhs.v[id]+ax*(sample(g,cur,i+1,j,k)+sample(g,cur,i-1,j,k))
                                 +ay*(sample(g,cur,i,j+1,k)+sample(g,cur,i,j-1,k))
                                 +az*(sample(g,cur,i,j,k+1)+sample(g,cur,i,j,k-1)))/den;
                        nxt[id]=nv;
                        local_md=std::max(local_md,std::abs(nv-cur[id]));
                    }
                }
            }
            #pragma omp critical
            {
                md=std::max(md,local_md);
            }
        }
        #else
        for(int k=0;k<g.nz;++k) for(int j=0;j<g.ny;++j) for(int i=0;i<g.nx;++i){
            auto id=id3(g,i,j,k);
            double nv=(rhs.v[id]+ax*(sample(g,cur,i+1,j,k)+sample(g,cur,i-1,j,k))
                     +ay*(sample(g,cur,i,j+1,k)+sample(g,cur,i,j-1,k))
                     +az*(sample(g,cur,i,j,k+1)+sample(g,cur,i,j,k-1)))/den;
            nxt[id]=nv;
            md=std::max(md,std::abs(nv-cur[id]));
        }
        #endif
        cur.swap(nxt); if(md<tol) break;
    }
    u.v.swap(cur);
}

static void diffusion(State&s,const Params&p,double dt,int iters,double tol){
    auto mk=[&](const Field&f){Field r(f.g);r.v=f.v;return r;};
    Field rC=mk(s.C),rE=mk(s.E),rTA=mk(s.TA),rTB=mk(s.TB),rS=mk(s.S),rA=mk(s.A),rH=mk(s.H);
    implicit_diff(s.C,rC,p.D_C,dt,iters,tol); implicit_diff(s.E,rE,p.D_E,dt,iters,tol);
    implicit_diff(s.TA,rTA,p.D_T,dt,iters,tol); implicit_diff(s.TB,rTB,p.D_T,dt,iters,tol);
    implicit_diff(s.S,rS,p.D_S,dt,iters,tol); implicit_diff(s.A,rA,p.D_A,dt,iters,tol); implicit_diff(s.H,rH,p.D_H,dt,iters,tol);
    s.clamp_nonneg();
}
static void strang(State&s,const Params&p,double dt,int sub,int iters,double tol){rk2(s,p,0.5*dt,sub);diffusion(s,p,dt,iters,tol);rk2(s,p,0.5*dt,sub);}

static void init_state(State&s,double tumor_radius,double tumor_value,double cart_layer,double cart_radius,double cart_value){
    const auto&g=s.C.g; double cx=0.5*g.Lx,cy=0.5*g.Ly,cz=0.5*g.Lz;
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(int k=0;k<g.nz;++k) for(int j=0;j<g.ny;++j) for(int i=0;i<g.nx;++i){
        double x=i*g.dx,y=j*g.dy,z=k*g.dz;
        double r=std::sqrt((x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz));
        s.TA.set(i,j,k,r<=tumor_radius?tumor_value:0.0); s.TB.set(i,j,k,0.0);
        s.E.set(i,j,k,0.0);s.S.set(i,j,k,0.0);s.A.set(i,j,k,0.0);s.H.set(i,j,k,0.0);
        double ryz=std::sqrt((y-cy)*(y-cy)+(z-cz)*(z-cz));
        s.C.set(i,j,k,(x<=cart_layer && ryz<=cart_radius)?cart_value:0.0);
    }
}
static double field_sum(const Field&f){
    double q=0.0; std::size_t n=f.v.size();
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:q)
    #endif
    for(std::ptrdiff_t ii=0;ii<(std::ptrdiff_t)n;++ii){q+=f.v[(std::size_t)ii];}
    return q;
}
static double tumour_sum(const State&s){ return field_sum(s.TA)+field_sum(s.TB); }
static double field_max(const Field&f){
    double m=0.0;
    for(double x:f.v) m=std::max(m,x);
    return m;
}
struct Snapshot {
    double R=0,TA_rel=0,TB_rel=0,C_rel=0,E_rel=0;
    double TA_mean=0,TB_mean=0,C_mean=0,E_mean=0,S_mean=0,A_mean=0,H_mean=0;
    double C_max=0,E_max=0,antigen_low_fraction=0;
};
static Snapshot snapshot(const State&s,double b0){
    const double n=double(s.TA.v.size());
    const double ta=field_sum(s.TA), tb=field_sum(s.TB), c=field_sum(s.C), e=field_sum(s.E);
    Snapshot q; q.R=(ta+tb)/b0; q.TA_rel=ta/b0; q.TB_rel=tb/b0; q.C_rel=c/b0; q.E_rel=e/b0;
    q.TA_mean=ta/n; q.TB_mean=tb/n; q.C_mean=c/n; q.E_mean=e/n;
    q.S_mean=field_sum(s.S)/n; q.A_mean=field_sum(s.A)/n; q.H_mean=field_sum(s.H)/n;
    q.C_max=field_max(s.C); q.E_max=field_max(s.E);
    q.antigen_low_fraction=(ta+tb)>0?tb/(ta+tb):0.0;
    return q;
}
static void write_traj_row(std::ofstream&o,const State&s,double t,double b0){
    Snapshot q=snapshot(s,b0);
    o<<t<<','<<q.R<<','<<q.TA_rel<<','<<q.TB_rel<<','<<q.C_rel<<','<<q.E_rel<<','
     <<q.TA_mean<<','<<q.TB_mean<<','<<q.C_mean<<','<<q.E_mean<<','
     <<q.S_mean<<','<<q.A_mean<<','<<q.H_mean<<','<<q.C_max<<','<<q.E_max<<','<<q.antigen_low_fraction<<'\n';
}
static void write_final_fields(const std::string&path,const State&s,double t){
    std::ofstream o(path); if(!o) throw std::runtime_error("cannot open fields file");
    o<<std::setprecision(16)<<"t,i,j,k,x,y,z,C,E,TA,TB,S,A,H\n"; const auto&g=s.C.g;
    for(int k=0;k<g.nz;++k)for(int j=0;j<g.ny;++j)for(int i=0;i<g.nx;++i){
        o<<t<<','<<i<<','<<j<<','<<k<<','<<i*g.dx<<','<<j*g.dy<<','<<k*g.dz<<','
         <<s.C.get(i,j,k)<<','<<s.E.get(i,j,k)<<','<<s.TA.get(i,j,k)<<','<<s.TB.get(i,j,k)<<','
         <<s.S.get(i,j,k)<<','<<s.A.get(i,j,k)<<','<<s.H.get(i,j,k)<<'\n';
    }
}

int main(int argc,char**argv){
    try{
        auto kv=parse_args(argc,argv); int threads=num<int>(kv,"threads",0);
        #ifdef _OPENMP
        if(threads>0) omp_set_num_threads(threads);
        #endif
        Grid3D g; g.nx=num<int>(kv,"nx",40);g.ny=num<int>(kv,"ny",40);g.nz=num<int>(kv,"nz",40);
        g.Lx=num<double>(kv,"Lx",1.0);g.Ly=num<double>(kv,"Ly",1.0);g.Lz=num<double>(kv,"Lz",1.0);g.finalize();
        double dt=num<double>(kv,"dt",0.01),T=num<double>(kv,"T",30.0);int sub=num<int>(kv,"exp_substeps",1),iters=num<int>(kv,"diff_iters",60);double tol=num<double>(kv,"diff_tol",1e-8);
        int traj_every=num<int>(kv,"trajectory_every",10); std::string summary_path=strv(kv,"summary_path","summary.csv"),traj_path=strv(kv,"trajectory_path","");
        std::string fields_path=strv(kv,"final_fields_path","");
        Params p;
        #define SETP(name) p.name=num<double>(kv,#name,p.name)
        SETP(D_C);SETP(D_T);SETP(D_E);SETP(D_S);SETP(D_A);SETP(D_H);SETP(rho_C);SETP(r_T);SETP(delta_CT);SETP(gamma);SETP(d_C);SETP(k_e0);SETP(K_C);SETP(K_T);SETP(sigma_S);SETP(lambda_S);SETP(sigma_A);SETP(sigma_M);SETP(lambda_A);SETP(eta);SETP(lambda_H);SETP(chi_0);SETP(p_r);SETP(d_E);SETP(d_TA);SETP(d_TB);SETP(mu_0);SETP(mu_1);SETP(nu);SETP(xi);SETP(alpha_A);SETP(beta_A);SETP(beta_H);SETP(K_H);
        #undef SETP
        double tr=num<double>(kv,"tumor_radius",0.2),tv=num<double>(kv,"tumor_value",0.1),cl=num<double>(kv,"cart_layer",0.05),cr=num<double>(kv,"cart_radius",0.2),cv=num<double>(kv,"cart_value",0.2);
        State s(g);init_state(s,tr,tv,cl,cr,cv); double b0=tumour_sum(s); if(!(b0>0)) throw std::runtime_error("initial tumour burden is zero");
        std::ofstream traj; if(!traj_path.empty()){
            traj.open(traj_path);
            traj<<std::setprecision(16)<<"t,R,TA_rel,TB_rel,C_rel,E_rel,TA_mean,TB_mean,C_mean,E_mean,S_mean,A_mean,H_mean,C_max,E_max,antigen_low_fraction\n";
            write_traj_row(traj,s,0.0,b0);
        }
        Snapshot q0=snapshot(s,b0);
        double ymin=1.0,tmin=0.0;
        const double initial_C_mean=q0.C_mean;
        const double initial_C_to_tumour_ratio=q0.C_rel;
        double c_peak=q0.C_mean,t_c_peak=0.0,e_peak=q0.E_mean,t_e_peak=0.0;
        double c_peak_local=q0.C_max,t_c_peak_local=0.0,a_peak=q0.A_mean,h_peak=q0.H_mean;
        int steps=(int)std::ceil(T/dt); double actual_dt=T/double(steps);
        auto wall_start=std::chrono::steady_clock::now();
        for(int n=1;n<=steps;++n){
            strang(s,p,actual_dt,sub,iters,tol); double R=tumour_sum(s)/b0; if(!std::isfinite(R)) throw std::runtime_error("non-finite tumour burden");
            if(R<ymin){ymin=R;tmin=n*actual_dt;}
            if(n%traj_every==0||n==steps){
                Snapshot q=snapshot(s,b0);
                if(q.C_mean>c_peak){c_peak=q.C_mean;t_c_peak=n*actual_dt;}
                if(q.E_mean>e_peak){e_peak=q.E_mean;t_e_peak=n*actual_dt;}
                if(q.C_max>c_peak_local){c_peak_local=q.C_max;t_c_peak_local=n*actual_dt;}
                a_peak=std::max(a_peak,q.A_mean); h_peak=std::max(h_peak,q.H_mean);
                if(traj) write_traj_row(traj,s,n*actual_dt,b0);
            }
        }
        double yend=tumour_sum(s)/b0,yreb=yend-ymin;
        Snapshot qf=snapshot(s,b0);
        auto wall_end=std::chrono::steady_clock::now();
        double runtime=std::chrono::duration<double>(wall_end-wall_start).count();
        std::ofstream out(summary_path); if(!out) throw std::runtime_error("cannot open summary file");
        out<<std::setprecision(16)
           <<"Y_end,Y_min,Y_reb,t_min,B0,B_end,dt,T,nx,ny,nz,C_peak_mean,t_C_peak,C_peak_local,t_C_peak_local,E_peak_mean,t_E_peak,initial_C_mean,initial_C_to_tumour_ratio,C_final_mean,E_final_mean,TA_final_mean,TB_final_mean,A_peak_mean,H_peak_mean,antigen_low_final,tumour_peak_local_final,runtime_seconds\n"
           <<yend<<','<<ymin<<','<<yreb<<','<<tmin<<','<<b0<<','<<tumour_sum(s)<<','<<actual_dt<<','<<T<<','<<g.nx<<','<<g.ny<<','<<g.nz<<','
           <<c_peak<<','<<t_c_peak<<','<<c_peak_local<<','<<t_c_peak_local<<','<<e_peak<<','<<t_e_peak<<','<<initial_C_mean<<','<<initial_C_to_tumour_ratio<<','
           <<qf.C_mean<<','<<qf.E_mean<<','<<qf.TA_mean<<','<<qf.TB_mean<<','
           <<a_peak<<','<<h_peak<<','<<qf.antigen_low_fraction<<','<<std::max(field_max(s.TA),field_max(s.TB))<<','<<runtime<<'\n';
        if(!fields_path.empty()) write_final_fields(fields_path,s,T);
        std::cout<<std::setprecision(12)<<"Y_end="<<yend<<" Y_min="<<ymin<<" Y_reb="<<yreb<<"\n";
        return 0;
    }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 2;}
}
