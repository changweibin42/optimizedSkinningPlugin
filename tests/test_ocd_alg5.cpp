// 用参考实现导出的 delta 校验 ocd_centers 的完整链路（含 buildEdges）。
// 数据由 tests/_dump_alg5.py 生成。
#include "ocd_centers.h"
#include "ocd_math.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
using namespace ocd_centers;

namespace {
struct Value {
    enum class K { Null, Num, Str, Arr, Obj } kind = K::Null;
    double num = 0; std::string text;
    std::vector<Value> arr; std::map<std::string, Value> obj;
    const Value& at(size_t i) const { return arr[i]; }
    const Value& at(const std::string& k) const { return obj.at(k); }
};
class P {
public:
    explicit P(const std::string& s) : s_(s) {}
    Value parse() { skip(); return val(); }
private:
    const std::string& s_; size_t i_ = 0;
    void skip() { while (i_ < s_.size() && (s_[i_]==' '||s_[i_]=='\n'||s_[i_]=='\r'||s_[i_]=='\t')) ++i_; }
    char peek() { skip(); return i_ < s_.size() ? s_[i_] : '\0'; }
    Value val() { char c = peek(); if (c=='[') return arr(); if (c=='{') return obj(); if (c=='"') return str(); if (c=='t'||c=='f'||c=='n') return lit(); return num(); }
    Value arr() { Value v; v.kind=Value::K::Arr; ++i_; while(true){ skip(); if(peek()==']'){++i_;break;} v.arr.push_back(val()); skip(); if(peek()==','){++i_;continue;} if(peek()==']'){++i_;break;} std::exit(2);} return v; }
    Value obj() { Value v; v.kind=Value::K::Obj; ++i_; while(true){ skip(); if(peek()=='}'){++i_;break;} std::string k=str().text; skip(); if(peek()==':')++i_; v.obj[k]=val(); skip(); if(peek()==','){++i_;continue;} if(peek()=='}'){++i_;break;} std::exit(2);} return v; }
    Value str() { Value v; v.kind=Value::K::Str; ++i_; while(i_<s_.size()&&s_[i_]!='"'){ if(s_[i_]=='\\')++i_; v.text.push_back(s_[i_++]); } ++i_; return v; }
    Value lit() { Value v; v.kind=Value::K::Num; if(s_.compare(i_,4,"true")==0){v.num=1;i_+=4;} else if(s_.compare(i_,5,"false")==0){i_+=5;} else { i_+=4; } return v; }
    Value num() { Value v; v.kind=Value::K::Num; char* e=nullptr; v.num=std::strtod(s_.c_str()+i_,&e); i_+=(size_t)(e-(s_.c_str()+i_)); return v; }
};
std::string slurp(const char* p){ std::FILE* f=std::fopen(p,"rb"); if(!f){std::printf("open fail %s\n",p);std::exit(2);} std::string o; char b[65536]; size_t n; while((n=std::fread(b,1,sizeof(b),f))>0) o.append(b,n); std::fclose(f); return o; }
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "ocd_alg5_reference.json";
    const Value root = P(slurp(path)).parse();
    int failures = 0; double worst = 0.0;
    for (const Value& c : root.at("cases").arr) {
        const double bend = c.at("bend").num;
        std::vector<Vec3> bind, original, guide;
        for (const Value& p : c.at("bind").arr) bind.push_back({p.at(0).num,p.at(1).num,p.at(2).num});
        for (const Value& p : c.at("original").arr) original.push_back({p.at(0).num,p.at(1).num,p.at(2).num});
        for (const Value& p : c.at("guide").arr) guide.push_back({p.at(0).num,p.at(1).num,p.at(2).num});
        std::vector<double> outer;
        for (const Value& v : c.at("outer").arr) outer.push_back(v.num);
        std::vector<std::array<double,3>> weights;
        for (const Value& row : c.at("weights").arr) {
            std::array<double,3> w{0,0,0};
            for (size_t k=0;k<row.arr.size()&&k<3;++k) w[k]=row.at(k).num;
            weights.push_back(w);
        }
        std::vector<Quat> rotations;
        for (const Value& q : c.at("rotations").arr) rotations.push_back({q.at(0).num,q.at(1).num,q.at(2).num,q.at(3).num});
        std::vector<std::vector<unsigned int>> faces;
        for (const Value& f : c.at("faces").arr) {
            std::vector<unsigned int> face;
            for (const Value& ix : f.arr) face.push_back((unsigned int)ix.num);
            faces.push_back(face);
        }
        const Edges edges = buildEdges(bind, faces);
        Settings s; s.rotations = rotations; s.centerLimit = 16.0;
        const auto got = structuralCenters(edges, original, weights, s, guide, outer);
        size_t idx = 0;
        for (const Value& d : c.at("delta").arr) {
            const Vec3 want{d.at(0).num,d.at(1).num,d.at(2).num};
            const double diff = std::max({std::abs(got[idx].x-want.x),std::abs(got[idx].y-want.y),std::abs(got[idx].z-want.z)});
            if (diff > worst) worst = diff;
            if (diff > 1e-9) { ++failures; if (failures<=10) std::printf("FAIL bend=%.0f idx=%zu %.3e\n", bend, idx, diff); }
            ++idx;
        }
    }
    std::printf("worst delta deviation %.3e\n", worst);
    std::printf(failures==0 ? "PASS: structural_centers full-chain matches reference\n" : "%d mismatches\n", failures);
    return failures==0?0:1;
}
