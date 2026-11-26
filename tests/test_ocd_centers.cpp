// 与 ocd_elbow_reference_new.py 的 structural_centers 逐点比对。
// 参考值由 tests/dump_ocd_centers.py 生成到 ocd_centers_dump.json。
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
    enum class Kind { Null, Number, String, Array, Object } kind = Kind::Null;
    double number = 0.0;
    std::string text;
    std::vector<Value> array;
    std::map<std::string, Value> object;

    const Value& at(size_t i) const { return array[i]; }
    const Value& at(const std::string& key) const { return object.at(key); }
};

class Parser {
public:
    explicit Parser(const std::string& input) : s_(input) {}
    Value parse() { skip(); return parseValue(); }

private:
    const std::string& s_;
    size_t i_ = 0;

    void skip() {
        while (i_ < s_.size()) {
            const char c = s_[i_];
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t') { ++i_; continue; }
            break;
        }
    }
    char peek() { skip(); return i_ < s_.size() ? s_[i_] : '\0'; }

    Value parseValue() {
        const char c = peek();
        if (c == '[') return parseArray();
        if (c == '{') return parseObject();
        if (c == '"') return parseString();
        if (c == 't' || c == 'f' || c == 'n') return parseLiteral();
        return parseNumber();
    }

    Value parseArray() {
        Value v;
        v.kind = Value::Kind::Array;
        ++i_;
        while (true) {
            skip();
            if (peek() == ']') { ++i_; break; }
            v.array.push_back(parseValue());
            skip();
            if (peek() == ',') { ++i_; continue; }
            if (peek() == ']') { ++i_; break; }
            std::printf("bad array at %zu\n", i_);
            std::exit(2);
        }
        return v;
    }

    Value parseObject() {
        Value v;
        v.kind = Value::Kind::Object;
        ++i_;
        while (true) {
            skip();
            if (peek() == '}') { ++i_; break; }
            const std::string key = parseString().text;
            skip();
            if (peek() == ':') ++i_;
            v.object[key] = parseValue();
            skip();
            if (peek() == ',') { ++i_; continue; }
            if (peek() == '}') { ++i_; break; }
            std::printf("bad object at %zu\n", i_);
            std::exit(2);
        }
        return v;
    }

    Value parseString() {
        Value v;
        v.kind = Value::Kind::String;
        ++i_;
        while (i_ < s_.size() && s_[i_] != '"') {
            if (s_[i_] == '\\') ++i_;
            v.text.push_back(s_[i_++]);
        }
        ++i_;
        return v;
    }

    Value parseLiteral() {
        Value v;
        if (s_.compare(i_, 4, "true") == 0) { v.kind = Value::Kind::Number; v.number = 1.0; i_ += 4; }
        else if (s_.compare(i_, 5, "false") == 0) { v.kind = Value::Kind::Number; v.number = 0.0; i_ += 5; }
        else if (s_.compare(i_, 4, "null") == 0) { i_ += 4; }
        return v;
    }

    Value parseNumber() {
        Value v;
        v.kind = Value::Kind::Number;
        const char* begin = s_.c_str() + i_;
        char* end = nullptr;
        v.number = std::strtod(begin, &end);
        i_ += static_cast<size_t>(end - begin);
        return v;
    }
};

std::string readFile(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::printf("cannot open %s\n", path);
        std::exit(2);
    }
    std::string out;
    char buffer[65536];
    size_t n = 0;
    while ((n = std::fread(buffer, 1, sizeof(buffer), f)) > 0) out.append(buffer, n);
    std::fclose(f);
    return out;
}

} // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "ocd_centers_dump.json";
    const Value root = Parser(readFile(path)).parse();

    int failures = 0;
    double worstDelta = 0.0, worstOuter = 0.0;
    int caseIndex = 0;

    for (const Value& c : root.at("cases").array) {
        ++caseIndex;
        const double bend = c.at("bend").number;

        std::vector<Vec3> bind, original, guide;
        for (const Value& p : c.at("bind").array) bind.push_back({p.at(0).number, p.at(1).number, p.at(2).number});
        for (const Value& p : c.at("original").array) original.push_back({p.at(0).number, p.at(1).number, p.at(2).number});
        for (const Value& p : c.at("guide").array) guide.push_back({p.at(0).number, p.at(1).number, p.at(2).number});

        std::vector<double> outer;
        for (const Value& v : c.at("outer").array) outer.push_back(v.number);

        std::vector<std::array<double, 3>> weights;
        for (const Value& row : c.at("weights").array) {
            std::array<double, 3> w{0.0, 0.0, 0.0};
            for (size_t k = 0; k < row.array.size() && k < 3; ++k) w[k] = row.at(k).number;
            weights.push_back(w);
        }

        std::vector<Quat> rotations;
        for (const Value& q : c.at("rotations").array)
            rotations.push_back({q.at(0).number, q.at(1).number, q.at(2).number, q.at(3).number});

        std::vector<std::vector<unsigned int>> faces;
        for (const Value& f : c.at("faces").array) {
            std::vector<unsigned int> face;
            for (const Value& ix : f.array) face.push_back(static_cast<unsigned int>(ix.number));
            faces.push_back(face);
        }

        // outer_participation 对照
        const auto gotOuter = outerParticipation(bind, weights, bend,
            c.at("twist").number, c.at("direction").number);
        for (size_t i = 0; i < outer.size() && i < gotOuter.size(); ++i) {
            const double diff = std::abs(gotOuter[i] - outer[i]);
            if (diff > worstOuter) worstOuter = diff;
            if (diff > 1e-9) {
                ++failures;
                if (failures <= 15) std::printf("FAIL outer case=%d bend=%.0f idx=%zu %.3e\n",
                    caseIndex, bend, i, diff);
            }
        }

        const Edges edges = buildEdges(bind, faces);
        Settings settings;
        settings.rotations = rotations;
        settings.centerLimit = 16.0;

        const auto got = structuralCenters(edges, original, weights, settings, guide, outer);
        size_t index = 0;
        for (const Value& d : c.at("delta").array) {
            const Vec3 want{d.at(0).number, d.at(1).number, d.at(2).number};
            const double diff = std::max({std::abs(got[index].x - want.x),
                std::abs(got[index].y - want.y), std::abs(got[index].z - want.z)});
            if (diff > worstDelta) worstDelta = diff;
            if (diff > 1e-6) {
                ++failures;
                if (failures <= 25)
                    std::printf("FAIL delta case=%d bend=%.0f idx=%zu diff=%.3e got=(%.6f %.6f %.6f) want=(%.6f %.6f %.6f)\n",
                        caseIndex, bend, index, diff, got[index].x, got[index].y, got[index].z,
                        want.x, want.y, want.z);
            }
            ++index;
        }
    }

    std::printf("\nworst deviation vs python reference:\n");
    std::printf("  outerParticipation %.3e\n", worstOuter);
    std::printf("  structuralCenters  %.3e\n", worstDelta);
    if (failures == 0) std::printf("PASS: structural_centers port matches reference\n");
    else std::printf("%d mismatching sample(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
