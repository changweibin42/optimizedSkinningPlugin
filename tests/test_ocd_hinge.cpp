// 与 scripts/ocd_elbow_reference.py 逐点比对 ocd_hinge 移植。
// 参考值由 tests/dump_ocd_reference.py 生成到 ocd_reference_dump.json。
#include "ocd_hinge.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using namespace ocd_hinge;

// ------------------------------------------------------------------
// 极简 JSON 读取：只需要数字、字符串与嵌套数组/对象，够用即可。
// ------------------------------------------------------------------
namespace {

struct Value;

struct Array {
    std::vector<Value> items;
};

struct Object {
    std::map<std::string, Value> items;
};

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

    Value parse() {
        skip();
        return parseValue();
    }

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

    char peek() {
        skip();
        return i_ < s_.size() ? s_[i_] : '\0';
    }

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
        ++i_;  // [
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
        ++i_;  // {
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
        ++i_;  // "
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

int failures = 0;
double worstElbow = 0.0, worstHinge = 0.0, worstSection = 0.0;

void report(const char* group, double diff, double& worst, const char* detail) {
    if (diff > worst) worst = diff;
    if (diff > 1e-9) {
        ++failures;
        if (failures <= 25) std::printf("FAIL %-8s %.3e  %s\n", group, diff, detail);
    }
}

} // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "ocd_reference_dump.json";
    const Value root = Parser(readFile(path)).parse();

    // ---------------- Elbow ----------------
    for (const Value& row : root.at("elbow").array) {
        Pose pose;
        pose.bend = row.at(0).number;
        pose.twist = row.at(1).number;
        pose.direction = row.at(2).number;
        pose.avoidOverlap = row.at(3).number != 0.0;
        if (row.array.size() > 10) {
            for (size_t k = 0; k < 11 && k < row.at(10).array.size(); ++k)
                pose.values[k] = row.at(10).at(k).number;
        }
        Elbow elbow(pose);
        const Vec3 got = elbow.point(row.at(4).number, row.at(5).number, row.at(6).number);
        const double wantX = row.at(7).number;
        const double wantY = row.at(8).number;
        const double wantZ = row.at(9).number;
        const double diff = std::max({std::abs(got.x - wantX), std::abs(got.y - wantY),
            std::abs(got.z - wantZ)});
        char detail[160];
        std::snprintf(detail, sizeof(detail), "bend=%.0f twist=%.0f dir=%.0f s=%.2f y=%.2f z=%.2f",
            pose.bend, pose.twist, pose.direction, row.at(4).number, row.at(5).number,
            row.at(6).number);
        report("elbow", diff, worstElbow, detail);
    }

    // ---------------- hinge_section ----------------
    for (const Value& row : root.at("section").array) {
        const double bend = row.at(0).number;
        const double depth = row.at(1).number;
        const double station = row.at(2).number;
        // 参考实现的 support 由调用方数组内部推导，这里复现同样的推导。
        // 单点数组不含 mixed 顶点，故 support = min(|s|,2.6)+1.6*sin^2(angle/2)。
        const double angle = std::abs(bend) * 3.14159265358979323846 / 180.0;
        const std::vector<double> stations{station};
        const std::vector<double> distances{std::abs(station)};
        const std::vector<char> mixed{0};
        const double support = hingeSupport(stations, distances, mixed, 1.0, angle);
        const HingePoint got = hingeSection(station, depth, 1.0, 1.0, 0.0, 0.0, angle, support);
        const double diff = std::max(std::abs(got.x - row.at(3).number),
            std::abs(got.h - row.at(4).number));
        char detail[160];
        std::snprintf(detail, sizeof(detail), "bend=%.0f depth=%.2f s=%.2f support=%.6f",
            bend, depth, station, support);
        report("section", diff, worstSection, detail);
    }

    // ---------------- hinge_displacement ----------------
    for (const Value& c : root.at("hinge").array) {
        const double bend = c.at("bend").number;
        const double twist = c.at("twist").number;
        const double direction = c.at("direction").number;
        const double support = c.at("support").number;

        std::vector<double> stations, depths, laterals;
        for (const Value& v : c.at("stations").array) stations.push_back(v.number);
        for (const Value& v : c.at("depths").array) depths.push_back(v.number);
        for (const Value& v : c.at("laterals").array) laterals.push_back(v.number);

        std::vector<HingeWeight> weights;
        for (const Value& w : c.at("weights").array) {
            weights.push_back({w.at(0).number, w.at(1).number,
                w.array.size() > 2 ? w.at(2).number : 0.0});
        }

        HingeParams params;
        params.bend = bend;
        params.twist = twist;
        params.direction = direction;
        params.radius = 1.0;

        // 参考实现直接把 bind 位置当作 original；这里令 current = 0，
        // 于是 displacement 就等于 target，便于逐点比较。
        std::vector<Vec3> current(stations.size(), Vec3{0.0, 0.0, 0.0});
        const auto got = hingeDisplacement(stations, depths, laterals, current, weights, params);

        size_t index = 0;
        for (const Value& t : c.at("target").array) {
            const Vec3 want{t.at(0).number, t.at(1).number, t.at(2).number};
            const double diff = std::max({std::abs(got[index].x - want.x),
                std::abs(got[index].y - want.y), std::abs(got[index].z - want.z)});
            char detail[200];
            std::snprintf(detail, sizeof(detail),
                "bend=%.0f twist=%.0f dir=%.0f support=%.6f idx=%zu", bend, twist, direction,
                support, index);
            report("hinge", diff, worstHinge, detail);
            ++index;
        }
    }

    std::printf("\nworst deviation vs python reference:\n");
    std::printf("  elbow     %.3e\n", worstElbow);
    std::printf("  section   %.3e\n", worstSection);
    std::printf("  hinge     %.3e\n", worstHinge);
    if (failures == 0) std::printf("PASS: C++ port matches ocd_elbow_reference.py\n");
    else std::printf("%d mismatching sample(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
