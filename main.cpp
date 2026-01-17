#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>

#include <inttypes.h>
#include <memory.h>
#include <signal.h>
#include <sys/time.h>
#include <time.h>
#include <sys/stat.h>
#include <assert.h>
#include <dirent.h>

#include <x86intrin.h>
#include <cpuid.h>
#include <pthread.h>
#include <sched.h>

#include <valgrind/callgrind.h>

//#include "gf_profiling.c"

#include "imgui.h"
#include "fb.h"

#define eps 1e-8f

#define ARRSIZE(x) (sizeof(x)/sizeof(x[0]))
#define ARRSIZEi(x) ((int)(sizeof(x)/sizeof(x[0])))

uint64_t get_time_usec()
{
	static struct timeval _time_stamp;
	gettimeofday(&_time_stamp, nullptr);
	return _time_stamp.tv_sec * 1000000 + _time_stamp.tv_usec;
}

uint64_t get_time_nsec() {
	static struct timespec _time_stamp;
    clock_gettime(CLOCK_MONOTONIC, &_time_stamp);
	return _time_stamp.tv_sec * 1000000000 + _time_stamp.tv_nsec;
}

uint64_t rdtsc() {

    uint32_t low, high;
    __asm__ volatile("RDTSCP\n\t"
        "mov %%edx, %0\n\t"
        "mov %%eax, %1\n\t"
        "CPUID\n\t": "=r" (high), "=r" (low):: "%rax", "%rbx", "%rcx", "%rdx");
    uint64_t res = high;
    return (res<<32)|low;

    //return __rdtsc();
    //__asm__ volatile("rdtsc" : "=a"(low), "=d"(high));
}


template<typename T> T inline min(const T& a, const T& b) {
    return a < b ? a : b;
}
template<typename T> T inline max(const T& a, const T& b) {
    return a > b ? a : b;
}
template<typename T> T inline lerp(const T& a, const T& b, const T& v) {
    //return a + (b - a)*v;
    // better precision if b and a are very different( e.g. 1e+8f - 2 = 1e+8f)
    return a*(1 - v) + b*v;
}

template<typename T> T inline clamp(const T& a, const T& vmin, const T& vmax) {
    return min(max(a, vmin), vmax);
}
template<typename T> T inline min3(T a, T b, T c) {
    return min(a, min(b, c));
}
template<typename T> T inline max3(T a, T b, T c) {
    return max(a, max(b, c));
}

template<typename T> void inline swap(T& a, T& b) {
    T t = a;
    a = b;
    b = t;
}


#define cswap(a,b) do { if(a > b) { float tmp = a; a = b; b = tmp; } } while(0)
void sort3(int v[3]) {
    cswap(v[0], v[1]);
    cswap(v[1], v[2]);
    cswap(v[0], v[1]);
}

#define arrsize(a) (sizeof(a)/sizeof(a[0])) 

struct vec2 {
    vec2(float xx, float yy):x(xx), y(yy) {}
    vec2()/*:x(0), y(0)*/ {}
    explicit vec2(float v):x(v), y(v) {}
    float x,y;
};

vec2 operator*(const vec2& v, float x) {
    return vec2(v.x*x, v.y*x);
}
vec2 operator*(float x, const vec2& v) {
    return vec2(v.x*x, v.y*x);
}
vec2 operator*(const vec2& a, const vec2& b) {
    return vec2(a.x*b.x, a.y*b.y);
}
vec2 operator+(const vec2& a, const vec2& b) {
    return vec2(a.x + b.x, a.y + b.y);
}
vec2 operator-(const vec2& a, const vec2& b) {
    return vec2(a.x - b.x, a.y - b.y);
}

vec2 floor(const vec2& v) {
    return vec2(floor(v.x), floor(v.y));
}

vec2 fract(const vec2& v) {
    return vec2(v.x - floor(v.x), v.y - floor(v.y));
}

float dot(const vec2& a, const vec2& b) {
    return a.x*b.x + a.y*b.y;
}

float len(const vec2& v) {
    return sqrtf(dot(v, v));
}

vec2 lerp(const vec2& a, const vec2& b, float v) {
    return vec2(lerp(a.x, b.x, v), lerp(a.y, b.y, v));
}

//TODO:https://thebookofshaders.com/13/
float random(const vec2& st) {
    float v = sinf(dot(st, vec2(12.9898f,78.233f))) * 43758.5453123f;
    return v - floor(v);
}

struct vec3 {
    vec3(float xx, float yy, float zz):x(xx), y(yy), z(zz) {}
    vec3(float xx, float yy):x(xx), y(yy), z(0) {}
    //vec3():x(0), y(0), z(0) {}
    vec3(){}
    explicit vec3(float v):x(v),y(v),z(v) {}
    union {
       struct {
            float x,y,z;
       };
       float v[3];
    };
    operator float*() { return &x; }
    float operator[](int i) const { assert(i>=0 && i<3); return v[i]; }
};

vec3 operator*(const vec3& v, float x) {
    return vec3(v.x*x, v.y*x, v.z*x);
}
vec3 operator*(float x, const vec3& v) {
    return vec3(v.x*x, v.y*x, v.z*x);
}
vec3 operator+(const vec3& a, const vec3& b) {
    return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}
vec3 operator-(const vec3& a, const vec3& b) {
    return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

float dot(const vec3& a, const vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

vec3 cross(const vec3& a, const vec3& b) {
    return vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

float len(const vec3& v) {
    return sqrtf(dot(v, v));
}

vec3 normalize(const vec3& v) {
    float len_sq = dot(v, v);
    assert(len_sq > eps);
    return (1.0f/sqrtf(len_sq)) * v;
}

struct vec4 {
    vec4(float xx, float yy, float zz, float ww):x(xx), y(yy), z(zz), w(ww) {}
    vec4(float xx, float yy, float zz):x(xx), y(yy), z(zz), w(0) {}
    vec4(float xx, float yy):x(xx), y(yy), z(0), w(0) {}
    vec4(vec3 v, float ww):x(v.x), y(v.y), z(v.z), w(ww) {}
    vec4(vec3 v):x(v.x), y(v.y), z(v.z), w(0) {}
    vec4(vec2 v):x(v.x), y(v.y), z(0), w(0) {}
    explicit vec4(float v):x(v), y(v),z(v),w(v) {}
    vec4()/*:x(0), y(0), z(0), w(0)*/ {}
    //vec4& operator=(const vec4& o) = default;
        //x = o.x;y=o.y;z=o.z;w=o.w;
        //return *this;
    //}
    union {
        struct {
            float x,y,z,w;
        };
        float v[4];
    };
    vec3 xyz() const { return vec3(x, y, z); }
    vec2 xy() const { return vec2(x, y); }

    operator float*() { return &x; }
    float operator[](int i) const { assert(i>=0 && i < 4); return v[i]; }
    float& operator[](int i) { assert(i>=0 && i < 4); return v[i]; }
};

vec4 operator*(const vec4& v, float x) {
    return vec4(v.x*x, v.y*x, v.z*x, v.w*x);
}
vec4 operator*(float x, const vec4& v) {
    return vec4(v.x*x, v.y*x, v.z*x, v.w*x);
}
vec4 operator*(const vec4& a, const vec4& b) {
    return vec4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

vec4 operator+(const vec4& a, const vec4& b) {
    return vec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
vec4 operator-(const vec4& a, const vec4& b) {
    return vec4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
float dot(const vec4& a, const vec4& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

float len(const vec4& v) {
    return sqrtf(dot(v, v));
}

vec4 lerp(const vec4& a, const vec4 b, float t) {
    return (b - a)*t + a;
}

vec4 normalize(const vec4& v) {
    float len_sq = dot(v, v);
    assert(len_sq > eps);
    return (1.0f/sqrtf(len_sq)) * v;
}

vec4 floor(const vec4& v) {
    return vec4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
}

vec4 fract(const vec4& v) {
    return v - floor(v);
}

struct m44 {
    union {
        vec4 r[4];
        float m[16];
    };

    m44() {}
    m44(const m44& o) {
        r[0] = o.r[0]; r[1] = o.r[1]; r[2] = o.r[2]; r[3] = o.r[3];
    }

    static m44 identity() { 
        m44 m;
        memset(m.m, 0, sizeof(m));
        m.r[0].x = 1;
        m.r[1].y = 1;
        m.r[2].z = 1;
        m.r[3].w = 1;
        return m;
    }

    m44& operator=(m44&& o) {
        *this = o;
        return *this;
    }

    m44& operator=(const m44& o) {
        memcpy(m, o.m, sizeof(m));
        return *this;
    }

    float operator[](int i) const { assert(i>=0 && i < 16); return m[i]; }
    float& operator[](int i) { assert(i>=0 && i < 16); return m[i]; }

    vec4 col(int i) const { return vec4(m[i], m[i + 4], m[i + 8], m[i + 12]); }
};

template <typename T>
struct Pt {
    Pt(T x_, T y_):x(x_),y(y_) {}
    Pt() {}
    T x, y;
    Pt<T>& operator+(const Pt<T>& other) {
        x += other.x;
        y += other.y;
        return *this;
    }
};

vec4 mul(const m44& m, const vec4 v) {
    vec4 r;
    r.x = dot(m.r[0], v);
    r.y = dot(m.r[1], v);
    r.z = dot(m.r[2], v);
    r.w = dot(m.r[3], v);
    return r;
}

// res <= a * b;
m44 mul(const m44& a, const m44 b) {
    m44 m;
    for(int j=0;j<4; j++) {
        vec4 row_j = a.r[j];
        for(int i=0;i<4; i++) {
            vec4 col_i = b.col(i);
            m.m[4*j + i] = dot(row_j, col_i);
        }
    }
    return m;
}

m44 translation(const vec3& t) {
    m44 r = m44::identity();
    r.r[0][3] = t.x;
    r.r[1][3] = t.y;
    r.r[2][3] = t.z;
    r.r[3][3] = 1.0f;
    return r;
}

m44 transpose(const m44& m) {
    m44 r;
    r.r[0] = m.col(0);
    r.r[1] = m.col(1);
    r.r[2] = m.col(2);
    r.r[3] = m.col(3);
    return r;
}

m44 rotateXZ(const float angle_rad) {
    float c,s;
    sincosf(angle_rad, &s, &c);
    m44 m;
    m.r[0] = { c, 0, s, 0};
    m.r[1] = { 0, 1, 0, 0};
    m.r[2] = {-s, 0, c, 0};
    m.r[3] = { 0, 0, 0, 1};
    return m;
}

// rotate about X, pitch - rotate about Y, yaw - rotate about Z
m44 rotateXYZ(float roll, float pitch, float yaw) {

    float cx,sx; sincosf(roll, &sx, &cx);
    float cy,sy; sincosf(pitch, &sy, &cy);
    float cz,sz; sincosf(yaw, &sz, &cz);
    m44 m;
    m.r[0] = { cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sz, 0};
    m.r[1] = { cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz, 0};
    m.r[2] = {-sy,    sx*cy,            cx*cy, 0};
    m.r[3] = { 0, 0, 0, 1};
    return m;
}

m44 mat_from_quat(const vec4 q) {
    m44 m;

    m.r[0][0] = 1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2]);
    m.r[0][1] = 2.0f * (q[0] * q[1] - q[2] * q[3]);
    m.r[0][2] = 2.0f * (q[2] * q[0] + q[1] * q[3]);
    m.r[0][3] = 0.0f;

    m.r[1][0] = 2.0f * (q[0] * q[1] + q[2] * q[3]);
    m.r[1][1]= 1.0f - 2.0f * (q[2] * q[2] + q[0] * q[0]);
    m.r[1][2] = 2.0f * (q[1] * q[2] - q[0] * q[3]);
    m.r[1][3] = 0.0f;

    m.r[2][0] = 2.0f * (q[2] * q[0] - q[1] * q[3]);
    m.r[2][1] = 2.0f * (q[1] * q[2] + q[0] * q[3]);
    m.r[2][2] = 1.0f - 2.0f * (q[1] * q[1] + q[0] * q[0]);
    m.r[2][3] = 0.0f;

    m.r[3][0] = 0.0f;
    m.r[3][1] = 0.0f;
    m.r[3][2] = 0.0f;
    m.r[3][3] = 1.0f;

    return m;
}

m44 make_proj(float left, float right, float top, float bottom, float near, float far) {
    m44 m;
    m.r[0] = { 2*near/(right - left), 0                    , (right+left)/(right - left)    , 0                        };
    m.r[1] = { 0                    , 2*near/(top - bottom), (top + bottom )/(top - bottom) , 0                        };
    m.r[2] = { 0                    , 0                    , -(far + near)/(far - near)     , -2*far*near/(far - near) };
    m.r[3] = { 0                    , 0                    , -1                             , 0                        };

    return m;
}

Pt<float> view_xy(float xp, float yp, float zv, const m44 mproj) {
    Pt<float> pv;
    pv.x = (xp - mproj.m[4*0 + 2] * zv) / mproj.m[4*0 + 0];
    pv.y = (yp - mproj.m[4*1 + 2] * zv) / mproj.m[4*1 + 1];
    return pv;
}

m44 make_frustum(float fov_y_deg, float aspect_w_by_h, float near, float far) {
    const float deg2rad = M_PIf / 180.0f;
    const float half_vert_fov = fov_y_deg / 2.0f;

    float tangent = tan(half_vert_fov * deg2rad);
    float half_top = near * tangent;              // half height of near plane
    float half_right = half_top * aspect_w_by_h;          // half width of near plane
                                              //
    return make_proj(-half_right, half_right, -half_top, half_top, near, far);
#if 0
    m44 m;
    m.m[0]  =  near / half_right;
    m.m[5]  =  far / half_top;
    m.m[10] = -(near + far) / (far - near);
    m.m[11] = -1;
    m.m[14] = -(2 * far * near) / (far - near);
    m.m[15] =  0;
    return m;
#endif
}

// takes advantage knowing that it is a view matrix (normalized rotation + translation parts)
m44 view_invert(const m44& v) {
    m44 t;
    t.r[0] = vec4(v.col(0).xyz(), -v[3]);
    t.r[1] = vec4(v.col(1).xyz(), -v[7]);
    t.r[2] = vec4(v.col(2).xyz(), -v[11]);
    t.r[3] = vec4(0,0,0,1);
    return t;
}

///////////////////////////////////////////////////////////////////////////////
m44 make_lookat(const vec3& eye, const vec3& target, const vec3& up_dir)
{
    // -z direction because right handed coord system
    vec3 fwd = normalize(eye - target);

    vec3 left = cross(up_dir, fwd);
    left = normalize(left);

    // recompute the orthonormal up vector
    vec3 up = cross(fwd, left);

    m44 m = m44::identity();

    // set rotation part, inverse rotation matrix: M^-1 = M^T for Euclidean transform
    m[0] = left.x;
    m[1] = left.y;
    m[2] = left.z;
    m[4] = up.x;
    m[5] = up.y;
    m[6] = up.z;
    m[8] = fwd.x;
    m[9] = fwd.y;
    m[10]= fwd.z;

    // set translation part
    m[3]= -dot(left, eye);
    m[7]= -dot(up, eye);
    m[11]= -dot(fwd, eye);

    return m;
}


typedef uint8_t u8;
typedef uint16_t u16;
typedef int16_t i16;
typedef uint32_t u32;
typedef int32_t i32;
typedef uint64_t u64;


// Based on Morgan McGuire @morgan3d https://www.shadertoy.com/view/4dS3Wd
float noise (vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (vec2(3.0f) - 2.0f * f);

    return  (b - a) * u.x + a +
            (c - a) * u.y * (1.0f - u.x) +
            (d - b) * u.x * u.y;
}

#define OCTAVES 6
float fbm (vec2 st) {
    // Initial values
    float value = 0.0;
    float amplitude = .5;
    //float frequency = 0.;
    //
    // Loop of octaves
    for (int i = 0; i < OCTAVES; i++) {
        value += amplitude * noise(st);
        st = 2.0f*st;
        amplitude *= .5f;
    }
    return value;
}



typedef short FP16; // 8.8

// POD only 
template<typename T, typename TSize = size_t>
class BufferT {
    enum { elSize = sizeof(T) };
    T* data_;
    TSize size_;
    TSize capacity_;

public:

    BufferT():data_(nullptr), size_(0), capacity_(0) { }

    void reset() {
        size_ = 0;
    }

    bool resize(TSize new_cap, bool b_shrink = false) {
        if(new_cap > capacity_ || (new_cap < capacity_ && b_shrink)) {

            T* new_data = (T*)malloc(elSize*new_cap);
            if(!new_data) {
                return false;
            }

            capacity_ = new_cap;
            size_ = new_cap < size_ ? new_cap : size_;

            if(data_) {
                memcpy(new_data, data_, size_*elSize);
                free(data_);
            }
            data_ = new_data;
        }
        return true;
    }

    const T& operator[](TSize i) const {
        assert(i < size_);
        return data_[i];
    }

    T& operator[](TSize i) {
        assert(i < size_);
        return data_[i];
    }

    void push(T el) {
        if(size_ == capacity_) {
            resize(capacity_ == 0 ? 16 : 2*capacity_);
        }
        data_[size_++] = el;
    }

    void remove_swap(int i) {
        assert(i < size_);
        data_[i] = data_[size_-1];
        size_--;
    }

    T& last() {
        assert(size_);
        return data_[size_ - 1];
    }

    T* data() { return data_; }
    const T* data() const { return data_; }

    TSize size() const { return size_; }

};

//////////////////////////////////////////////////////////////////////////
/// Str - string 

struct Str {
    const char* data;
    ptrdiff_t len;
};

struct Cut {
    Str head;
    Str tail;
    bool b_ok;
};

Str from_charptr(const char* ptr) {
    return { ptr, (ptrdiff_t)(ptr ? strlen(ptr) + 1: 0) };
}

ptrdiff_t findleft(Str s, char c) {
    ptrdiff_t i=0;
    for(; i<s.len; ++i) {
        if(s.data[i] == c) {
            return i;
        }
    }
    return i;
}

Str trimleft(Str s, char c) {
    for(; s.len && *s.data == c; s.data++, s.len--) {}
    return s;
}

Str trimright(Str s, char c) {
    for(; s.len && s.data[s.len-1] == c; s.len--) {}
    return s;
}

Str trimleft_le(Str s, char c) {
    for(; s.len && *s.data <= c; s.data++, s.len--) {}
    return s;
}

Str trimright_le(Str s, char c) {
    for(; s.len && s.data[s.len-1] <= c; s.len--) {}
    return s;
}

Str substr(Str s, ptrdiff_t offset) {
    ptrdiff_t off = min(offset, s.len);
    s.data += off;
    s.len -= off;
    return s;
}

bool streq(Str a, Str b) {
    return a.len == b.len && 0 == memcmp(a.data, b.data, a.len);
}

Cut cut(Str s, char c) {
    Cut rv;
    ptrdiff_t i = findleft(s, c);
    rv.b_ok = i != s.len;
    rv.head = { s.data, i };
    rv.tail = substr(s, i + (rv.b_ok?1:0));
    return rv;
}

void serialize(const Str& s, FILE* fp) {
    fwrite(&s.len, sizeof(s.len), 1, fp);
    fwrite(s.data, s.len, 1, fp);
}

extern struct StrArena g_string_arena;
Str alloc_str_from_arena(struct StrArena& arena, const int len, char** strptr);
Str deserialize(FILE* fp) {
    Str s;
    fread(&s.len, sizeof(s.len), 1, fp);
    char* strptr = 0;
    s = alloc_str_from_arena(g_string_arena, s.len, &strptr);
    fread(strptr, s.len, 1, fp);
    return s;
}


char* print(char* buf, Str s) {
    memcpy(buf, s.data, s.len);
    buf[s.len] = '\0';
    return buf;
}

void test_str() {
    const char* t[] = {
        " some string to   test the stuff  ",
        "some test the stuff  ",
        "   some string",
        "  frontspaces",
        "endspaces   ",
        "lalala",
        "hello, world",
        "just,a,string, to  , split  ,me, in,parts ",
        nullptr
    };

    int idx = 0;
    char buf[128];
    char srcbuf[128];
    while(t[idx]) {

        printf("Original: %s\n", t[idx]);

        const ptrdiff_t len = strlen(t[idx]);
        memcpy(srcbuf, t[idx], len);

        Str s = { srcbuf, len };

        Str tl = trimleft(s, ' ');
        printf("tl:'%s'\n", print(buf, tl));

        Str tr = trimright(tl, ' ');
        printf("tr:'%s'\n", print(buf, tr));

        Cut c;
        c.tail = s;
        c.b_ok = true;
        while(c.b_ok) {
            c = cut(trimleft(c.tail, ' '), ' ');
            printf("head:%s\n", print(buf, c.head));
            if(c.b_ok) {
                printf("tail:%s\n", print(buf, c.tail));
            }
        }

        c.tail = s;
        c.b_ok = true;
        while(c.b_ok) {
            c = cut(trimleft(c.tail, ' '), ',');
            printf("\thead:'%s'\n", print(buf, c.head));
            if(c.b_ok) {
                printf("\ttail:'%s'\n", print(buf, c.tail));
            }
        }

        idx++;
    }
}

struct StrArena {
    char* mem;
    int size;
    int capacity;
} g_string_arena;

StrArena make_str_arena(int size) {
    return { (char*)malloc(size), 0, size };
};

void destroy_str_arena(StrArena& a) {
    free(a.mem);
    a.size = a.capacity = 0;
}

Str alloc_str_from_arena(StrArena& arena, const int len, char** strptr) {

    char* const arena_free = arena.mem + arena.size;
    assert(arena.size + len < arena.capacity);
    arena.size += len;
    if(strptr) *strptr = arena_free;
    return { arena_free, len };
}


// always allocates null terminated string
Str alloc_str_from_arena(StrArena& arena, const char* str) {
    char* const arena_free = arena.mem + arena.size;
    const int len = strlen(str) + 1;
    assert(arena.size + len < arena.capacity);

    memcpy(arena_free, str, len-1);
    arena_free[len-1] = '\0';

    Str s = { arena_free, len };

    arena.size += len;
    return s;
}

// always allocates null terminated string
Str alloc_str_from_arena(StrArena& arena, const char* str, int len) {
    char* start = nullptr;
    Str s = alloc_str_from_arena(arena, len, &start);
    memcpy(start, str, len - 1);
    start[len - 1] = '\0';
    return s;
}

// always allocates null terminated string
Str alloc_concat_from_arena(StrArena& arena, Str a, Str b) {
    char* pstr = nullptr;
    Str s = alloc_str_from_arena(arena, a.len + b.len + 1, &pstr);

    memcpy(pstr, a.data, a.len);
    memcpy(pstr + a.len, b.data, b.len);
    pstr[a.len + a.len] = '\0';

    return s;
}


///////////////////////////////////////////////////////////////////////////
///  TGA Loader
// http://www.paulbourke.net/dataformats/tga/
#pragma pack(push, 1)
typedef struct {
   char  idlength;
   char  colourmaptype;
   char  datatypecode;
   short int colourmaporigin;
   short int colourmaplength;
   char  colourmapdepth;
   short int x_origin;
   short int y_origin;
   short width;
   short height;
   char  bitsperpixel;
   char  imagedescriptor;
} TGA_HEADER;
#pragma pack(pop)

enum class eTGADataType:char {
    kNoData = 0, //  -  No image data included.
    kUncompressedColorMapped = 1, //  -  Uncompressed, color-mapped images.
    kUncompressedRGB = 2, //  -  Uncompressed, RGB images.
    kUncompressedBW = 3, //  -  Uncompressed, black and white images.
    kRLEColorMapped = 9,//  -  Runlength encoded color-mapped images.
    kRLERGB = 10, //  -  Runlength encoded RGB images.
    kRLEBW = 11, //  -  Compressed, black and white images.
    kCompressedColorMaped = 32, //  -  Compressed color-mapped data, using Huffman, Delta, and runlength encoding.
    kCompressed4Pass = 33, //   -  Compressed color-mapped data, using Huffman, Delta, and runlength encoding.  4-pass quadtree-type process.
};

struct Image {
    static void destroy(Image* img) {
        free(img->data);
        free(img);
    }
    struct Pixel {
        u8 r, g, b, a;
        u32 asU32() const { 
            return (u32)b | ((u32)g<<8) | ((u32)r<<16) | ((u32)a<<24);
        }
        vec4 asVec4() const { 
            return (1.0f/255.0f)*vec4(r, g, b, a);
        }
    };

    u32 width, height, bpp;
    Pixel* data;
    Str filename_;
};

Image::Pixel tga_merge_bytes(const u8 *p, int bytes) {
    Image::Pixel pix;

    if (bytes == 4) {
        pix.r = p[2];
        pix.g = p[1];
        pix.b = p[0];
        pix.a = p[3];
    } else if (bytes == 3) {
        pix.r = p[2];
        pix.g = p[1];
        pix.b = p[0];
        pix.a = 255;
    } else if (bytes == 2) {
        pix.r = (p[1] & 0x7c) << 1;
        pix.g = ((p[1] & 0x03) << 6) | ((p[0] & 0xe0) >> 2);
        pix.b = (p[0] & 0x1f) << 3;
        pix.a = (p[1] & 0x80);
    }

    return pix;
}

Image* tga_load(const char* filename) {

    FILE* f = fopen(filename, "r");
    if(!f) return nullptr;

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if(size < sizeof(TGA_HEADER)) {
        puts("Malformed TGA file");
        fclose(f);
        return nullptr;
    }

    TGA_HEADER hdr;
    fread(&hdr, sizeof(TGA_HEADER), 1, f);

    if(hdr.bitsperpixel != 16 && hdr.bitsperpixel != 24 && hdr.bitsperpixel != 32) {
       puts("Only bpp of 16, 24 or 32 is supported");
       fclose(f);
       return nullptr;
    }

    if (hdr.datatypecode != (char)eTGADataType::kUncompressedRGB && 
            hdr.datatypecode != (char)eTGADataType::kRLERGB) {
        puts("Can only handle TGA image type 2 and 10");
        return nullptr;
    }

    if (hdr.colourmaptype != 0 && hdr.colourmaptype != 1) {
        puts("Can only handle colour map types of 0 and 1\n");
        return nullptr;
    }

    int skipover = hdr.idlength + hdr.colourmaptype * hdr.colourmaplength;
    fprintf(stderr,"TGA Skip over %d bytes\n",skipover);
    fseek(f, skipover,SEEK_CUR);

    const int bytes_per_pix = (hdr.bitsperpixel/8);
    const int datasize = hdr.width*hdr.height*bytes_per_pix;
    u8* pmem = (u8*)malloc(datasize);
    size_t num_read = fread(pmem, datasize, 1, f);
    if(num_read !=1) {
        puts("TGA: Unexpected EOF when reading data\n");
        fclose(f);
        return nullptr;
    }

    Image* img = (Image*)malloc(sizeof(Image));
    img->bpp = hdr.bitsperpixel;
    img->width = hdr.width;
    img->height = hdr.height;
    img->data = (Image::Pixel*)malloc(hdr.width*hdr.height*sizeof(Image::Pixel));

    const int npix = img->width*img->height;
    if (hdr.datatypecode == (char)eTGADataType::kUncompressedRGB) {
        for(int pi = 0; pi < npix; pi++) {
            img->data[pi] = tga_merge_bytes(pmem + pi*bytes_per_pix, bytes_per_pix);
        }
    }

    free(pmem);
    fclose(f);
    return img;
}


///////////////////////////////////////////////////////////////////////////
/// OBJ Parser

struct ObjModel {
    struct V {
        int v, t, n;
    };
    BufferT<vec3> v;
    BufferT<vec2> t;
    BufferT<vec3> n;

    BufferT<V> ib;
    Str filename_;
};

void parse_comment(ObjModel* obj, Str s) {
}

bool parse_float_array(Str s, float* fa, int cnt) {

    Cut c;
    c.tail = s;
    c.b_ok = true;

    bool rv = true;

    for(int i=0;i<cnt;++i) {
        if(!c.b_ok) {
            fa[i] = 0;
            rv &= false;
        } else {
            c = cut(trimleft_le(c.tail, ' '), ' ');
            fa[i] = strtof(c.head.data, nullptr);
        }
    }

    return rv;
}

bool parse_vertex(ObjModel* obj, Str s) {
    vec3 v;
    bool rv = !parse_float_array(s, (float*)&v, 3);
    if(rv) {
        char buf[128];
        printf("Warning, failed to parse vertex: %s\n", print(buf, s));
    } else {
        printf("V: %f %f %f\n", v.x, v.y, v.z);
    }
    obj->v.push(v);
    return rv;
}

bool parse_texcoord(ObjModel* obj, Str s) {
    vec2 v;
    bool rv = !parse_float_array(s, (float*)&v, 2);
    if(rv) {
        char buf[128];
        printf("Warning, failed to parse texcoord: %s\n", print(buf, s));
    } else {
        printf("T: %f %f\n", v.x, v.y);
    }
    obj->t.push(v);
    return rv;
}

bool parse_normal(ObjModel* obj, Str s) {
    vec3 v;
    bool rv = !parse_float_array(s, (float*)&v, 3);
    if(rv) {
        char buf[128];
        printf("Warning, failed to parse normal: %s\n", print(buf, s));
    } else {
        printf("N: %f %f %f\n", v.x, v.y, v.z);
    }
    obj->n.push(v);
    return rv;
}

bool parse_triplet(Str s, int* vtn) {

    bool rv = true;

    Cut c;
    c.b_ok = true;
    c.tail = s;

    for(int i=0;i<3;++i) {
        if(!c.b_ok) {
            rv &= false;
        } else {
            c = cut(trimleft_le(c.tail, ' '), '/');
            vtn[i] = atoi(c.head.data);
        }
    }
    return rv;
}

bool parse_face(ObjModel* obj, Str s) {
    Cut c;
    c.b_ok = true;
    c.tail = s;

    int vtn[3*3] = {0};

    bool rv = true;
    for(int i=0;i<3;++i) {

        if(c.b_ok) {
            c = cut(trimleft_le(c.tail, ' '), ' ');
            rv &= parse_triplet(trimleft_le(c.head, ' '), vtn + 3*i);
        } else {
            rv &= false;
        }

    }

    printf("F: [%d, %d, %d] [%d, %d, %d] [%d, %d, %d]\n",
            vtn[0], vtn[1], vtn[2], 
            vtn[3], vtn[4], vtn[5], 
            vtn[6], vtn[7], vtn[8]);

    // 1based indexing -> 0 based
    obj->ib.push({vtn[0]-1, vtn[1]-1, vtn[2]-1});
    obj->ib.push({vtn[3]-1, vtn[4]-1, vtn[5]-1});
    obj->ib.push({vtn[6]-1, vtn[7]-1, vtn[8]-1});
    
    return rv;
}

bool parse_objname(ObjModel* obj, Str s) {
    return true;
}


ObjModel* parse_obj(const char* data) {
    FILE* f = fopen(data, "r");
    if(!f) return nullptr;

    fseek(f, 0, SEEK_END);
    ptrdiff_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* arena = (char*)malloc(size);
    ptrdiff_t num = fread(arena, size, 1, f); (void)num;
    assert(num == 1);

    //char buf[128];
    ObjModel* obj = new ObjModel;

    Cut c;
    c.tail = { arena, size };
    c.b_ok = true;

    int line = 0;
    while(c.b_ok) {
        c = cut(c.tail, '\n');
        line++;

        Cut fields = cut(trimright_le(c.head, ' '), ' ');

        if(streq(fields.head, { "#", 1 }))  {
            parse_comment(obj, fields.tail);
        } else if(streq(fields.head, { "v", 1 }))  {
            parse_vertex(obj, fields.tail);
        } else if(streq(fields.head, { "vn", 2 }))  {
            parse_normal(obj, fields.tail);
        } else if(streq(fields.head, { "vt", 2 }))  {
            parse_texcoord(obj, fields.tail);
        } else if(streq(fields.head, { "f", 1 }))  {
            parse_face(obj, fields.tail);
        } else if(streq(fields.head, { "o", 1 }))  {
            parse_objname(obj, fields.tail);
        }
        line++;
    }

    fclose(f);
    free(arena);

    return obj;
}

/////////////////////////////////////////////////////////////////////////////////
/// Object selector

struct DirList { 
    StrArena arena;
    Str base_path;
    BufferT<Str, int> entries;
};

DirList make_dir_list(const char* base_path, const char* ext) {

    DirList dl;

    DIR *dir;
    struct dirent *entry;

    if(!(dir = opendir(base_path))) {
        perror("Error opening directory");
        return dl;
    }

    dl.arena = make_str_arena(16*1024);
    dl.base_path = alloc_str_from_arena(dl.arena, base_path);

    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
            const char* e_ext = strrchr(entry->d_name, '.');
            if(!ext || (e_ext && strcmp(ext, e_ext+1)==0)) {
                printf("%s\n", entry->d_name);
                Str s = alloc_str_from_arena(dl.arena, entry->d_name);
                dl.entries.push(s);
            }
        }
    }

    closedir(dir);
    return dl;
}

void destroy_dir_list(DirList& dl) {
    dl.entries.reset();
    destroy_str_arena(dl.arena);
}

//#define Integer 12
//#define Fractional 4 
#define MAX_FP 0x7FFF
#define MIN_FP 0x8000

template<u8 Fractional>
FP16 toFP(float v) {
    constexpr const float k = (1<<Fractional);
    v = v * k + 0.5f*(v>=0?1:-1);
    assert(v < MAX_FP);
    assert(v > -MAX_FP);
    return (FP16)v;
}

template<u8 Fractional, typename T = FP16>
float fromFP(T v) {
    constexpr const float k = 1.0f/(1<<Fractional);
    return v * k;
}

// can't be
template<u8 Fractional> float fromFP(float v);

typedef float D;
i32 g_tex_w = 320;
i32 g_tex_h = 200;
DynamicTexture* g_dyn_texture;
u32* g_fb;
u32 g_fb_size_bytes; 
D* g_fb_z;
u32 g_fb_z_size_bytes; 
Image* g_current_texture = nullptr;

#define CULL_FACE_NONE 0
#define CULL_FACE_CW 1
#define CULL_FACE_CCW 2
u8 g_cull_face = CULL_FACE_NONE; // 0 - none, true; // false by default?

bool do_exit = false;


Pt<float> NDC2Win(float x, float y, float w, float h) {
    return Pt<float>((x*0.5f + 0.5f)*w, (y*0.5f+0.5f)*h);
}

Pt<float> Win2NDC(float x, float y, float w, float h) {
    return Pt<float>((x/w - 0.5f)*2.0f, (y/h - 0.5f)*2.0f);
}

template<typename T> 
T edge(const Pt<T>& a, const Pt<T>& b, const Pt<T>& c) {
    return -(b.y - a.y)*(c.x - a.x) + (b.x - a.x)*(c.y - a.y);
};

template<typename T> 
float edge(const T& a, const T& b, const T& c) {
    return -(b.y - a.y)*(c.x - a.x) + (b.x - a.x)*(c.y - a.y);
};


/*        CW
//           2
//          / \
//       _ /   \
//        /|   _\|
//       /       \
//      /         \
//    0 ----<------ 1
*/

template <typename T>
struct TriSetup {

    T a[3];
    T b[3];
    T c[3];
    bool t[3];

    vec4 v0,v1,v2;
    vec4 p[3]; // debug
    Pt<T> v[3];

    float oo2A;

    TriSetup(const vec4& p0, const vec4& p1, const vec4& p2) {

        vec4 pt0 = g_cull_face == CULL_FACE_CCW ? p1 : p0;
        vec4 pt1 = g_cull_face == CULL_FACE_CCW ? p0 : p1;
        vec4 pt2 = p2;

        p[0] = pt0;
        p[1] = pt1;
        p[2] = pt2;

        setup(NDC2Win(pt0.x/pt0.w, pt0.y/pt0.w, g_tex_w, g_tex_h), 
                NDC2Win(pt1.x/pt1.w, pt1.y/pt1.w, g_tex_w, g_tex_h),
                NDC2Win(pt2.x/pt2.w, pt2.y/pt2.w, g_tex_w, g_tex_h));
    }

    void setup(const Pt<T>& pt0, const Pt<T>& pt1, const Pt<T>& pt2) {

        v[0] = pt0;
        v[1] = pt1;
        v[2] = pt2;

        b[0] = pt2.x - pt1.x;
        a[0] = -(pt2.y - pt1.y);
        t[0] = a[0]!=0 ? a[0]>0 : b[0]<0;

        b[1] = pt0.x - pt2.x;
        a[1] = -(pt0.y - pt2.y);
        t[1] = a[1]!=0 ? a[1]>0 : b[1]<0;

        b[2] = pt1.x - pt0.x;
        a[2] = -(pt1.y - pt0.y);
        t[2] = a[2]!=0 ? a[2]>0 : b[2]<0;

        // TODO: handle if area is 0 so that oo2A is inf
        oo2A = 1.0f/edge(v[0], v[1], v[2]);

    }

    bool inside0(const Pt<T>& p) const {
        int r = e0(p);
        return r > 0 || (r == 0 && t[0]);
    }
    bool inside1(const Pt<T>& p) const {
        int r = e1(p);
        return r > 0 || (r == 0 && t[1]);
    }
    bool inside2(const Pt<T>& p) const {
        int r = e2(p);
        return r > 0 || (r == 0 && t[2]);
    }

    bool inside0(int e) { return e > 0 || (e == 0 && t[0]); }
    bool inside1(int e) { return e > 0 || (e == 0 && t[1]); }
    bool inside2(int e) { return e > 0 || (e == 0 && t[2]); }

    bool inside(const Pt<T>& p) {
        return inside0(e0(p)) && inside1(e1(p)) && inside2(e2(p));
    }

    T e0(const Pt<T>& p) const { return a[0]*(p.x - v1.x) + b[0]*(p.y - v1.y); }
    T e1(const Pt<T>& p) const { return a[1]*(p.x - v2.x) + b[1]*(p.y - v2.y); }
    T e2(const Pt<T>& p) const { return a[2]*(p.x - v0.x) + b[2]*(p.y - v0.y); }

    T e0dx(T x) const { return x*a[0]; }
    T e1dx(T x) const { return x*a[1]; }
    T e2dx(T x) const { return x*a[2]; }

    T e0dy(T x) const { return x*b[0]; }
    T e1dy(T x) const { return x*b[1]; }
    T e2dy(T x) const { return x*b[2]; }

    bool e(const Pt<T>& p) const { return e0(p) | e1(p) | e2(p); }
};

#define FP_FracBits 4
#define HalfFp (1<<(FP_FracBits-1))  // aka 0.5
constexpr const i32 FP_MaxInteger = 1 << (sizeof(FP16)*8 - FP_FracBits - 1);

FP16 toFixed(float v) {
    return toFP<FP_FracBits>(v);
}

FP16 toFixed(i32 v) {
    assert(v < FP_MaxInteger);
    return (FP16)(v<<FP_FracBits);
}

Pt<FP16> toFixed(const Pt<float>& pt) {
    return Pt<FP16>(toFixed(pt.x), toFixed(pt.y));
}

template<typename T>
float fromFixed(T v) {
    return fromFP<FP_FracBits, T>(v);
}

template<>
float fromFixed(float v) {
    return v;
}

Pt<float> fromFixed(const Pt<FP16>& v) {
    return Pt<float>(fromFP<FP_FracBits>(v.x), fromFP<FP_FracBits>(v.y));
}

Pt<float> fromFixed(const Pt<float>& v) {
    return v;
}

int fromFixed2Int(FP16 v) {
    return v >> FP_FracBits;
}

Pt<FP16> float2FPSnapped(float x, float y) {

    u32 bits2remove = (FP_FracBits + 1)/2;
    u32 mask = (1<<bits2remove) - 1;
    u32 onehalf = 1<<(bits2remove - 1);
    return Pt<FP16>((toFixed(x) + onehalf)&~mask, (toFixed(y) + onehalf)&~mask);
}

ImVec2 FixedToUI(Pt<FP16> v, int scale) {
    return ImVec2(fromFixed(v.x)*scale, fromFixed(v.y)*scale);
}

float ToUI(FP16 v, int scale) {
    return fromFixed(v)*scale;
}

const constexpr int ATT_COUNT = 3;
const constexpr int ATT_COLOR = 0;
const constexpr int ATT_TEXCOORD= 1;
const constexpr int ATT_NORMAL= 2;


template <>
struct TriSetup<FP16> {
    /*
    struct A { FP16 a0, a1, a2; };
    union {
        FP16 aa[3];
        A a;
    };
    */
    FP16 a[3];
    FP16 b[3];
    i32  c[3];
    bool t[3];

    Pt<FP16> v[3];
    float hz[3];
    float hz_pc[3]; // perspercitve corr
    float hw[3];

    float ooHw[3];

    float oo2A;
    vec4 attribs[ATT_COUNT][3]; // TODO pointer to attrib indices
    vec4 oo_attribs[ATT_COUNT][3]; // TODO pointer to attrib indices

    TriSetup() {};

    TriSetup(const vec4& p0, const vec4& p1, const vec4& p2) {

        vec4 pt0 = g_cull_face == CULL_FACE_CCW ? p1 : p0;
        vec4 pt1 = g_cull_face == CULL_FACE_CCW ? p0 : p1;
        vec4 pt2 = p2;

        hz[0] = pt0.z;
        hz[1] = pt1.z;
        hz[2] = pt2.z;
        hw[0] = pt0.w;
        hw[1] = pt1.w;
        hw[2] = pt2.w;


        //setup(Pt<float>(pt0.x/pt0.w, pt0.y/pt0.w), Pt<float>(pt1.x/pt1.w, pt1.y/pt1.w), Pt<float>(pt2.x/pt2.w, pt2.y/pt2.w));
        setup(NDC2Win(pt0.x/pt0.w, pt0.y/pt0.w, g_tex_w, g_tex_h), 
                NDC2Win(pt1.x/pt1.w, pt1.y/pt1.w, g_tex_w, g_tex_h),
                NDC2Win(pt2.x/pt2.w, pt2.y/pt2.w, g_tex_w, g_tex_h));
    }

    //TriSetup(const Pt<int>& pt0, const Pt<int>& pt1, const Pt<int>& pt2) {
    //    setup(pt0, pt1, pt2);
    //}

    //TriSetup(const Pt<float>& pt0, const Pt<float>& pt1, const Pt<float>& pt2) {
    //    setup(pt0, pt1, pt2);
    //}

    void set_attribs(int idx, const vec4& a0, const vec4& a1, const vec4& a2) {
        assert(idx >=0 && idx < ATT_COUNT);
        attribs[idx][0] = g_cull_face == CULL_FACE_CCW ? a1 : a0;
        attribs[idx][1] = g_cull_face == CULL_FACE_CCW ? a0 : a1;
        attribs[idx][2] = a2;

        // perspercitve corrected
        oo_attribs[idx][0] = attribs[idx][0]*ooHw[0];
        oo_attribs[idx][1] = attribs[idx][1]*ooHw[1];
        oo_attribs[idx][2] = attribs[idx][2]*ooHw[2];
    }

private:
    template<typename A>
    void setup(const Pt<A>& pt0, const Pt<A>& pt1, const Pt<A>& pt2) {

        // I snap vertices (and not just toFixed()) to FP_FracBits-1 precision, loosing one precision bit.
        // So vertices are snapped to a more coarse grid.
        // This is done to avoid precision issues when calculating edge functions
        // E.g. for 2 tris sharing an edge: e0 for tri1 and e1 for tri1, 
        // there could be a situation when 1st tri will have e.g. 
        // e0(p): aa + bb = 15 (then shifted by FP_FracBits will be 0) - point not inside if not left/top edge
        // another will have e1(p): aa + bb = -15 (then shifted by FP_FracBits will be -1) - point not inside
        // so if 1st tri e0 is not left/top edge, it will also be counted as point is not inside
        // so none of the 2 tris contains pixes -> hole
        // Another way to solve this is to not do a final shift when calculating edge functios,
        // effectively leaving 2*FP_FracBits precision (see traverse_aabb FpPrec)
        // But for now it seems that adding "0.5" in edge calculation fixes the issue as well, see e0/e1/e2 functions 

        //v[0] = float2FPSnapped(pt0.x, pt0.y);
        //v[1] = float2FPSnapped(pt1.x, pt1.y);
        //v[2] = float2FPSnapped(pt2.x, pt2.y);

        v[0] = toFixed(pt0);
        v[1] = toFixed(pt1);
        v[2] = toFixed(pt2);
        
        //                a                           b
        // e(x,y) = -(v2.y - v1.y)*(x - v1.x) + (v2.x - v1.x)*(y - v1.y)
        //        = a*x - a*v1.x + b*y - b*v1.y 
        //                           c
        //        = a*x + b*y -(a*v1.x + b*v1.y)
        
        //                a                           b
        // e(x,y) = -(v2.y - v1.y)*(x - v1.x) + (v2.x - v1.x)*(y - v1.y)

        b[0] = v[2].x - v[1].x;
        a[0] = -(v[2].y - v[1].y);
        c[0] = -((i32)a[0]*v[1].x + (i32)b[0]*v[1].y);
        t[0] = a[0]!=0 ? a[0]>0 : b[0]>0;

        b[1] = v[0].x - v[2].x;
        a[1] = -(v[0].y - v[2].y);
        c[1] = -((i32)a[1]*v[2].x + (i32)b[1]*v[2].y);
        t[1] = a[1]!=0 ? a[1]>0 : b[1]>0;

        b[2] = v[1].x - v[0].x;
        a[2] = -(v[1].y - v[0].y);
        c[2] = -((i32)a[2]*v[0].x + (i32)b[2]*v[0].y);
        t[2] = a[2]!=0 ? a[2]>0 : b[2]>0;

        //FIXME: looks like we actually need to calculate oo2A using fp coords, as then 
        // we use it to multiply to get u,v,w and if calculating using floats, barycentric could be > 1
        // so can just calc area using floats to get CW/CCW ordering, but store actual area calculated with fp
        
        // TODO: handle if area is 0 so that oo2A is inf
        oo2A = 1.0f/edge(fromFixed(v[0]), fromFixed(v[1]), fromFixed(v[2]));
        // calculte area using floats as fixed point (12.4) do not have enough precision
        // for very thin large triangles (predominantly in hor or vert orientation, 
        // due to small a.x - b.x or a.y - b.y differences) and may flip CW to CCW
        //float TwiceArea = edge(pt0, pt1, pt2);
#if 0 // now moved to traverse_aabb
        // even with floats, there could be issues with thin triangles
        if(TwiceArea < 0.001f) {
            TwiceArea = 0;
            oo2A = INFINITY;
            assert(isinf(INFINITY));
        } else {
            oo2A = 1.0f/TwiceArea;
        }
#else
        //oo2A = 1.0f/TwiceArea;
#endif

        ooHw[0] = 1.0f/hw[0];
        ooHw[1] = 1.0f/hw[1];
        ooHw[2] = 1.0f/hw[2];

        hz_pc[0] = hz[0]*ooHw[0];
        hz_pc[1] = hz[1]*ooHw[1];
        hz_pc[2] = hz[2]*ooHw[2];
    }
public:

    bool inside0(const Pt<FP16>& p) const {
        int r = e0(p);
        return r > 0 || (r == 0 && t[0]);
    }
    bool inside1(const Pt<FP16>& p) const {
        int r = e1(p);
        return r > 0 || (r == 0 && t[1]);
    }
    bool inside2(const Pt<FP16>& p) const {
        int r = e2(p);
        return r > 0 || (r == 0 && t[2]);
    }

    bool inside(const Pt<FP16>& p) const { return inside0(p) && inside1(p) && inside2(p); }

    bool inside0(int e) const { return e > 0 || (e == 0 && t[0]); }
    bool inside1(int e) const { return e > 0 || (e == 0 && t[1]); }
    bool inside2(int e) const { return e > 0 || (e == 0 && t[2]); }

    bool inside(const Pt<FP16>& p) {
        return inside0(e0(p)) && inside1(e1(p)) && inside2(e2(p));
    }

    i32 e0_alt(const Pt<FP16>& p) const {
        // c is already 2*FP_FracBits, so can add safely
        i32 r = (i32)a[0]*p.x + (i32)b[0]*p.y + c[0];
        return r >> FP_FracBits;
    }
    i32 e1_alt(const Pt<FP16>& p) const {
        i32 r = (i32)a[1]*p.x + (i32)b[1]*p.y + c[1];
        return r >> FP_FracBits;
    }
    i32 e2_alt(const Pt<FP16>& p) const {
        i32 r = (i32)a[2]*p.x + (i32)b[2]*p.y + c[2];
        return r >> FP_FracBits;
    }

    template<bool HiPrec = false>
    i32 e0(const Pt<FP16>& p) const {
        i32 dx = p.x - v[1].x;
        i32 aa = ((i32)a[0] * dx);

        i32 dy = p.y - v[1].y;
        i32 bb = ((i32)b[0] * dy);
        // adding .5 improves precision
        return HiPrec ? aa + bb : (aa + bb + HalfFp) >> FP_FracBits;
    }

    template<bool HiPrec = false>
    i32 e1(const Pt<FP16>& p) const {
        i32 dx = p.x - v[2].x;
        i32 aa = ((i32)a[1] * dx);

        i32 dy = p.y - v[2].y;
        i32 bb = ((i32)b[1] * dy);
        // adding .5 improves precision
        return HiPrec ? aa + bb : (aa + bb + HalfFp) >> FP_FracBits;
    }

    template<bool HiPrec = false>
    i32 e2(const Pt<FP16>& p) const {
        i32 dx = p.x - v[0].x;
        i32 aa = ((i32)a[2] * dx);

        i32 dy = p.y - v[0].y;
        i32 bb = ((i32)b[2] * dy);
        // adding .5 improves precision
        return HiPrec ? aa + bb : (aa + bb + HalfFp) >> FP_FracBits;
    }

    i32 e0dx(FP16 x) const {
        i32 c = (i32)x * (i32)a[0];
        //assert(c < (MAX_FP<<FP_FracBits));
        return c>>FP_FracBits;
    }
    i32 e1dx(FP16 x) const { return ((i32)x*(i32)a[1])>>FP_FracBits; }
    i32 e2dx(FP16 x) const { return ((i32)x*(i32)a[2])>>FP_FracBits; }

    i32 e0dy(FP16 x) const { 
        i32 c = (i32)x * (i32)b[0];
        //assert(c < (MAX_FP<<FP_FracBits) && c > -(MAX_FP<<FP_FracBits));
        return c>>FP_FracBits;
    }
    i32 e1dy(FP16 x) const { return ((i32)x*(i32)b[1])>>FP_FracBits; }
    i32 e2dy(FP16 x) const { return ((i32)x*(i32)b[2])>>FP_FracBits; }

    // because I add this "high" precision feature I have to use i32 as a return value :(
    // but I already have all my edge variables as i32 so not a big deal
    // P.S. I am not using this feature really, so could just to back to FP16 as a ret val
    template <bool FpPrec = false> i32 e0ddx() const { return a[0] << (FpPrec ? FP_FracBits : 0); }
    template <bool FpPrec = false> i32 e1ddx() const { return a[1] << (FpPrec ? FP_FracBits : 0); }
    template <bool FpPrec = false> i32 e2ddx() const { return a[2] << (FpPrec ? FP_FracBits : 0); }

    template <bool FpPrec = false> i32 e0ddy() const { return b[0] << (FpPrec ? FP_FracBits : 0); }
    template <bool FpPrec = false> i32 e1ddy() const { return b[1] << (FpPrec ? FP_FracBits : 0); }
    template <bool FpPrec = false> i32 e2ddy() const { return b[2] << (FpPrec ? FP_FracBits : 0); }

    bool e(const Pt<FP16>& p) const { return e0(p) | e1(p) | e2(p); }
};

// see https://groups.csail.mit.edu/graphics/classes/6.837/F00/Lecture06/lecture6.pdf
// also see https://www.chrishecker.com/Miscellaneous_Technical_Articles (Par1: Foundations)
template<typename T>
struct Grad {

    T a;
    T b;
    T c;

    // no need to inverse multiply by 2*TriangleArea because in perspective-correct interpolation
    // we divide one gradient over another, so these factors cancel
    Grad(const TriSetup<FP16>& s, const T& v0, const T& v1, const T& v2) {
        a = fromFixed(s.e0ddx()) * v0 + fromFixed(s.e1ddx()) * v1 + fromFixed(s.e2ddx()) * v2;
        b = fromFixed(s.e0ddy()) * v0 + fromFixed(s.e1ddy()) * v1 + fromFixed(s.e2ddy()) * v2;
        // need to >> FP_FracBits because c is actually 8.16 after multiplication
        c = fromFixed(s.c[0]>>FP_FracBits) * v0 + fromFixed(s.c[1]>>FP_FracBits) * v1 + fromFixed(s.c[2]>>FP_FracBits) * v2;
    }

    T dx() const { return a; }
    T dy() const { return b; }
    T e(const vec2& pt) const { return a*pt.x + b*pt.y + c; } // evaluate edge equation
};


//typedef int S;
typedef FP16 S;
using TriBuffer = BufferT<TriSetup<S>>;
using TriBufferF = BufferT<TriSetup<float>>;

struct trivec4 {
    vec4 v0, v1, v2;
};

// v0 <- v1 | v1 <- v2 | v2 <- v0
void rotate_left(vec4& v0, vec4& v1, vec4& v2) {
    vec4 t = v0;
    v0 = v1;
    v1 = v2;
    v2 = t;
}

// v2 -> v0 | v0 -> v1 | v1 -> v2
void rotate_right(vec4& v0, vec4& v1, vec4& v2) {
    vec4 t = v0;
    v0 = v2;
    v2 = v1;
    v1 = t;
}

// A - axis, V - value
template<int A, int V>
int clip_plane(u8 flags, vec4 v0, vec4 v1, vec4 v2, trivec4(& i_attr)[ATT_COUNT], trivec4* tris,  trivec4* o_attr) {

    int ntri = 0;
    switch (flags) {
    case 0:
        tris[ntri] = { v0, v1, v2 };
        for(int i=0;i<ATT_COUNT;++i)
            o_attr[ntri*ATT_COUNT + i] = i_attr[i];
        return 1;
    case 7:
        return 0;
    // permute triangle vertices to one of two canonical forms:
    // 1. Just v2 outside of clipping volume (flag == 4)
    // 2. v1 and v2 outside of clippling volume (flag == 6)
    case 1:
        flags = 4;
    case 5:
        rotate_left(v0, v1, v2);
        for(int i=0;i<ATT_COUNT;++i)
            rotate_left(i_attr[i].v0, i_attr[i].v1, i_attr[i].v2);
        break;
    case 2:
        flags = 4;
    case 3:
        rotate_right(v0, v1, v2);
        for(int i=0;i<ATT_COUNT;++i)
            rotate_right(i_attr[i].v0, i_attr[i].v1, i_attr[i].v2);
        break;
    }

    if (flags == 4) { // hither clipping yields 2 triangles
        const float ooHw[3] = { 1.0f/v0.w, 1.0f/v1.w, 1.0f/v2.w };

        const vec4 v0p = ooHw[0] * v0;
        const vec4 v1p = ooHw[1] * v1;
        const vec4 v2p = ooHw[2] * v2;

        const float t12 = (V - v1p[A])/(v2p[A] - v1p[A]);
        const float Hw12 = 1.0f/lerp(ooHw[1], ooHw[2], t12);
        const vec4 v12 = Hw12 * lerp(v1p, v2p, t12);

        //TODO: can actually just add attribs to just added TriSetup triangle above

        vec4 a0v12 = Hw12*lerp(ooHw[1]*i_attr[0].v1, ooHw[2]*i_attr[0].v2, t12);
        vec4 a1v12 = Hw12*lerp(ooHw[1]*i_attr[1].v1, ooHw[2]*i_attr[1].v2, t12);
        vec4 a2v12 = Hw12*lerp(ooHw[1]*i_attr[2].v1, ooHw[2]*i_attr[2].v2, t12);

        tris[ntri] = { v0, v1, v12 };
        o_attr[ntri*ATT_COUNT + 0] = {i_attr[0].v0, i_attr[0].v1, a0v12};
        o_attr[ntri*ATT_COUNT + 1] = {i_attr[1].v0, i_attr[1].v1, a1v12};
        o_attr[ntri*ATT_COUNT + 2] = {i_attr[2].v0, i_attr[2].v1, a2v12};
        ntri++;

        const float t02 = (V - v0p[A])/(v2p[A] - v0p[A]);
        const float Hw02 = 1.0f/lerp(ooHw[0], ooHw[2], t02);
        vec4 v02 = Hw02 * lerp(v0p, v2p, t02);

        vec4 a0v02 = Hw02 * lerp(ooHw[0]*i_attr[0].v0, ooHw[2]*i_attr[0].v2, t02);
        vec4 a1v02 = Hw02 * lerp(ooHw[0]*i_attr[1].v0, ooHw[2]*i_attr[1].v2, t02);
        vec4 a2v02 = Hw02 * lerp(ooHw[0]*i_attr[2].v0, ooHw[2]*i_attr[2].v2, t02);

        tris[ntri] = { v0, v12, v02 };
        o_attr[ntri*ATT_COUNT + 0] = {i_attr[0].v0, a0v12, a0v02};
        o_attr[ntri*ATT_COUNT + 1] = {i_attr[1].v0, a1v12, a1v02};
        o_attr[ntri*ATT_COUNT + 2] = {i_attr[2].v0, a2v12, a2v02};

        return 2;

    } else { // hither clipping yields one triangle
        const float ooHw[3] = { 1.0f/v0.w, 1.0f/v1.w, 1.0f/v2.w };

        const vec4 v0p = ooHw[0] * v0;
        const vec4 v1p = ooHw[1] * v1;
        const vec4 v2p = ooHw[2] * v2;

        const float t01 = (V - v0p[A])/(v1p[A] - v0p[A]);
        const float Hw01 = 1.0f/lerp(ooHw[0], ooHw[1], t01);
        const vec4 v01 = Hw01 * lerp(v0p, v1p, t01);

        const float t02 = (V - v0p[A])/(v2p[A] - v0p[A]);
        const float Hw02 = 1.0f/lerp(ooHw[0], ooHw[2], t02);
        const vec4 v02 = Hw02 * lerp(v0p, v2p, t02);

        tris[ntri] = { v0, v01, v02 };

        vec4 a0v01 = Hw01 * lerp(ooHw[0]*i_attr[0].v0, ooHw[1]*i_attr[0].v1, t01);
        vec4 a1v01 = Hw01 * lerp(ooHw[0]*i_attr[1].v0, ooHw[1]*i_attr[1].v1, t01);
        vec4 a2v01 = Hw01 * lerp(ooHw[0]*i_attr[2].v0, ooHw[1]*i_attr[2].v1, t01);

        vec4 a0v02 = Hw02 * lerp(ooHw[0]*i_attr[0].v0, ooHw[2]*i_attr[0].v2, t02);
        vec4 a1v02 = Hw02 * lerp(ooHw[0]*i_attr[1].v0, ooHw[2]*i_attr[1].v2, t02);
        vec4 a2v02 = Hw02 * lerp(ooHw[0]*i_attr[2].v0, ooHw[2]*i_attr[2].v2, t02);

        o_attr[ntri*ATT_COUNT + 0] = {i_attr[0].v0, a0v01, a0v02};
        o_attr[ntri*ATT_COUNT + 1] = {i_attr[1].v0, a1v01, a1v02};
        o_attr[ntri*ATT_COUNT + 2] = {i_attr[2].v0, a2v01, a2v02};

        return 1;
    }
}

#define MAX_CLIP_TRI 64

// super nice doc, real joy
// On-Line Computer Graphics Notes CLIPPING by Kenneth Joy
// https://fabiensanglard.net/polygon_codec/clippingdocument/Clipping.pdf
// also classic Blinn/Newell paper p245-blinn.pdf CLIPPING USING HOMOGENEOUS COORDINATES
// TODO: rewrite my awful clip_plane according to this paper as well
int clip_w(vec4 v0, vec4 v1, vec4 v2, trivec4(& i_attr)[ATT_COUNT], trivec4* tris,  trivec4* o_attr) {

    const float W = 0.0001f; // to avoid div by 0

    u8 flags = v0.w < W ? 1 : 0;
    flags += v1.w < W ? 2 : 0;
    flags += v2.w < W ? 4 : 0;

    int ntri = 0;
    switch (flags) {
    case 0:
        tris[ntri] = { v0, v1, v2 };
        for(int i=0;i<ATT_COUNT;++i)
            o_attr[ntri*ATT_COUNT + i] = i_attr[i];
        return 1;
    case 7:
        return 0;
    // permute triangle vertices to one of two canonical forms:
    // 1. Just v2 outside of clipping volume (flag == 4)
    // 2. v1 and v2 outside of clippling volume (flag == 6)
    case 1:
        flags = 4;
    case 5:
        rotate_left(v0, v1, v2);
        for(int i=0;i<ATT_COUNT;++i)
            rotate_left(i_attr[i].v0, i_attr[i].v1, i_attr[i].v2);
        break;
    case 2:
        flags = 4;
    case 3:
        rotate_right(v0, v1, v2);
        for(int i=0;i<ATT_COUNT;++i)
            rotate_right(i_attr[i].v0, i_attr[i].v1, i_attr[i].v2);
        break;
    }

    if (flags == 4) { // hither clipping yields 2 triangles

        const float t12 = (W - v1.w)/(v2.w - v1.w);
        assert(!isnan(t12));
        const vec4 v12 = lerp(v1, v2, t12);

        vec4 a0v12 = lerp(i_attr[0].v1, i_attr[0].v2, t12);
        vec4 a1v12 = lerp(i_attr[1].v1, i_attr[1].v2, t12);
        vec4 a2v12 = lerp(i_attr[2].v1, i_attr[2].v2, t12);

        assert(!isnan(v12.x) && !isnan(v12.y) && !isnan(v12.z) && !isnan(v12.w));

        tris[ntri] = { v0, v1, v12 };
        o_attr[ntri*ATT_COUNT + 0] = {i_attr[0].v0, i_attr[0].v1, a0v12};
        o_attr[ntri*ATT_COUNT + 1] = {i_attr[1].v0, i_attr[1].v1, a1v12};
        o_attr[ntri*ATT_COUNT + 2] = {i_attr[2].v0, i_attr[2].v1, a2v12};
        ntri++;

        const float t02 = (W - v0.w)/(v2.w - v0.w);
        assert(!isnan(t02));
        const vec4 v02 = lerp(v0, v2, t02);

        vec4 a0v02 = lerp(i_attr[0].v0, i_attr[0].v2, t02);
        vec4 a1v02 = lerp(i_attr[1].v0, i_attr[1].v2, t02);
        vec4 a2v02 = lerp(i_attr[2].v0, i_attr[2].v2, t02);

        tris[ntri] = { v0, v12, v02 };
        o_attr[ntri*ATT_COUNT + 0] = {i_attr[0].v0, a0v12, a0v02};
        o_attr[ntri*ATT_COUNT + 1] = {i_attr[1].v0, a1v12, a1v02};
        o_attr[ntri*ATT_COUNT + 2] = {i_attr[2].v0, a2v12, a2v02};

        return 2;


    } else { // hither clipping yields one triangle

        const float t01 = (W - v0.w)/(v1.w - v0.w);
        const vec4 v01 = lerp(v0, v1, t01);

        const float t02 = (W - v0.w)/(v2.w - v0.w);
        const vec4 v02 = lerp(v0, v2, t02);
        assert(!isnan(v02.x) && !isnan(v02.y) && !isnan(v02.z) && !isnan(v02.w));

        tris[ntri] = { v0, v01, v02 };

        vec4 a0v01 = lerp(i_attr[0].v0, i_attr[0].v1, t01);
        vec4 a1v01 = lerp(i_attr[1].v0, i_attr[1].v1, t01);
        vec4 a2v01 = lerp(i_attr[2].v0, i_attr[2].v1, t01);

        vec4 a0v02 = lerp(i_attr[0].v0, i_attr[0].v2, t02);
        vec4 a1v02 = lerp(i_attr[1].v0, i_attr[1].v2, t02);
        vec4 a2v02 = lerp(i_attr[2].v0, i_attr[2].v2, t02);

        o_attr[ntri*ATT_COUNT + 0] = {i_attr[0].v0, a0v01, a0v02};
        o_attr[ntri*ATT_COUNT + 1] = {i_attr[1].v0, a1v01, a1v02};
        o_attr[ntri*ATT_COUNT + 2] = {i_attr[2].v0, a2v01, a2v02};

        return 1;
    }
}

// worst case is actually 64! triangles if it is clipped by all planes
//int clip_triangle(vec4 v0, vec4 v1, vec4 v2, trivec4(& i_attr)[ATT_COUNT], trivec4* o_tris,  trivec4 (& o_attr)[2][12]) {
int clip_triangle(const u8 mask, vec4 v0, vec4 v1, vec4 v2, trivec4(& i_attr)[ATT_COUNT], trivec4* o_tris,  trivec4* o_attr, const int o_size) {

    //int ntri = 0;
    u8 flags = 0;

    trivec4 itri[MAX_CLIP_TRI];
    trivec4 iattr[ATT_COUNT*MAX_CLIP_TRI];
    int icount = 0;

    trivec4 otri[MAX_CLIP_TRI];
    trivec4 oattr[ATT_COUNT*MAX_CLIP_TRI];

    trivec4* it = itri, *ia = iattr, *ot = otri, *oa = oattr;

    const float z_near = 0.5f; // TODO:
    if(v0.w < z_near && v1.w < z_near && v2.w < z_near)
        return 0;

    // to make all calls same
    icount = 1;
    it[0] = { v0, v1, v2 };
    for(int i=0;i<ATT_COUNT;++i) {
        ia[i] = i_attr[i];
    }
    int ocount = 0;

    if(mask & 0x40) {
        ocount = 0;
        for(int i=0;i<icount; ++i) {
            v0 = it[i].v0;
            v1 = it[i].v1;
            v2 = it[i].v2;
            trivec4 a[ATT_COUNT] = {ia[ATT_COUNT*i + 0], ia[ATT_COUNT*i + 1], ia[ATT_COUNT*i + 2]  };

            assert(o_size > 2);
            ocount += clip_w(v0, v1, v2, a, ot + ocount, oa + ATT_COUNT*ocount);
        }
        swap(ia, oa);
        swap(it, ot);
        swap(icount, ocount);
    }

    // can use something like index buffer, so that we only do interpolation of attributes once at the end if triangle is clipped by multiple planes,
    // or does it matter in this case?

    // -x
    if(mask&0x1) {
        ocount = 0;
        for(int i=0;i<icount; ++i) {

            v0 = it[i].v0;
            v1 = it[i].v1;
            v2 = it[i].v2;
            trivec4 a[ATT_COUNT] = {ia[ATT_COUNT*i + 0], ia[ATT_COUNT*i + 1], ia[ATT_COUNT*i + 2]  };

            flags = v0[0] < -v0.w ? 1 : 0;
            flags += v1[0] < -v1.w ? 2 : 0;
            flags += v2[0] < -v2.w ? 4 : 0;

            assert(o_size > 2);
            ocount += clip_plane<0, -1>(flags, v0, v1, v2, a, ot + ocount, oa + ATT_COUNT*ocount);
        }

        swap(ia, oa);
        swap(it, ot);
        swap(icount, ocount);
    }

    // +x
    if(mask&0x2) {
        ocount = 0;
        for(int i=0;i<icount; ++i) {
            v0 = it[i].v0;
            v1 = it[i].v1;
            v2 = it[i].v2;
            trivec4 a[ATT_COUNT] = {ia[ATT_COUNT*i + 0], ia[ATT_COUNT*i + 1], ia[ATT_COUNT*i + 2]  };

            flags = v0[0] > v0.w ? 1 : 0;
            flags += v1[0] > v1.w ? 2 : 0;
            flags += v2[0] > v2.w ? 4 : 0;

            assert(o_size > ocount + 2);
            ocount += clip_plane<0, 1>(flags, v0, v1, v2, a, ot + ocount, oa + ATT_COUNT*ocount);
        }

        swap(ia, oa);
        swap(it, ot);
        swap(icount, ocount);
    }

    // -y
    if(mask&0x4) {
        ocount = 0;
        for(int i=0;i<icount;++i) {
            v0 = it[i].v0;
            v1 = it[i].v1;
            v2 = it[i].v2;
            trivec4 a[ATT_COUNT] = {ia[ATT_COUNT*i + 0], ia[ATT_COUNT*i + 1], ia[ATT_COUNT*i + 2]  };

            flags = v0[1] < -v0.w ? 1 : 0;
            flags += v1[1] < -v1.w ? 2 : 0;
            flags += v2[1] < -v2.w ? 4 : 0;

            assert(o_size > ocount + 2);
            ocount += clip_plane<1, -1>(flags, v0, v1, v2, a, ot + ocount, oa + ATT_COUNT*ocount);
        }

        swap(ia, oa);
        swap(it, ot);
        swap(icount, ocount);
    }

    // +y
    if(mask&0x8) {
        ocount = 0;
        for(int i=0;i<icount;++i) {
            v0 = it[i].v0;
            v1 = it[i].v1;
            v2 = it[i].v2;
            trivec4 a[ATT_COUNT] = {ia[ATT_COUNT*i + 0], ia[ATT_COUNT*i + 1], ia[ATT_COUNT*i + 2]  };

            flags = v0[1] > v0.w ? 1 : 0;
            flags += v1[1] > v1.w ? 2 : 0;
            flags += v2[1] > v2.w ? 4 : 0;

            assert(o_size > ocount + 2);
            ocount += clip_plane<1, 1>(flags, v0, v1, v2, a, ot + ocount, oa + ATT_COUNT*ocount);
        }

        swap(ia, oa);
        swap(it, ot);
        swap(icount, ocount);
    }

    // -z
    if(mask&0x10) {
        ocount = 0;
        for(int i=0;i<icount;++i) {
            v0 = it[i].v0;
            v1 = it[i].v1;
            v2 = it[i].v2;
            trivec4 a[ATT_COUNT] = {ia[ATT_COUNT*i + 0], ia[ATT_COUNT*i + 1], ia[ATT_COUNT*i + 2]  };

            flags = v0[2] < -v0.w ? 1 : 0;
            flags += v1[2] < -v1.w ? 2 : 0;
            flags += v2[2] < -v2.w ? 4 : 0;

            assert(o_size > ocount + 2);
            ocount += clip_plane<2, -1>(flags, v0, v1, v2, a, ot + ocount, oa + ATT_COUNT*ocount);
        }

        swap(ia, oa);
        swap(it, ot);
        swap(icount, ocount);
    }

    // -z
    if(mask&0x20) {
        ocount = 0;
        for(int i=0;i<icount;++i) {
            v0 = it[i].v0;
            v1 = it[i].v1;
            v2 = it[i].v2;
            trivec4 a[ATT_COUNT] = {ia[ATT_COUNT*i + 0], ia[ATT_COUNT*i + 1], ia[ATT_COUNT*i + 2]  };

            flags = v0[2] > v0.w ? 1 : 0;
            flags += v1[2] > v1.w ? 2 : 0;
            flags += v2[2] > v2.w ? 4 : 0;

            assert(o_size > ocount + 2);
            ocount += clip_plane<2, 1>(flags, v0, v1, v2, a, ot + ocount, oa + ATT_COUNT*ocount);
        }

        swap(ia, oa);
        swap(it, ot);
        swap(icount, ocount);
    }

    assert(icount <= MAX_CLIP_TRI);
    for(int ti=0;ti<icount;++ti) {
        o_tris[ti] = { it[ti].v0, it[ti].v1, it[ti].v2 };

        for(int att=0;att<ATT_COUNT;++att) {
            o_attr[ATT_COUNT*ti + att] = ia[ATT_COUNT*ti + att];
        }
    }

    return icount;
}


struct Object {
    Str name_;
    ObjModel* model_;
    Image* texture_;
    vec3 pos_;
    vec3 euler_;
    vec3 scale_;
    struct DrawParams {
        vec2 scroll_uv;
        bool b_blend;
    } draw_flags;

};

using ObjList = BufferT<Object, int>;
ObjList g_objects;
int g_active_obj = 0;

// just a hack to fill values from initial imgui variables and not trigger change event by hand every program start
bool g_b_imgui_initialized = false;

enum eTraverseType { kTraverseAABB, kTraverseZigZag };
enum eShadingViewMode: u8 { kColor, kTexture, kNormals, kLighting, kShader };

bool g_b_draw_outline = false;
bool g_b_draw_z_buf = false;
bool g_b_draw_degenerate_tris = true;
bool g_b_persp_corr = true;
bool g_b_persp_corr_2nd_way = false;
bool g_b_blend = true;
bool g_b_clip = false;
bool g_b_no_overdraw = true;
bool g_b_delta_update = true;
bool g_b_animate = false;
bool g_b_draw_grid = true;
bool g_b_draw_wireframe = true;
bool g_b_draw_only_active_tri = false;
bool g_b_force_checker = false;
bool g_b_uv_override = false;

bool g_b_depth_write_enable = false;
bool g_b_depth_test_enable = false;
int g_num_z_fail = 0;

eTraverseType g_traverse_type = kTraverseAABB;
int g_shading_view_mode = eShadingViewMode::kColor;
u8 g_alpha = 0x55;
int g_clip_mask = 0xff; // -x, +x, -y, +y, -z, +z
u32 g_clear_color;
u32 g_tri_color;
u32 g_outline_color;
float g_sel_pix_x = 0, g_sel_pix_y = 0;
int g_active_triangle = 0;
vec4 g_uvw_under_cursor;
struct PixInfo {
    D z;
    float hw;
    vec4 attr[ATT_COUNT];

};
PixInfo g_pix_info_under_cursor;

void clearFB(u32 c, float z) {
    //memset(g_fb, c, sizeof(g_fb[0])*g_tex_w*g_tex_h);
    for(int y = 0; y < g_tex_h; ++y) {
        for(int x = 0; x < g_tex_w; ++x) {
            g_fb[y*g_tex_w + x] = c;
            g_fb_z[y*g_tex_w + x] = z; //TODO: float -> D might require some convertion
        }
    }
}
void setTexture(Image* img) {
    g_current_texture = g_b_force_checker ? 0 : img;
}

// 0xAARRGGBB
struct FColor {
    FColor(float aa, float bb, float gg, float rr):a(aa),b(bb),g(gg),r(rr) {}
    FColor() {}
    float a, b, g, r;
    static FColor fromU32(u32 c) {
        FColor dst;
        float s = 1/255.0f;
        dst.a = (c  >> 24)       * s;
        dst.r = ((c >> 16)&0xFF) * s;
        dst.g = ((c >>  8)&0xFF) * s;
        dst.b = ((c      )&0xFF) * s;
        return dst;
    }

    u32 toU32() const {
        //u32 aa = clamp((int)(a * 255.0f + 0.5f), 0, 255);
        //u32 bb = clamp((int)(b * 255.0f + 0.5f), 0, 255);
        //u32 gg = clamp((int)(g * 255.0f + 0.5f), 0, 255);
        //u32 rr = clamp((int)(r * 255.0f + 0.5f), 0, 255);
        u32 aa = (u32)(clamp(a, 0.0f, 1.0f) * 255.0f);
        u32 bb = (u32)(clamp(b, 0.0f, 1.0f) * 255.0f);
        u32 gg = (u32)(clamp(g, 0.0f, 1.0f) * 255.0f);
        u32 rr = (u32)(clamp(r, 0.0f, 1.0f) * 255.0f);
        return bb | (gg<<8) | (rr<<16) | (aa<<24);
    }

    vec4 toVec4() const { return vec4(r,g,b,a);}
};

FColor blend(FColor s, FColor d, float src_a, float dst_a) {
    FColor o;
    o.a = s.a * src_a + d.a * dst_a;
    o.r = s.r * src_a + d.r * dst_a;
    o.g = s.g * src_a + d.g * dst_a;
    o.b = s.b * src_a + d.b * dst_a;
    return o; 
}

FColor blend_add(FColor s, FColor d) {
    return blend(s, d, s.a, 1);
}

FColor c0 = FColor(1, 0, 0, 1);
FColor c1 = FColor(1, 0, 1, 0);
FColor c2 = FColor(1, 1, 0, 0);

inline FColor interpolate(FColor& c0, FColor& c1, FColor& c2, float u, float v, float w) {
    FColor r;
    r.a = c0.a*u + c1.a*v + c2.a*w;
    r.b = c0.b*u + c1.b*v + c2.b*w;
    r.g = c0.g*u + c1.g*v + c2.g*w;
    r.r = c0.r*u + c1.r*v + c2.r*w;
    return r;
}

inline vec4 interpolate(const vec4& c0, const vec4& c1, const vec4& c2, float u, float v, float w) {
    vec4 r;
    r.x = c0.x*u + c1.x*v + c2.x*w;
    r.y = c0.y*u + c1.y*v + c2.y*w;
    r.z = c0.z*u + c1.z*v + c2.z*w;
    r.w = c0.w*u + c1.w*v + c2.w*w;
    return r;
}

inline float interpolate(const float a, const float b, const float c, float u, float v, float w) {
    return a*u + b*v + c*w;
}

void beginFrame() {
    g_num_z_fail = 0;
}

void endFrame() {
}

FColor sampleTexture(vec2 uv, Image* texture, FColor in_color) {
    FColor c;
    if (texture) {
        int tu = (int)(texture->width*uv.x + 0.0f) % texture->width;
        int tv = (int)(texture->height*(1-uv.y) + 0.0f) % texture->height;
        assert(tu>=0 && tv>=0);
        Image::Pixel pix = texture->data[tv*texture->width + tu];
        c = FColor::fromU32(pix.asU32());
    } else { // checker texture
        uv = uv*32;
        c = ((((int)uv.x)>>2)&1) ^ ((((int)uv.y)>>2)&1) ? in_color : FColor(0.5f,0.1f,0.1f,0.1f);
    }

    return c;
}

// 0xAARRGGBB
void renderPixel(int x, int y, float u, float v, float w, const TriSetup<FP16>& ts, u32 incolor, const vec4* att_override, const float hw, float oohw) {
    assert(y>=0 && y < g_tex_h);
    assert(x>=0 && x < g_tex_w);

    u32 c = g_fb[y*g_tex_w + x];

    FColor in_color = FColor::fromU32(incolor);
    FColor src = in_color;
    float temp_a  = src.a;
    float z;

    vec4 color;
    vec4 normal;
    vec4 texcoord;

    //const float oohw = interpolate(ts.ooHw[0], ts.ooHw[1], ts.ooHw[2], u, v, w);
    if(g_b_persp_corr && !g_b_persp_corr_2nd_way) {
        color = hw * interpolate(ts.oo_attribs[ATT_COLOR][0], ts.oo_attribs[ATT_COLOR][1], ts.oo_attribs[ATT_COLOR][2], u, v, w);
        if(!g_b_uv_override) {
            texcoord = hw * interpolate(ts.oo_attribs[ATT_TEXCOORD][0], ts.oo_attribs[ATT_TEXCOORD][1], ts.oo_attribs[ATT_TEXCOORD][2], u, v, w);
        } else {
            texcoord = att_override[ATT_TEXCOORD];
        }
        normal = hw * interpolate(ts.oo_attribs[ATT_NORMAL][0], ts.oo_attribs[ATT_NORMAL][1], ts.oo_attribs[ATT_NORMAL][2], u, v, w);
        z = interpolate(ts.hz_pc[0], ts.hz_pc[1], ts.hz_pc[2], u, v, w); // no need to divide by "oohw" as clip space is what we need ;
    } else {
        color = interpolate(ts.attribs[ATT_COLOR][0], ts.attribs[ATT_COLOR][1], ts.attribs[ATT_COLOR][2], u, v, w);
        texcoord = interpolate(ts.attribs[ATT_TEXCOORD][0], ts.attribs[ATT_TEXCOORD][1], ts.attribs[ATT_TEXCOORD][2], u, v, w);
        normal = interpolate(ts.attribs[ATT_NORMAL][0], ts.attribs[ATT_NORMAL][1], ts.attribs[ATT_NORMAL][2], u, v, w);
        z = interpolate(ts.hz[0], ts.hz[1], ts.hz[2], u, v, w) * oohw;
    }

    // alternative way to calc clip space z
    //float far = 0.5f, near = 10;
    //float zview = -hw;
    //float oo_zview = -oohw;
    //z = ((zview)*-(far + near)/(far - near) -2*far*near/(far - near))*(oo_zview);

    if((int)g_sel_pix_x == x && (int)g_sel_pix_y == y) {
        g_pix_info_under_cursor.attr[ATT_COLOR] = color;
        g_pix_info_under_cursor.attr[ATT_TEXCOORD] = texcoord;
        g_pix_info_under_cursor.attr[ATT_NORMAL] = normal;
        g_pix_info_under_cursor.z = z;
        g_pix_info_under_cursor.hw = hw;
    }

    if(g_b_depth_test_enable) {
        if(g_fb_z[y*g_tex_w + x] <= z) {
            g_num_z_fail++;
            return;
        }
    }

    const vec3 light_dir = vec3(0,-1, 0);
    if(g_shading_view_mode == eShadingViewMode::kColor) {
        src = FColor(color.w, color.z, color.y, color.x);
    } else if(g_shading_view_mode == eShadingViewMode::kTexture) {
        src = sampleTexture(texcoord.xy(), g_current_texture, in_color);
    } else if(g_shading_view_mode == eShadingViewMode::kNormals) {
        vec3 n = 0.5f*normal.xyz() + vec3(0.5f);
        src = FColor(n.x, n.y, n.z, color.w);
    } else if(g_shading_view_mode == eShadingViewMode::kLighting) {
        const float ndotl = max(0.1f, dot(-1.0f*light_dir, normal.xyz()));
        src = FColor(ndotl, ndotl, ndotl, color.w);
    } else { // if(g_shading_view_mode == eShadingViewMode::kShader) {
        vec4 tex = sampleTexture(texcoord.xy(), g_current_texture, in_color).toVec4();
        const float ndotl = max(0.1f, dot(-1.0f*light_dir, normal.xyz()));
        vec4 c = tex * ndotl;
        src = FColor(c.w, c.z, c.y, c.x);
    }

    src.a = g_b_blend ? temp_a : 1;

    if(g_b_blend) {
        FColor dst = FColor::fromU32(c);
        FColor o = blend_add(src, dst);
        g_fb[y*g_tex_w + x] = o.toU32();
    } else {
        g_fb[y*g_tex_w + x] = src.toU32();
    }

    if(g_b_depth_write_enable) {
        g_fb_z[y*g_tex_w + x] = z;
    }
}

void plotLine(int x0, int y0, int x1, int y1, u32 color) {
    int dx = abs(x1 - x0);
    int sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0);
    int sy = y0 < y1 ? 1 : -1;
    int error = dx + dy;
    
    TriSetup<FP16> dummy = TriSetup<FP16>(vec4(0), vec4(0), vec4(0));
    vec4 dummy_att[ATT_COUNT] = {vec4(0.0f)};
    while(true) {
        renderPixel(x0, y0, 0, 0, 0, dummy, color, dummy_att, 1, 1);
        int e2 = 2 * error;
        if (e2 >= dy) {
            if (x0 == x1) break;
            error = error + dy;
            x0 = x0 + sx;
        }
        if (e2 <= dx) {
            if (y0 == y1) break;
            error = error + dx;
            y0 = y0 + sy;
        }
    }
}

void test_fixed_point() {
    // add
    {
    float fa = 12.5f;
    float fb = 38.15f;
    float fc = fa + fb;
    FP16 a = toFP<8>(fa);
    FP16 b = toFP<8>(fb);
    assert((u32)a + (u32)b < MAX_FP);
    FP16 c = a + b;
    printf("ADD: %f + %f = %f | %d + %d = %f => %f\n", fa, fb, fc, a, b, fromFP<8>(c), fc - fromFP<8>(c));
    }

    // mul
    {
        for(int i=0;i<=10;++i) {
            float fa = 10.0f + i/10.0f;
            float fb = 2.55f;
            float fc = fa * fb;
            u32 a = toFP<4>(fa);
            u32 b = toFP<4>(fb);
            u32 c = (a * b);
            assert(c < MAX_FP);
            c = c >> 4;
            printf("MUL POS: %f * %f = %f | %d * %d = %f => %f\n", fa, fb, fc, a, b, fromFP<4>(c), fc - fromFP<4>(c));
        }
    }

    // mul neg
    {
        for(int i=0;i<=10;++i) {
            for(int j=0;j<2;++j) {
                float fa = -10.0f + i/10.0f;
                float fb = j==0 ? 2.55f : -2.55;
                float fc = fa * fb;
                i32 a = toFP<4>(fa);
                i32 b = toFP<4>(fb);
                i32 c = (a * b);
                assert(c < MAX_FP);
                c = c >> 4;
                printf("MUL NEG: %f * %f = %f | %d * %d = %d(%f) => %f\n", fa, fb, fc, a, b, c, fromFP<4>(c), fc - fromFP<4>(c));
            }
        }
    }
}

struct DrawCallInfo {
    Image* texture;
    int s_tri, ntri;
    bool b_blend;
};

using VertexBuffer2D = BufferT<Pt<float>>;
using VertexBuffer = BufferT<vec4>;
using IndexBuffer = BufferT<u8>;
using DrawCallList = BufferT<DrawCallInfo>;


struct Mesh {
    enum ePrimType { kTri, kTriStrip };
    BufferT<u8> ib;
    VertexBuffer vb;
    ePrimType primType;
};

Mesh g_meshes;


bool* map0 = nullptr;
bool* map1 = nullptr;

float g_freq = 0.25;
int g_scale_idx = 1;
Pt<float> g_v0_offset;

TriBuffer g_tris;
TriBufferF g_tris_notclipped;
VertexBuffer2D g_vertices;
DrawCallList g_draw_calls;

struct WorkingPoints {
    enum eTriType { kTris, kTriStrip };
    Pt<float> pts[3];
    i32 idx = 0;
    i32 count = 0;
    //TriBuffer* pvTris = nullptr;
    VertexBuffer2D* pvVB = nullptr;
    //WorkingPoints(TriBuffer* tri_buffer):pvTris(tri_buffer){}
    WorkingPoints(VertexBuffer2D* vb):pvVB(vb){}

    void add_point(float px, float py, eTriType type) {
        assert(count < 3);
        pts[idx] = Pt<float>(px, py);
        idx = (idx+1) % 3;
        printf("add: %.2f %.2f idx:%d\n", px, py, idx);
        if(++count == 3 && kTris == type) {
            printf("add tri\n");
            //TriSetup<S> ts = TriSetup<S>(pts[0], pts[1], pts[2]);
            //pvTris->push(ts);
            pvVB->push(Pt<float>(pts[0].x, pts[0].y));
            pvVB->push(Pt<float>(pts[1].x, pts[1].y));
            pvVB->push(Pt<float>(pts[2].x, pts[2].y));
            count = 0;
            idx = 0;
            // only works for small triangles, otherwise we overflow
            //i32 A0 = ts.e0(ts.v[0]);
            //i32 A1 = ts.e1(ts.v[1]);
            //i32 A2 = ts.e2(ts.v[2]);
            //assert(A0 == A1 && A1 == A2);
            //printf("Area %.3f %.3f %.3f\n", A, fromFP<FP_FracBits>(A0)/2, fromFP<FP_FracBits>(A1)/2, fromFP<FP_FracBits>(A2)/2);
            // so use float calculations
            printf("Area %.3f\n", edge(pts[0], pts[1], pts[2]));
        }
    }

    Pt<float> get(int i) const {
        i = i % 3;
        return pts[(idx + 3 - count + i) % 3];
    }
};
WorkingPoints g_tri_adder(&g_vertices);

struct Projection {
    float left, right;
    float top, bottom;
    float near, far;
};
struct Frustum {
    float FOVY;
    float aspect_w_by_h;
    float near, far;
};
const Projection g_proj_params_def = {0, 1, 1, 0, 0.5f, 10};
Projection g_proj_params = g_proj_params_def;
const Frustum g_frustum_def = { 90.0f, (float)g_tex_w/g_tex_h, g_proj_params_def.near, g_proj_params_def.far};
Frustum g_frustum = g_frustum_def;
m44 g_mproj = m44::identity();
m44 g_mview = m44::identity();


float g_s = 1.0f;
vec4 g_p = vec4(0,0,0,1);//vec4(g_tex_w/2, g_tex_h/4, -4*g_proj_params_def.near, 1);
                           //
ObjModel* g_obj = nullptr;

bool exit_request() { return do_exit; }


void init_scene(TriBuffer& tb, TriBufferF& tbnc, VertexBuffer2D& dyn_vb, m44 mproj, const m44 mview) {

    const m44 viewproj = mul(mproj, mview);

    static const vec4 quad_uv[4] = {
        vec4(0, 0, 0, 0),
        vec4(0, 1, 0, 0),
        vec4(1, 1, 0, 0),
        vec4(1, 0, 0, 0),
    };


    float view_z = -1;
    vec4 vb[] = {
        vec4(20,  170, view_z), // 0
        vec4(160, 110, view_z), // 1
        vec4(100, 50, view_z),// 2
        vec4(30,  80, view_z),// 3

        vec4(170, 160), // 4
        vec4(250, 160), // 5
        vec4(250, 80),// 6
        vec4(170, 80),// 7
                          
        vec4(40, 45), // 8
        vec4(60, 45), // 9
        vec4(60, 25), // 10
        vec4(40, 25), // 11

        vec4(50, 200-165), // 12
    };

    vec4 attribs[] = {
        vec4(1, 0, 0, 1),
        vec4(0, 1, 0, 1),
        vec4(0, 0, 1, 1),
        vec4(1, 0, 1, 1),

        vec4(1, 0, 0, 1),
        vec4(0, 1, 0, 1),
        vec4(0, 0, 1, 1),
        vec4(1, 0, 1, 1),

        vec4(1, 0, 0, 1),
        vec4(0, 1, 0, 1),
        vec4(0, 0, 1, 1),
        vec4(1, 0, 1, 1),

        vec4(1, 1, 1, 1), // 12
    };

    Pt<float> uv_scale(1,1);
    vec4 attribs_uv[] = {
        vec4(vb[0].x*uv_scale.x, vb[0].y*uv_scale.y, 0, 0),
        vec4(vb[1].x*uv_scale.x, vb[1].y*uv_scale.y, 0, 0),
        vec4(vb[2].x*uv_scale.x, vb[2].y*uv_scale.y, 0, 0),
        vec4(vb[3].x*uv_scale.x, vb[3].y*uv_scale.y, 0, 0),

        vec4(vb[4].x*uv_scale.x, vb[4].y*uv_scale.y, 0, 0),
        vec4(vb[5].x*uv_scale.x, vb[5].y*uv_scale.y, 0, 0),
        vec4(vb[6].x*uv_scale.x, vb[6].y*uv_scale.y, 0, 0),
        vec4(vb[7].x*uv_scale.x, vb[7].y*uv_scale.y, 0, 0),

        vec4(0, 0, 0, 0),
        vec4(1, 0, 0, 0),
        vec4(1, 1, 0, 0),
        vec4(0, 1, 0, 0),
        vec4(0.5f, 0.5f, 0, 0),
        //vec4(vb[8].x*uv_scale.x, vb[8].y*uv_scale.y, 0, 0),
        //vec4(vb[9].x*uv_scale.x, vb[9].y*uv_scale.y, 0, 0),
        //vec4(vb[10].x*uv_scale.x, vb[10].y*uv_scale.y, 0, 0),
        //vec4(vb[11].x*uv_scale.x, vb[11].y*uv_scale.y, 0, 0),

        //vec4(vb[12].x*uv_scale.x, vb[12].y*uv_scale.y, 0, 0),
    }; (void)attribs_uv;

    // why using attribs_uv as is is not producing a regular pattern?

    for(int i=0;i<(int)arrsize(vb); ++i) {
        const float sx = 0.008f;
        const float sy = sx;
        vec4 off(-1.1f,-.5f,0,0);
        vec4 in( sx*vb[i].x, sy*vb[i].y, vb[i].z == 0 ? view_z : vb[i].z, 1);
        vb[i] = mul(viewproj, in + off);
    }

    struct fake_uv_maker {
        void operator()(TriSetup<S>& ts) {
            ts.set_attribs(1, 
                    vec4(fromFixed(ts.v[0].x)/g_tex_w, fromFixed(ts.v[0].y)/g_tex_h), 
                    vec4(fromFixed(ts.v[1].x)/g_tex_w, fromFixed(ts.v[1].y)/g_tex_h), 
                    vec4(fromFixed(ts.v[2].x)/g_tex_w, fromFixed(ts.v[2].y)/g_tex_h));
        }
    } make_fake_uv;

    tb.reset();
    tbnc.reset();
    g_draw_calls.reset();

#if 1
    tb.push(TriSetup<S>(vb[0], vb[1], vb[2]));
    tb.last().set_attribs(0, attribs[0], attribs[1],attribs[2]);  
    tb.last().set_attribs(1, quad_uv[0], quad_uv[1], quad_uv[2]);
    //make_fake_uv(tb.last());
    tb.push(TriSetup<S>(vb[0], vb[2], vb[3]));
    tb.last().set_attribs(0, attribs[0], attribs[2],attribs[3]);  
    tb.last().set_attribs(1, quad_uv[0], quad_uv[2],quad_uv[3]);  

    tb.push(TriSetup<S>(vb[4], vb[5], vb[6]));
    tb.last().set_attribs(0, attribs[4], attribs[5],attribs[6]);  
    tb.last().set_attribs(1, quad_uv[4%4], quad_uv[5%4],quad_uv[6%4]);  
    tb.push(TriSetup<S>(vb[4], vb[6], vb[7]));
    tb.last().set_attribs(0, attribs[4], attribs[6],attribs[7]);  
    tb.last().set_attribs(1, quad_uv[4%4], quad_uv[6%4],quad_uv[7%4]);  

    tb.push(TriSetup<S>(vb[12], vb[8], vb[9]));
    tb.last().set_attribs(0, attribs[12], attribs[8],attribs[9]);  
    tb.last().set_attribs(1, attribs_uv[12], attribs_uv[8],attribs_uv[9]);  
    tb.push(TriSetup<S>(vb[12], vb[9], vb[10]));
    tb.last().set_attribs(0, attribs[12], attribs[9],attribs[10]);  
    tb.last().set_attribs(1, attribs_uv[12], attribs_uv[9],attribs_uv[10]);  
    tb.push(TriSetup<S>(vb[12], vb[10], vb[11]));
    tb.last().set_attribs(0, attribs[12], attribs[10],attribs[11]);  
    tb.last().set_attribs(1, attribs_uv[12], attribs_uv[10],attribs_uv[11]);  
    tb.push(TriSetup<S>(vb[12], vb[11], vb[8]));
    tb.last().set_attribs(0, attribs[12], attribs[11],attribs[8]);  
    tb.last().set_attribs(1, attribs_uv[12], attribs_uv[11],attribs_uv[8]);  

    g_draw_calls.push(DrawCallInfo{nullptr, 0, (int)tb.size(), false});

    // add handdrawn triangles
    const int handdrawn_start = (int)tb.size();
    int counter = 0;
    vec4 tri[3];
    const vec4 fake_p = mul(mproj, vec4(0,0, -1, 1)); (void)fake_p;
    for(int i=0;i<(int)dyn_vb.size(); ++i) {
        vec4 in(dyn_vb[i].x, dyn_vb[i].y, -1, 1);
        //tri[counter++] = mul(mproj, in);

        // back transform from NDC to clip space
        Pt<float> pproj = Win2NDC(dyn_vb[i].x, dyn_vb[i].y, g_tex_w, g_tex_h);

#if 1
        //float zp = 0;
        float zv = -1;//(zp - mproj.m[4*2 + 3]) / mproj.m[4*2 + 2];
        Pt<float> pv = view_xy(pproj.x, pproj.y, zv, mproj);
        vec4 pp = mul(mproj, vec4(pv.x, pv.y, zv, 1));
#else // remove div and mul by inverse
        Pt<float> pv = Pt<float>(xp - mproj.m[4*0 + 2], yp - mproj.m[4*1 + 2]);
        vec4 pp;
        pp.x = pv.x + zv * mproj.m[4*0 + 3];
        pp.y = pv.y + zv * mproj.m[4*1 + 3];
        pp.z = mproj.m[4*2 + 2] * zv + mproj.m[4*2 + 3]*1;
        pp.w = -zv;
#endif

        Pt<float> ptWin2 = NDC2Win(pp.x, pp.y, g_tex_w, g_tex_h);(void)(ptWin2);
        assert(fabsf(dyn_vb[i].x - ptWin2.x) < 1e-4);
        assert(fabsf(dyn_vb[i].y - ptWin2.y) < 1e-4);
        //printf("%d %f %f -> %f %f\n", i, dyn_vb[i].x, dyn_vb[i].y, ptNDC.x, ptNDC.y);

        tri[counter++] = pp;

        if(counter == 3) {
            tb.push(TriSetup<S>(tri[0], tri[1], tri[2]));
            tb.last().set_attribs(0, attribs[0], attribs[1],attribs[2]);  
            make_fake_uv(tb.last());
            counter = 0;
        }
    }
    g_draw_calls.push(DrawCallInfo{nullptr, handdrawn_start, (int)tb.size() - handdrawn_start, false});
#endif

    const int meshes_start_tri = (int)tb.size();
    g_meshes.ib.reset();
    g_meshes.vb.reset();

    vec4 mesh_attr[3] = {
        vec4(1, 0, 0, 1),
        vec4(0, 1, 0, 1),
        vec4(0, 0, 1, 1),
    };
#if 0
    vec4 pyramid_pos = g_p;
    float pyramid_scale = 1;
    // pyramid
    g_meshes.vb.push(pyramid_scale*vec4(0,    1,  0,        0) + pyramid_pos);
    g_meshes.vb.push(pyramid_scale*vec4(-0.7f,0, -0.7f, 0) + pyramid_pos);
    g_meshes.vb.push(pyramid_scale*vec4(0.7f, 0, -0.7f, 0) + pyramid_pos);
    g_meshes.vb.push(pyramid_scale*vec4(0,    0,  1,        0) + pyramid_pos);

    g_meshes.ib.push(0);
    g_meshes.ib.push(1);
    g_meshes.ib.push(2);

    g_meshes.ib.push(0);
    g_meshes.ib.push(2);
    g_meshes.ib.push(3);

    g_meshes.ib.push(0);
    g_meshes.ib.push(3);
    g_meshes.ib.push(1);

    g_meshes.ib.push(3);
    g_meshes.ib.push(2);
    g_meshes.ib.push(1);
#endif

    // tilted quad
    float quad_scale = 1; (void)quad_scale;
    vec4 quad_pos = vec4(-0.5f*0, 0,-0,0) + g_p;
    const int quad_vb_offset = (int)g_meshes.vb.size();
#if 0
    g_meshes.vb.push(quad_scale*vec4(-1*g_s, -0.95f, 1*g_s, 1) + quad_pos);
    g_meshes.vb.push(quad_scale*vec4(-1*g_s, -0.95f, -1*g_s, 1) + quad_pos);
    g_meshes.vb.push(quad_scale*vec4( 1*g_s, -0.95f, -1*g_s, 1) + quad_pos);
    g_meshes.vb.push(quad_scale*vec4( 1*g_s, -0.95f, 1*g_s, 1) + quad_pos);
#endif

#if 1
    g_meshes.vb.push(quad_scale*vec4(-1*g_s, -0.95f, -2*g_s, 0) + quad_pos);
    g_meshes.vb.push(quad_scale*vec4(-1*g_s, 0.95f, -1.5*g_s, 0) + quad_pos);
    g_meshes.vb.push(quad_scale*vec4( 1*g_s, 0.95f, -1.5*g_s, 0) + quad_pos);
    g_meshes.vb.push(quad_scale*vec4( 1*g_s, -0.95f, -1*g_s, 0) + quad_pos);
#endif
    const int quad_vb_count = (int)g_meshes.vb.size() - quad_vb_offset;

    // rotate quad 
    static float rot_angle_rad = 6.972558f;
    static float u_offset = 0;
    static float v_offset = 0;
    u_offset = u_offset + 0.01f;
    v_offset = v_offset + 0.01f;

    if(g_b_animate) {
        rot_angle_rad += .5f*M_PIf/180.0f;
    } else {
        rot_angle_rad = 0;//6.981285f;
    }

    //printf("anim angle = %f\n", rot_angle_rad);

    m44 rotM = rotateXZ(rot_angle_rad);
    for(int i=0; i < quad_vb_count; ++i) {
        vec4 v = g_meshes.vb[i + quad_vb_offset] - quad_pos;
        v = mul(rotM, v);
        v = v + quad_pos;
        g_meshes.vb[i + quad_vb_offset] = v;
    }
#if 0
    g_meshes.ib.push(quad_vb_offset + 0);
    g_meshes.ib.push(quad_vb_offset + 1);
    g_meshes.ib.push(quad_vb_offset + 2);
#endif

#if 1
    g_meshes.ib.push(quad_vb_offset + 0);
    g_meshes.ib.push(quad_vb_offset + 2);
    g_meshes.ib.push(quad_vb_offset + 3);
#endif


    m44 my_proj = mproj;//make_proj(-g_proj_params.right, g_proj_params.right, 
            //-g_proj_params.top, g_proj_params.top, 
            //g_proj_params.near, g_proj_params.far);
    (void)my_proj;
    // 2 attributes to interpolate, each plane can create 2 triangles out of 1, so max produced tris is 2^6
    trivec4 o_tri[MAX_CLIP_TRI];
    trivec4 o_attr[MAX_CLIP_TRI*ATT_COUNT]; 

    // add triangles from mesh
    {
        const IndexBuffer& ib = g_meshes.ib;
        const VertexBuffer& vb = g_meshes.vb;
        int inc = g_meshes.primType == Mesh::kTri ? 3 : 1;
        for(int i=0;i<(int)ib.size(); i+=inc) {
            const i32 i0 = ib[i+0];
            const i32 i1 = ib[i+1];
            const i32 i2 = ib[i+2];

            const vec4& v0 = vb[i0];
            const vec4& v1 = vb[i1];
            const vec4& v2 = vb[i2];
            
            vec4 v0p = mul(viewproj, v0);
            vec4 v1p = mul(viewproj, v1);
            vec4 v2p = mul(viewproj, v2);

            trivec4 attribs[ATT_COUNT] = {
                { mesh_attr[i0%3], mesh_attr[i1%3], mesh_attr[i2%3]},
                { quad_uv[i0%4], quad_uv[i1%4], quad_uv[i2%4] },
                { vec4(1), vec4(1), vec4(1) }
            };

            if(g_b_clip) {
                const int ntris = clip_triangle((u8)g_clip_mask, v0p, v1p, v2p, attribs, o_tri, o_attr, MAX_CLIP_TRI);
                assert(ntris < MAX_CLIP_TRI);
                for(int i=0;i<ntris; ++i) {
                    tb.push(TriSetup<S>(o_tri[i].v0, o_tri[i].v1, o_tri[i].v2));
                    for(int ai = 0; ai < ATT_COUNT; ai++) {
                        const int att_idx = ai + ATT_COUNT*i;
                        tb.last().set_attribs(ai, o_attr[att_idx].v0, o_attr[att_idx].v1, o_attr[att_idx].v2);  
                    }
                }
            } else {
                tb.push(TriSetup<S>(v0p, v1p, v2p));
                tb.last().set_attribs(0, mesh_attr[i0%3], mesh_attr[i1%3], mesh_attr[i2%3]);  
                tb.last().set_attribs(1, quad_uv[i0%4], quad_uv[i1%4], quad_uv[i2%4]);  
            }

            tbnc.push(TriSetup<float>(v0p, v1p, v2p));
        }
    }

    g_draw_calls.push(DrawCallInfo{nullptr, meshes_start_tri, (int)tb.size() - meshes_start_tri, false});

    for(int oi=0; oi<g_objects.size(); oi++)
    {
        const int obj_model_start_tri = (int)tb.size();

        //vec4 s = vec4(1, 1, 1, 1);
        //vec4 pos = vec4(5.7, -2.8, -0, 0);

        const Object& o = g_objects[oi];
        const ObjModel* mdl = o.model_;
        const BufferT<ObjModel::V>& ib = mdl->ib;
        int inc = g_meshes.primType == Mesh::kTri ? 3 : 1;

        vec3 vmin = vec3(FLT_MAX, FLT_MAX, FLT_MAX);
        vec3 vmax = vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for(int i=0;i<(int)ib.size(); i++) {
            const ObjModel::V i0 = ib[i+0];
            vec3 v = mdl->v[i0.v];
            vmin.x = min(v.x, vmin.x);
            vmin.y = min(v.y, vmin.y);
            vmin.z = min(v.z, vmin.z);

            vmax.x = max(v.x, vmax.x);
            vmax.y = max(v.y, vmax.y);
            vmax.z = max(v.z, vmax.z);
        }

        //vec3 center = 0.5f * (vmax + vmin);
        //pos.x = -center.x;
        //pos.y = -center.y;
        //pos.z = -1;

        m44 rotM = rotateXYZ(o.euler_.x, o.euler_.y, o.euler_.z);
        //m44 rotM = rotateXZ(rot_angle_rad);
        bool b_has_normals = mdl->n.size() > 0;

        for(int i=0;i<(int)ib.size(); i+=inc) {
            const ObjModel::V i0 = ib[i+0];
            const ObjModel::V i1 = ib[i+1];
            const ObjModel::V i2 = ib[i+2];

            vec4 v0 = vec4(mdl->v[i0.v], 1);
            vec4 v1 = vec4(mdl->v[i1.v], 1);
            vec4 v2 = vec4(mdl->v[i2.v], 1);

            vec4 t0 = mdl->t[i0.t];
            vec4 t1 = mdl->t[i1.t];
            vec4 t2 = mdl->t[i2.t];

            if(o.draw_flags.scroll_uv.x) {
                t0.x = t0.x + u_offset*o.draw_flags.scroll_uv.x;
                t1.x = t1.x + u_offset*o.draw_flags.scroll_uv.x;
                t2.x = t2.x + u_offset*o.draw_flags.scroll_uv.x;
            }
            if(o.draw_flags.scroll_uv.y) {
                t0.y = t0.y + v_offset*o.draw_flags.scroll_uv.y;
                t1.y = t1.y + v_offset*o.draw_flags.scroll_uv.y;
                t2.y = t2.y + v_offset*o.draw_flags.scroll_uv.y;
            }

            vec4 n0 = b_has_normals ? vec4(mdl->n[i0.n], 0) : normalize(cross(normalize((v1 - v0).xyz()), normalize((v2 - v0).xyz())));
            vec4 n1 = b_has_normals ? vec4(mdl->n[i1.n], 0) : normalize(cross(normalize((v1 - v0).xyz()), normalize((v2 - v0).xyz())));
            vec4 n2 = b_has_normals ? vec4(mdl->n[i2.n], 0) : normalize(cross(normalize((v1 - v0).xyz()), normalize((v2 - v0).xyz())));

            v0 = v0 * vec4(o.scale_, 1);
            v0 = mul(rotM, v0);
            v0 = v0 + o.pos_;
            v1 = v1 * vec4(o.scale_, 1);
            v1 = mul(rotM, v1);
            v1 = v1 + o.pos_;
            v2 = v2 * vec4(o.scale_, 1);
            v2 = mul(rotM, v2);
            v2 = v2 + o.pos_;

            //vec4 v0v = mul(mview, v0);
            //vec4 v0p = mul(mproj, v0v);
            vec4 v0p = mul(viewproj, v0);
            vec4 v1p = mul(viewproj, v1);
            vec4 v2p = mul(viewproj, v2);

            n0 = mul(mview, mul(rotM, n0));
            n1 = mul(mview, mul(rotM, n1));
            n2 = mul(mview, mul(rotM, n2));

            trivec4 attribs[ATT_COUNT] = {
                { n0, n1, n2 },
                { t0, t1, t2 },
                { n0, n1, n2 },
            };

            if(g_b_clip) {
                const int ntris = clip_triangle((u8)g_clip_mask, v0p, v1p, v2p, attribs, o_tri, o_attr, MAX_CLIP_TRI);
                assert(ntris < MAX_CLIP_TRI);
                for(int i=0;i<ntris; ++i) {
                    tb.push(TriSetup<S>(o_tri[i].v0, o_tri[i].v1, o_tri[i].v2));
                    for(int ai = 0; ai < ATT_COUNT; ai++) {
                        const int att_idx = ai + ATT_COUNT*i;
                        tb.last().set_attribs(ai, o_attr[att_idx].v0, o_attr[att_idx].v1, o_attr[att_idx].v2);  
                    }
                }
            } else {
                tb.push(TriSetup<S>(v0p, v1p, v2p));
                //tb.last().set_attribs(0, mesh_attr[i0.v%3], mesh_attr[i1.v%3], mesh_attr[i2.v%3]);  
                tb.last().set_attribs(0, n0, n1, n2);  
                tb.last().set_attribs(1, t0, t1, t2);  
                tb.last().set_attribs(2, n0, n1, n2);  
            }

            tbnc.push(TriSetup<float>(v0p, v1p, v2p));
        }

        g_draw_calls.push(DrawCallInfo{o.texture_, obj_model_start_tri, (int)tb.size() - obj_model_start_tri, false});
    }
}

Image* make_some_noise(int w, int h) {
    Image* img = (Image*)malloc(sizeof(Image));
    img->bpp = 32;
    img->width = w;
    img->height =h;
    img->data = (Image::Pixel*)malloc(w*h*sizeof(Image::Pixel));
    img->filename_.len = 0;
    img->filename_.data = nullptr;

    for(int y = 0; y < h; y++) {
        for(int x = 0; x < w; x++) {
            float fx = x / (float)w;
            float fy = y / (float)h;
            u32 v = (u32)(clamp(255.0f*fbm(3.0f*vec2(fx, fy)), 0.0f, 255.0f));
            //b g r a
            //img->data[y*w + x] = v | (v<<8) | (v<<16) | 0x000000ff;
            img->data[y*w + x].r = v;
            img->data[y*w + x].g = v;
            img->data[y*w + x].b = v;
            img->data[y*w + x].a = 255;
        }
    }
    return img;
}

struct ModelRes {
    const char* name;
    const char* path;
    ObjModel* model;
};
BufferT<ModelRes, int> g_models;

struct TextureRes {
    const char* name;
    const char* path;
    Image* tex;
};
BufferT<TextureRes, int> g_textures;

ObjModel* get_model(const char* name, int len = -1) {
    for(int i=0;i<g_models.size();++i) {
        const ModelRes& mres = g_models[i];
        if(strncmp(mres.name, name, (u32)len) == 0) {
            return mres.model;
        }
    }
    return nullptr;
}

const char* find_name(ObjModel* mdl) {
    for(int i=0;i<g_models.size();++i) {
        const ModelRes& mres = g_models[i];
        if(mres.model == mdl) {
            return mres.name;
        }
    }
    return nullptr;
}

const ModelRes* find_res(ObjModel* mdl) {
    for(int i=0;i<g_models.size();++i) {
        const ModelRes& mres = g_models[i];
        if(mres.model == mdl) {
            return &mres;
        }
    }
    return nullptr;
}

Image* get_texture(const char* name, int len = -1) {
    for(int i=0;i<g_textures.size();++i) {
        const TextureRes& tres = g_textures[i];
        if(strncmp(tres.name, name, (u32)len) == 0) {
            return tres.tex;
        }
    }
    return nullptr;
}

const char* find_name(Image* img) {
    for(int i=0;i<g_textures.size();++i) {
        const TextureRes& tres = g_textures[i];
        if(tres.tex == img) {
            return tres.name;
        }
    }
    return nullptr;
}

const TextureRes* find_res(Image* img) {
    for(int i=0;i<g_textures.size();++i) {
        const TextureRes& tres = g_textures[i];
        if(tres.tex == img) {
            return &tres;
        }
    }
    return nullptr;
}


ObjModel* add_resource(BufferT<ModelRes, int>& db, const char* name, const char* path) {
    ObjModel* model = get_model(name);
    if(!model) {
        if(!(model = parse_obj(path))) {
            printf("Failed to load %s\n", path);
            return nullptr;
        }

        Str path_str = alloc_str_from_arena(g_string_arena, path);
        model->filename_ = path_str;
        db.push({ alloc_str_from_arena(g_string_arena, name).data, path_str.data, model});

        printf("Loaded model\n\tv:%d uv:%d n:%d\n", (int)model->n.size(), (int)model->t.size(), (int)model->n.size());
    } else {
        printf("Model resource with name %s already exitst\n", name);
    }
    return model;
}

Image* add_resource(BufferT<TextureRes, int>& db, const char* name, const char* path) {
    Image* img = get_texture(name);
    if(!img) {
        if(!(img = tga_load(path))) {
            printf("Failed to load %s\n", path);
            return nullptr;
        }

        Str path_str = alloc_str_from_arena(g_string_arena, path);
        img->filename_ = path_str;
        db.push({ alloc_str_from_arena(g_string_arena, name).data, path_str.data, img});

        printf("Loaded texture %s\n", path);
    } else {
        printf("Texture resource with name %s already exitst\n", name);
    }
    return img;
}

bool load_resources() {
    //test_str();
    for(int i=0;i<g_models.size();++i) {
        if(g_models[i].name && !g_models[i].model) {
            g_models[i].model = parse_obj(g_models[i].path);
            assert(g_models[i].model);
            g_models[i].model->filename_ = alloc_str_from_arena(g_string_arena, g_models[i].path);
            if(!g_models[i].model) {
                printf("Failed to load %s\n", g_models[i].path);
                return false;
            }
            printf("Loaded model\n\tv:%d uv:%d n:%d\n", (int)g_models[i].model->n.size(), (int)g_models[i].model->t.size(), (int)g_models[i].model->n.size());
        }
    }

    for(int i=0;i<g_textures.size();++i) {
        if(g_textures[i].name && !g_textures[i].tex) {
            g_textures[i].tex = tga_load(g_textures[i].path);
            assert(g_textures[i].tex);
            g_textures[i].tex->filename_ = alloc_str_from_arena(g_string_arena, g_textures[i].path);
            if(!g_textures[i].tex) {
                printf("Failed to load %s\n", g_textures[i].path);
                return false;
            }
        }
    }

    return true;
}

struct { 
    u64 traverse;
    u64 render;
    u64 freq;

    u64 time_prev;
    u64 ticks_prev;
    u64 time;
    u64 ticks;

    bool init() {

        // also see https://agner.org/optimize/optimizing_cpp.pdf for rdtsc usage

        cpu_set_t cpuset;
        pthread_t thread = pthread_self();

        CPU_ZERO(&cpuset);
        CPU_SET(0, &cpuset);

        int s = pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);
        return s == 0;
    }

    void begin() {
        traverse = 0;
        render = 0;
        time = get_time_nsec();
        __builtin_ia32_lfence();
        //__get_cpuid();
        ticks = rdtsc();
        freq = 1000000000 * (ticks - ticks_prev) / (time - time_prev);
    }

    void end() {
        time_prev = time;
        ticks_prev = ticks;
    }

    double tick2time(u64 ticks) const {
        return (double)ticks / freq;
    }

} g_stats;

void on_init() {
    bool b_test_fixed_point_snapping_causes_very_thin_triangles_to_flip_side = false;
    if(b_test_fixed_point_snapping_causes_very_thin_triangles_to_flip_side) {

        Pt<float> v0f(90.318260f, 178.040451f);
        Pt<float> v1f(212.846146f,169.207764f);
        Pt<float> v2f(223.862900f,168.413589f);
        float A2f = edge(v0f, v1f, v2f);

        //Pt<FP16> v0(1444, 2848); // 90.250, 178.000
        //Pt<FP16> v1(3408, 2708); // 213.000, 169.250
        //Pt<FP16> v2(3584, 2696); // 224.000, 168.500
        Pt<FP16> v0(toFixed(v0f.x), toFixed(v0f.y)); // 90.250, 178.000
        Pt<FP16> v1(toFixed(v1f.x), toFixed(v1f.y)); // 213.000, 169.250
        Pt<FP16> v2(toFixed(v2f.x), toFixed(v2f.y)); // 224.000, 168.500

        Pt<float> v0fp = fromFixed(v0);
        Pt<float> v1fp = fromFixed(v1);
        Pt<float> v2fp = fromFixed(v2);
        float A2 = edge(v0fp, v1fp, v2fp);

        printf("FP 2Area (inv) = %.2f (%.2f) float 2Area (inv) = %.2f (%.2f)\n", A2, 1.0f/A2, A2f, 1.0f/A2f);

    }

    if(!g_stats.init()) {
        printf("Stats init failed, results may be incorrect\n");
    }

    g_string_arena = { (char*)malloc(16*1024), 0, 16*1024 };

    g_models.push({"pumpkin_03", "./data/halloween/Models/pumpkin_03.obj", nullptr});
    g_models.push({"pumpkin_03_hat", "./data/halloween/Models/pumpkin_03_hat.obj", nullptr});
    g_models.push({"candle_01", "./data/halloween/Models/candle_01.obj", nullptr});
    g_models.push({"text", "./data/halloween/Halloween_text.obj", nullptr});
    g_models.push({"triangle", "./data/triangle.obj", nullptr});
    g_models.push({"o", "./data/test/o2.obj", nullptr});
    g_models.push({"o2", "./data/test/o2.obj", nullptr});
    g_models.push({"o2", "./data/sci-fi/_models/uvmap.tga", nullptr});

    g_textures.push({ "pumpkin_03", "./data/halloween/Textures/pumpkin_03.tga", nullptr });
    g_textures.push({ "Candles", "./data/halloween/Textures/tga/Candles.tga", nullptr });
    g_textures.push({ "noise", nullptr, make_some_noise(256, 256)});

    g_textures.push({ "fighter_wilko", "./data/sci-fi/_models/wilko.tga", nullptr });
    g_textures.push({ "fighter_uvmap", "./data/sci-fi/_models/uvmap.tga", nullptr });

    if(!load_resources()) {
        exit(1);
    }

    g_objects.push({alloc_str_from_arena(g_string_arena, "pumpkin1"), get_model("pumpkin_03"), get_texture("pumpkin_03"), vec3(0, 0, -1), vec3(0), vec3(1) });
    g_objects.push({alloc_str_from_arena(g_string_arena, "hat"), get_model("pumpkin_03_hat"), get_texture("pumpkin_03"), vec3(0, 0, -1), vec3(0), vec3(1) });
    g_objects.push({alloc_str_from_arena(g_string_arena, "text"), get_model("text"), get_texture("noise"), vec3(0, 0, -2), vec3(0), vec3(1) });
    g_objects.push({alloc_str_from_arena(g_string_arena, "candle01"), get_model("candle_01"), get_texture("candles"), vec3(1, 0, -2), vec3(0), vec3(1) });

    g_dyn_texture = DT_Create(g_tex_w, g_tex_h);

    g_fb_size_bytes = g_tex_w*g_tex_h*sizeof(u32);
    g_fb = (u32*)malloc(g_fb_size_bytes);
    memset(g_fb, 0xff, g_fb_size_bytes);

    g_fb_z_size_bytes = g_tex_w*g_tex_h*sizeof(D);
    g_fb_z = (D*)malloc(g_fb_z_size_bytes);
    memset(g_fb_z, 0, g_fb_size_bytes);

    test_fixed_point();
    //exit(0);
    map0 = (bool*)malloc(g_tex_w*g_tex_h);
    map1 = (bool*)malloc(g_tex_w*g_tex_h);

}

struct TraverseDbgCtx {
    int Sc;
    int Ss;
    bool b_change_dir;
    bool b_advance_line;
    bool b_was_changing_dir;
    bool b_was_inside;
    bool b_just_crossed_mid_point;
    int x,y;
    int dir;

    int debug_triangle_idx;
    int num_steps_to_traverse;
    int cur_step;
    int break_on_step;
    bool b_need_break;

    TraverseDbgCtx():break_on_step(-1) {
        b_need_break = false;
    }

    void init() {
        cur_step = 0;

        x = 0;
        y = 0;
        dir = 0;
        Sc = 0;
        Ss = 0;
        b_change_dir = false;
        b_advance_line = false;
        b_was_changing_dir = false;
        b_just_crossed_mid_point = false;
        b_was_inside = false;
    }

    bool step(int tri_idx) {
        if(tri_idx == debug_triangle_idx) {
            cur_step++;
        }
        return cur_step > num_steps_to_traverse;
    }
    bool need_brake() const {
        return b_need_break && cur_step == break_on_step;
    }

} g_traverse_dgb_ctx;


void traverse_aabb(const float sample_offset, TriSetup<S>* tris, int count) {
    
    const uint64_t start = rdtsc();
    for(int i=0;i<count;++i) {

        g_traverse_dgb_ctx.init();

        const bool b_is_active_tri = g_active_triangle <= (int)g_tris.size() && g_active_triangle >= 0 && &g_tris[g_active_triangle] == tris + i;
        if(g_b_draw_only_active_tri && !b_is_active_tri) {
            continue;
        }


        const TriSetup<S>& ts = tris[i];

        // small thin tris, could give incorrect area
        if(ts.oo2A > 1000.0f && !g_b_draw_degenerate_tris) {
            continue;
        }

        if(isinf(ts.oo2A) || ts.oo2A < 0) 
            continue;

        S minX = min3(ts.v[0].x, ts.v[1].x, ts.v[2].x);
        S minY = min3(ts.v[0].y, ts.v[1].y, ts.v[2].y);
        S maxX = max3(ts.v[0].x, ts.v[1].x, ts.v[2].x);
        S maxY = max3(ts.v[0].y, ts.v[1].y, ts.v[2].y);

        minX = max(minX, (S)0);
        minY = max(minY, (S)0);
        maxX = min(maxX, toFixed(g_tex_w - 1));
        maxY = min(maxY, toFixed(g_tex_h - 1));

        int min_xi = fromFixed2Int(minX);
        int max_xi = fromFixed2Int(maxX);
        int min_yi = fromFixed2Int(minY);
        int max_yi = fromFixed2Int(maxY);
        const int Width = max_xi - min_xi;
        const int Height = max_yi - min_yi;

        constexpr const bool FpPrec = false;
        constexpr const int EdgeFracBits = FpPrec ? 2*FP_FracBits : FP_FracBits;

        //printf("%d) minX:%d minY:%d maxX:%d maxY:%d\n", i, minX, minY, maxX, maxY);
        //printf("%d) min_x:%d min_y:%d max_x:%d max_y:%d\n", i, min_xi, min_yi, max_xi, max_yi);

        // should always be sampling point, not some arbitrary min triangle AABB point, because all triangles should have same fractional offset inside a pixel
        // we may add sampling offset in the inner loop to x & y
        Pt<S> o(toFixed(min_xi + sample_offset), toFixed(min_yi + sample_offset));

        i32 e0o = ts.e0<FpPrec>(o);
        i32 e1o = ts.e1<FpPrec>(o);
        i32 e2o = ts.e2<FpPrec>(o);

        i32 e0 = e0o, e1 = e1o, e2 = e2o;
        i32 e0row = e0o, e1row = e1o, e2row = e2o;

        const float oo2A = ts.oo2A;

        Grad<float> ooZ_grad(ts, ts.ooHw[0]*oo2A, ts.ooHw[1]*oo2A, ts.ooHw[2]*oo2A);
        float ooZ_row = ooZ_grad.e(vec2(min_xi + sample_offset, min_yi + sample_offset));
        float ooZ = 0;

        Grad<vec4> uv_grad(ts, ts.oo_attribs[ATT_TEXCOORD][0]*oo2A, ts.oo_attribs[ATT_TEXCOORD][1]*oo2A, ts.oo_attribs[ATT_TEXCOORD][2]*oo2A);
        vec4 uv_row = uv_grad.e(vec2(min_xi + sample_offset, min_yi + sample_offset));
        vec4 uv;

        for (i32 y = 0; y <= Height; y++) {

            if(g_b_delta_update) {
                e0 = e0row;
                e1 = e1row;
                e2 = e2row;
                uv = uv_row;
                ooZ = ooZ_row;
            }

            for (i32 x = 0; x <= Width; x++) {

                Pt<S> p(o.x + toFixed(x), o.y + toFixed(y));
            
                if(!g_b_delta_update) {
                    e0 = ts.e0<FpPrec>(p);
                    e1 = ts.e1<FpPrec>(p);
                    e2 = ts.e2<FpPrec>(p);
                    vec2 pflt = vec2(fromFixed(p.x), fromFixed(p.y));
                    uv = uv_grad.e(pflt);
                    ooZ = ooZ_grad.e(pflt);
                }

                bool b_w;
                if(g_b_no_overdraw) {
                    b_w = ts.inside0(e0) && ts.inside1(e1) && ts.inside2(e2);
                } else {
                    b_w = (e0 | e1 | e2) > 0; // negative sign will be preserved if using bitwise OR, so only one bool check
                }

                if (b_w) {
                    int px = min_xi + x;
                    int py = min_yi + y;
                    // can remove oo2A multiply and do it after interpolation (or better even incorporate into values, to do it only once at setup)
                    const vec3 ev = vec3(fromFP<EdgeFracBits>(e0), fromFP<EdgeFracBits>(e1), fromFP<EdgeFracBits>(e2));
                    float u = ev[0]*oo2A;
                    float v = ev[1]*oo2A;
                    float w = (1-u-v);
                    float pp_denom = (ev[0]*ts.ooHw[0] + ev[1]*ts.ooHw[1] + ev[2]*ts.ooHw[2]);
                    float upp = ev[0]*ts.ooHw[0]/pp_denom;
                    float vpp = ev[1]*ts.ooHw[1]/pp_denom;
                    float wpp = (1-upp-vpp);
                    if(g_b_persp_corr && g_b_persp_corr_2nd_way) {
                        u = upp;
                        v = vpp;
                        w = wpp;
                    }
                    assert(u+v+w <=1+eps && u+v+w>=0);
                    if((int)g_sel_pix_x == px && (int)g_sel_pix_y == py) {
                        g_uvw_under_cursor = vec4(u, v, w);
                    }

                    const uint64_t rs = rdtsc();
                    float z = 1.0f/ooZ;
                    vec4 att[ATT_COUNT];
                    att[ATT_TEXCOORD] = uv*z;
                    att[ATT_COLOR] = vec4(0);
                    att[ATT_NORMAL] = vec4(0);
                    renderPixel(px, py, u, v, w, ts, b_is_active_tri ? g_tri_color|0xf0000000 : g_tri_color /*0xAA55AA55*/, att, z, ooZ);
                    g_stats.render += rdtsc() - rs;
                }

                if(g_b_delta_update) {
                    e0 += ts.e0ddx<FpPrec>();
                    e1 += ts.e1ddx<FpPrec>();
                    e2 += ts.e2ddx<FpPrec>();
                    uv = uv + uv_grad.dx();
                    ooZ = ooZ + ooZ_grad.dx();
                }
            }

            if(g_b_delta_update) {
                e0row = e0row + ts.e0ddy<FpPrec>();
                e1row = e1row + ts.e1ddy<FpPrec>();
                e2row = e2row + ts.e2ddy<FpPrec>();
                uv_row = uv_row + uv_grad.dy();
                ooZ_row = ooZ_row + ooZ_grad.dy();
            }

        }
    }

    g_stats.traverse += rdtsc() - start;
}

int test_min3(int v0, int v1, int v2, int top) {

    int top_i = -1;
    if(v0 < v1) {
        top_i = v0 < v2 ? 0 : 2;
    } else {
        top_i = v1 < v2 ? 1 : 2;
    }

    assert(top_i == top);
    return top_i;
}

u8 to_bits(i32 e0, i32 e1, i32 e2) {
    u8 v = ((e0&0x80000000) | ((e1&0x80000000)>>1) | ((e2&0x80000000)>>2)) >> 29;
    return (~v)&0b111;
}


void traverse_zigzag(const float sample_offset, TriSetup<S>* tris, int count) {

    test_min3(0,1,2, 0);
    test_min3(0,2,1, 0);
    test_min3(1,0,2, 1);
    test_min3(2,0,1, 1);
    test_min3(1,2,0, 2);
    test_min3(2,1,0, 2);

    for(int i=0;i<count;++i) {
    //for(int i=8;i<9;++i) {
        g_traverse_dgb_ctx.init();

        const TriSetup<S>& ts = tris[i];

        S maxY = max3(ts.v[0].y, ts.v[1].y, ts.v[2].y);
        maxY = min(maxY, toFixed(g_tex_h - 1));
        int max_yi = fromFixed2Int(maxY);

        const float oo2A = ts.oo2A;

        // find topmost point
        int top_i = -1;
        if(ts.v[0].y < ts.v[1].y) {
            if(ts.v[0].y == ts.v[2].y) { // must pick leftmost point as initial direction is to the right
                top_i = ts.v[0].x < ts.v[2].x ? 0 : 2;
            } else {
                top_i = ts.v[0].y < ts.v[2].y ? 0 : 2;
            }
        } else if(ts.v[0].y > ts.v[1].y) {
            if(ts.v[1].y == ts.v[2].y) { // must pick leftmost point as initial direction is to the right
                top_i = ts.v[1].x < ts.v[2].x ? 1 : 2;
            } else {
                top_i = ts.v[1].y < ts.v[2].y ? 1 : 2;
            }
        } else { // == 
            if(ts.v[0].y < ts.v[2].y) {
                top_i = 0;
            } else if(ts.v[0].y > ts.v[2].y) {
                top_i = 2;
            } else {
                // all 3 pt have same y
                //assert(0 && "degenerate triangle");
                printf("degenerate tri: %d\n", i);
                continue;
            }
        }

        int yy[3] = {ts.v[0].y, ts.v[1].y, ts.v[2].y };
        sort3(yy);
        float mid_y = fromFixed(yy[1]);
        bool b_mid_y_degen = yy[1] == yy[0] || yy[1] == yy[2];

        const FP16 f_one = toFixed(1);

        int x = fromFixed2Int(ts.v[top_i].x);
        int y = fromFixed2Int(ts.v[top_i].y);
        i32 dir = 1;

        Pt<S> cur_pt(toFixed(x + sample_offset), toFixed(y + sample_offset));
        i32 e0s = ts.e0(cur_pt);
        i32 e1s = ts.e1(cur_pt);
        i32 e2s = ts.e2(cur_pt);

        i32 e0 = e0s;
        i32 e1 = e1s;
        i32 e2 = e2s;

        u8 Sc;// = to_bits(e0, e1, e2);
        u8 Ss = 0;

        // calc prev pt
        if(0)
        {
            Pt<S> prev_pt(toFixed(x + sample_offset), toFixed(y + sample_offset));
            i32 pe0 = ts.e0(prev_pt);
            i32 pe1 = ts.e1(prev_pt);
            i32 pe2 = ts.e2(prev_pt);
            Ss = to_bits(pe0, pe1, pe2);
        }

        bool b_was_inside = false;
        bool b_was_changing_dir = false;
        bool b_just_advanced_line = false;
        bool b_just_crossed_mid_point = false;

        do {

            if(g_traverse_dgb_ctx.need_brake()) {
                //raise(SIGTRAP); 
                g_traverse_dgb_ctx.b_need_break = false;
                g_traverse_dgb_ctx.break_on_step *= -1;
            }

            cur_pt = Pt<S>(toFixed(x + sample_offset), toFixed(y + sample_offset));

            bool b_w = (e0 | e1 | e2) > 0; // negative sign will be preserved if using bitwise OR, so only one bool check
            Sc = to_bits(e0, e1, e2);
            u8 NotSc = (~Sc)&0b111;


            bool b_change_dir_after_advance_line = (Sc!=0b111&&b_just_advanced_line);
            bool b_change_dir = !b_was_changing_dir && ((Sc!=0b111 && b_was_inside) || (NotSc|Ss) == 0b111 || b_change_dir_after_advance_line || (Sc!=0b111 && b_just_crossed_mid_point));
            // out of trianble and has opposite signs of edges
            bool b_advance_line = !b_change_dir && (Sc!=0b111) && (((Sc|Ss)==0b111 && !b_just_crossed_mid_point) || (b_was_changing_dir&b_was_inside));// && b_was_changing_dir;

            b_was_inside |= Sc==0b111;
            b_just_advanced_line = false;

            if(g_traverse_dgb_ctx.step(i))
                break;

            if(g_traverse_dgb_ctx.debug_triangle_idx == i) {
                g_traverse_dgb_ctx.Sc = Sc;
                g_traverse_dgb_ctx.Ss = Ss;
                g_traverse_dgb_ctx.x = x;
                g_traverse_dgb_ctx.y = y;

                g_traverse_dgb_ctx.b_was_changing_dir = b_was_changing_dir;
                g_traverse_dgb_ctx.b_was_inside = b_was_inside;
                g_traverse_dgb_ctx.b_just_crossed_mid_point = b_just_crossed_mid_point;
                g_traverse_dgb_ctx.dir = dir;
            }

            // opposite side check
            if(b_advance_line) {
                y++;
                b_was_inside = false;
                b_was_changing_dir = false;
                b_just_advanced_line = true;
                e0 = e0 + ts.e0ddy();
                e1 = e1 + ts.e1ddy();
                e2 = e2 + ts.e2ddy();

                bool sign0 = (y-1)+sample_offset - mid_y >=0;
                bool sign1 = y+sample_offset - mid_y >=0;
                b_just_crossed_mid_point = (sign0 ^ sign1) && !b_mid_y_degen ;

                Ss = Sc;
            }
            // end of triangle line check
            else if(b_change_dir) {
                dir = -dir;
                x = x + dir;
                b_was_changing_dir = true;
                //Ss = Sc;
                if(dir > 0) {
                    e0 = e0 + ts.e0ddx();
                    e1 = e1 + ts.e1ddx();
                    e2 = e2 + ts.e2ddx();
                } else {
                    e0 = e0 - ts.e0ddx();
                    e1 = e1 - ts.e1ddx();
                    e2 = e2 - ts.e2ddx();
                }
            }

            if(g_traverse_dgb_ctx.debug_triangle_idx == i) {
                g_traverse_dgb_ctx.b_change_dir = b_change_dir;
                g_traverse_dgb_ctx.b_advance_line = b_advance_line;
            }

            if(b_change_dir || b_advance_line) 
                continue;


            if(b_w) {

                bool b_inside = ts.inside0(e0) && ts.inside1(e1) && ts.inside2(e2);
                if(b_w && b_inside) {
                    int px = x;//fromFixed2Int(p.x);
                    int py = y;//fromFixed2Int(p.y);
                    float u = fromFixed(e0)*oo2A;
                    float v = fromFixed(e1)*oo2A;
                    float w = (1-u-v);//fromFixed(e2)*oo2A;
                    float pp_denom = (fromFixed(e0)*ts.ooHw[0] + fromFixed(e1)*ts.ooHw[1] + fromFixed(e2)*ts.ooHw[2]);
                    float upp = fromFixed(e0)*ts.ooHw[0]/pp_denom;
                    float vpp = fromFixed(e1)*ts.ooHw[1]/pp_denom;
                    float wpp = (1-upp-vpp);
                    if(g_b_persp_corr && g_b_persp_corr_2nd_way) {
                        u = upp;
                        v = vpp;
                        w = wpp;
                    }
                    assert(u+v+w <=1+eps && u+v+w>=0);
                    if((int)g_sel_pix_x == px && (int)g_sel_pix_y == py) {
                        g_uvw_under_cursor = vec4(u, v, w);
                    }

                    bool b_is_active_tri = g_active_triangle <= (int)g_tris.size() && g_active_triangle >= 0 && &g_tris[g_active_triangle] == tris + i;
                    vec4 dummy_att[ATT_COUNT] = {vec4(0)};
                    renderPixel(px, py, u, v, w, ts, b_is_active_tri ? g_tri_color|0xf0000000 : g_tri_color /*0xAA55AA55*/, dummy_att, 1, 1);           
                }
            }

            x += dir;
            if(dir > 0) {
                e0 = e0 + ts.e0ddx();
                e1 = e1 + ts.e1ddx();
                e2 = e2 + ts.e2ddx();
            } else {
                cur_pt.x = cur_pt.x + f_one;
                e0 = e0 - ts.e0ddx();
                e1 = e1 - ts.e1ddx();
                e2 = e2 - ts.e2ddx();
            }

        } while(y <= max_yi);
    } // for g_tris
}


void on_update() {

    memset(map0, 0, g_tex_w*g_tex_h);
    memset(map1, 0, g_tex_w*g_tex_h);
#if 0
    g_mproj = make_proj(g_proj_params.left, g_proj_params.right, 
            g_proj_params.top, g_proj_params.bottom, 
            g_proj_params.near, g_proj_params.far);
#else
    g_mproj = make_frustum(g_frustum.FOVY, g_frustum.aspect_w_by_h, g_frustum.near, g_frustum.far);
#endif


    init_scene(g_tris, g_tris_notclipped, g_vertices, g_mproj, g_mview);

#if 0
    uint64_t cur_time_ms = get_time_usec() / 1000;
    double cur_time_d = cur_time_ms;
    if(g_b_animate && g_tris.size()>0) {
        g_tris[0].v0.x = 20.0f + 40.0f*(0.5f*sin(g_freq*cur_time_d*1e-3*2.0*M_PI) + 0.5f);
    }
    v0 = v0 + g_v0_offset + Pt<float>(1.619f, 0);
#endif

    clearFB(g_clear_color, 1);
    setTexture(nullptr);

    beginFrame();

    const float sample_offset = 0.5f;

    //const int s = g_b_draw_only_active_tri ? g_active_triangle : 0;
    //const int e = g_b_draw_only_active_tri ? g_active_triangle + 1 : (int)g_tris.size();

    CALLGRIND_START_INSTRUMENTATION;
    CALLGRIND_TOGGLE_COLLECT;

    g_stats.begin();

    for(int i = 0; i < (int)g_draw_calls.size(); ++i) {
        const DrawCallInfo& dc = g_draw_calls[i];

        if(dc.ntri == 0) { // to avoid crash on g_tris[dc.s_tri] if in reality we do not have triangles
            continue;
        }

        if(g_b_draw_only_active_tri && (g_active_triangle < dc.s_tri || g_active_triangle > dc.s_tri + dc.ntri)) {
            continue;
        }

        setTexture(dc.texture);
        switch(g_traverse_type) {
            case kTraverseAABB:
                traverse_aabb(sample_offset, &g_tris[dc.s_tri], dc.ntri);
                break;
            case kTraverseZigZag:
                traverse_zigzag(sample_offset, &g_tris[dc.s_tri], dc.ntri);
                break;
        }
    }

    g_stats.end();

    CALLGRIND_TOGGLE_COLLECT;
    CALLGRIND_STOP_INSTRUMENTATION;

    endFrame();

    //if(s < (int)g_tris.size()) {

    //}

    ImGuiIO& io = ImGui::GetIO();

    if(g_b_draw_outline) {
        for(int i=0;i<(int)g_tris.size();++i) {
        //for(int i=0;i<(int)1;++i) {
            const TriSetup<S>& ts = g_tris[i];
            //bool blend = g_b_blend;
            //g_b_blend = false;
            Pt<int> v0 = Pt<int>(fromFixed2Int(ts.v[0].x), fromFixed2Int(ts.v[0].y));
            Pt<int> v1 = Pt<int>(fromFixed2Int(ts.v[1].x), fromFixed2Int(ts.v[1].y));
            Pt<int> v2 = Pt<int>(fromFixed2Int(ts.v[2].x), fromFixed2Int(ts.v[2].y));
            plotLine(v0.x, v0.y, v1.x, v1.y, g_outline_color);
            plotLine(v1.x, v1.y, v2.x, v2.y, g_outline_color);
            plotLine(v2.x, v2.y, v0.x, v0.y, g_outline_color);
            //g_b_blend = blend;
        }
    }

    {
    ImGui::Begin("Rasterizer", NULL, ImGuiWindowFlags_HorizontalScrollbar);
    if(g_b_draw_z_buf) {
        DT_Update(g_dyn_texture, g_fb_z, [](void* dst, void* puserdata) {
                u32* p = (u32*)dst;
                float* src = (float*)puserdata;
                bool b_invert = false;
                for(int y = 0; y< g_tex_h; ++y) {
                    for(int x = 0; x< g_tex_w; ++x) {
                        // depth -1 .. 1     
                        u32 v = (u32)(255*clamp(0.5f*src[y*g_tex_w + x] + 0.5f, 0.0f, 1.0f));
                        if(b_invert) {
                            v = 255 - v;
                        }
                        //0xAARRGGBB
                        p[y*g_tex_w + x] = v | (v<<8) | (v<<16) | (255<<24);
                    }
                }
            });
    } else {
        DT_Update(g_dyn_texture, g_fb, [](void* dst, void* puserdata) {
                u32* p = (u32*)dst;
                u32* src = (u32*)puserdata;
                bool b_invert = false;
                for(int y = 0; y< g_tex_h; ++y) {
                    for(int x = 0; x< g_tex_w; ++x) {
                        u32 v = src[y*g_tex_w + x];//(u32)clamp(Brightening*(amplitude + BrighteningOffset), 0.0f, 255.0f);
                        if(b_invert) {
                            v = 255 - v;
                        }
                        //0xAARRGGBB
                        //p[y*g_tex_w + x] = v | v<<8 | v<<16 | 255<<24;
                        p[y*g_tex_w + x] = v;
                    }
                }
            });
    }
    int scale = 1 << g_scale_idx;
    ImGui::Image((ImTextureID)(intptr_t)DT_GetTextureID(g_dyn_texture), ImVec2(scale*g_tex_w, scale*g_tex_h));

    const ImVec2 vMin = ImGui::GetWindowContentRegionMin();
    const ImVec2 vMax = ImGui::GetWindowContentRegionMax();
    const ImVec2 wPos = ImGui::GetWindowPos();
    ImGui::Text("Content min:%f,%f max:%f,%f\n", vMin.x, vMin.y, vMax.x, vMax.y);


    const int inc = 1<<g_scale_idx;
    ImDrawList* dl = ImGui::GetWindowDrawList(); // ImGui::GetForegroundDrawList()
                                                 //
    const ImVec2 pmin = ImVec2(wPos.x + vMin.x, wPos.y + vMin.y);
    const ImVec2 mousePos = io.MousePos;

    if(mousePos.x > pmin.x && mousePos.y> pmin.y) {
        // convert to pixel
        g_sel_pix_x = (mousePos.x - pmin.x)/inc;
        g_sel_pix_y = (mousePos.y - pmin.y)/inc;
    } else {
        g_sel_pix_x = -1;
        g_sel_pix_y = -1;
    }

    // grid
    if(g_b_draw_grid && inc > 1) {

        const ImU32 grid_col = IM_COL32( 200, 200, 200, 150);

        // horizontal lines
        for(int y = 0; y <= g_tex_h; y++) {
            const ImVec2 p0 = ImVec2(pmin.x,               pmin.y + y*inc);
            const ImVec2 p1 = ImVec2(pmin.x + g_tex_w*inc, pmin.y + y*inc);
            dl->AddLine( p0, p1, grid_col, 1 );
        }

        // vertical lines
        for(int x = 0; x <= g_tex_w; x++) {
            const ImVec2 p0 = ImVec2(pmin.x + x*inc, pmin.y);
            const ImVec2 p1 = ImVec2(pmin.x + x*inc, pmin.y + g_tex_h*inc);
            dl->AddLine( p0, p1, grid_col, 1 );
        }

        dl->AddCircle(ImVec2(g_traverse_dgb_ctx.x*inc + inc/2.0f + pmin.x, g_traverse_dgb_ctx.y*inc + inc/2.0f + pmin.y), inc/4.0f, IM_COL32(0, 200, 200, 255), 16, 2);
    }

    if(g_b_draw_wireframe) {
        // triangles
        for(int i=0;i<(int)g_tris.size();++i) {
            const TriSetup<S>& ts = g_tris[i];
            int p0x = fromFixed(ts.v[0].x) * inc;
            int p0y = fromFixed(ts.v[0].y) * inc;
            int p1x = fromFixed(ts.v[1].x) * inc;
            int p1y = fromFixed(ts.v[1].y) * inc;
            int p2x = fromFixed(ts.v[2].x) * inc;
            int p2y = fromFixed(ts.v[2].y) * inc;
            const ImVec2 e0s = ImVec2(p0x + pmin.x, p0y + pmin.y);
            const ImVec2 e0e = ImVec2(p1x + pmin.x, p1y + pmin.y);
            dl->AddLine( e0s, e0e, IM_COL32( 0, 200, 200, 200), 1 );
            const ImVec2 e1s = ImVec2(p1x + pmin.x, p1y + pmin.y);
            const ImVec2 e1e = ImVec2(p2x + pmin.x, p2y + pmin.y);
            dl->AddLine( e1s, e1e, IM_COL32( 0, 200, 200, 200), 1 );
            const ImVec2 e2s = ImVec2(p2x + pmin.x, p2y + pmin.y);
            const ImVec2 e2e = ImVec2(p0x + pmin.x, p0y + pmin.y);
            dl->AddLine( e2s, e2e, IM_COL32( 0, 200, 200, 200), 1 );
        }
    }

    if(g_b_draw_grid)
    {
        if(g_sel_pix_x>=0 && g_sel_pix_y>=0) {
            // selected pixel
            ImVec2 p0 = ImVec2(pmin.x + (int)g_sel_pix_x*inc,     pmin.y + (int)g_sel_pix_y*inc);
            ImVec2 p1 = ImVec2(pmin.x + ((int)g_sel_pix_x+1)*inc, pmin.y + ((int)g_sel_pix_y+1)*inc);
            dl->AddRectFilled( p0, p1, IM_COL32( 200, 200, 200, 100), 0, 0);

            // sampling point
            ImVec2 sampling_point = ImVec2(((int)g_sel_pix_x + sample_offset)*inc, ((int)g_sel_pix_y + sample_offset)*inc);
            p0 = ImVec2(pmin.x + sampling_point.x - 1, pmin.y + sampling_point.y - 1) ;
            p1 = ImVec2(pmin.x + sampling_point.x + 1, pmin.y + sampling_point.y + 1) ;
            dl->AddRectFilled( p0, p1, IM_COL32( 255, 100, 100, 255), 0, 0);

            // mouse -> fixed point
            Pt<S> fp = float2FPSnapped(g_sel_pix_x, g_sel_pix_y);
            ImVec2 fixed_point = ImVec2(ToUI(fp.x, inc), ToUI(fp.y, inc));
            p0 = ImVec2(pmin.x + fixed_point.x - 1, pmin.y + fixed_point.y - 1) ;
            p1 = ImVec2(pmin.x + fixed_point.x + 1, pmin.y + fixed_point.y + 1) ;
            dl->AddRectFilled( p0, p1, IM_COL32( 100, 255, 100, 255), 0, 0);
        }
    }

    if(ImGui::IsWindowHovered()) {
        Pt<S> fp = Pt<S>(toFixed(g_sel_pix_x), toFixed(g_sel_pix_y));
        if(ImGui::IsKeyPressed(ImGuiKey_A)) {
            for(int i=0;i<(int)g_tris.size();++i) {
                const TriSetup<S>& ts = g_tris[i];
                if(ts.inside(fp)) {
                    g_active_triangle = i;
                }
            }
        } else if(ImGui::IsKeyDown(ImGuiKey_ModCtrl) && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            g_tri_adder.add_point(g_sel_pix_x, g_sel_pix_y, WorkingPoints::kTris);
        }
    }


    // draw current edited triange
    Pt<S> fp = Pt<S>(toFixed(g_sel_pix_x), toFixed(g_sel_pix_y));
    for(int i=0;i<g_tri_adder.count;++i) {

        // that is a lot of conversions...

        Pt<float> p0 = g_tri_adder.get(i);
        Pt<float> p1 = (i == g_tri_adder.count - 1) ? Pt<float>(fromFixed(fp.x), fromFixed(fp.y)) : g_tri_adder.get(i+1);

        Pt<S> fp0 = float2FPSnapped(p0.x, p0.y);
        Pt<S> fp1 = float2FPSnapped(p1.x, p1.y);

        const ImVec2 ps = ImVec2(ToUI(fp0.x, inc) + pmin.x, ToUI(fp0.y, inc) + pmin.y);
        const ImVec2 pe = ImVec2(ToUI(fp1.x, inc) + pmin.x, ToUI(fp1.y, inc) + pmin.y);

        dl->AddLine( ps, pe, IM_COL32(255, 255, 255, 255), 1 );
    }


    ImGui::Text("Application average %.3f ms/frame (%.1f FPS) z-fail:%d", 1000.0f / io.Framerate, io.Framerate, g_num_z_fail);
    ImGui::End();

    }

    // property panel
    {
        static ImVec4 clear_color = ImVec4(0.4f, 0.4f, 0.4f, .4f);
        static ImVec4 tri_color = ImVec4(2/255.0f, 93/255.0f, 165.0f/255.0f, 188/255.0f);
        static ImVec4 outline_color = ImVec4(0.1f, 0.1f, 0.1f, 1.f);

        ImGui::Begin("Setup");

        if(ImGui::BeginTabBar("Properties")) {


            if(ImGui::BeginTabItem("Info")) {

                ImGui::SliderFloat("g_s:", &g_s, 0.01f, 5.0f);
                ImGui::SliderFloat2("g_p:", (float*)&g_p, -3, 3);
                ImGui::SliderFloat("g_pz:", &g_p.z, -g_proj_params_def.near, -50*g_proj_params_def.near);

                ImGui::SeparatorText("Rasterizer:");
                ImGui::Checkbox("Tie breaker rule", &g_b_no_overdraw); ImGui::SameLine();
                ImGui::Checkbox("Delta Update", &g_b_delta_update); ImGui::SameLine();
                ImGui::Checkbox("Persp Corr", &g_b_persp_corr);
                ImGui::Checkbox("Persp Corr 2nd Way", &g_b_persp_corr_2nd_way); ImGui::SameLine();

                const char* cull_face_str[] = { "None", "CW", "CCW" };
                ImGui::SetNextItemWidth(80);
                if (ImGui::BeginCombo("Cull Face Override:", cull_face_str[g_cull_face], 0)) {
                    for (int n = 0; n < IM_ARRAYSIZE(cull_face_str); n++) {

                        const bool is_selected = (g_cull_face == n);
                        if (ImGui::Selectable(cull_face_str[n], is_selected))
                            g_cull_face = n;

                        // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }

                ImGui::Checkbox("Z-test", &g_b_depth_test_enable); ImGui::SameLine();
                ImGui::Checkbox("Z-write", &g_b_depth_write_enable);
                ImGui::Checkbox("Clip", &g_b_clip);
                if(g_b_clip) {
                    ImGui::SameLine();
                    ImGui::CheckboxFlags("-x", &g_clip_mask, 0x1);ImGui::SameLine();
                    ImGui::CheckboxFlags("+x", &g_clip_mask, 0x2);ImGui::SameLine();
                    ImGui::CheckboxFlags("-y", &g_clip_mask, 0x4);ImGui::SameLine();
                    ImGui::CheckboxFlags("+y", &g_clip_mask, 0x8);ImGui::SameLine();
                    ImGui::CheckboxFlags("-z", &g_clip_mask, 0x10);ImGui::SameLine();
                    ImGui::CheckboxFlags("+z", &g_clip_mask, 0x20);
                    ImGui::CheckboxFlags("w", &g_clip_mask, 0x40);
                }

                ImGui::SeparatorText("Debug Draw:");
                ImGui::Checkbox("Blend", &g_b_blend);ImGui::SameLine();
                ImGui::Checkbox("Outline", &g_b_draw_outline); ImGui::SameLine();
                ImGui::Checkbox("Zbuf", &g_b_draw_z_buf); ImGui::SameLine();
                ImGui::Checkbox("DrawDegenerate", &g_b_draw_degenerate_tris); ImGui::SameLine();
                ImGui::Checkbox("Animate", &g_b_animate); ImGui::SameLine();
                ImGui::Checkbox("Grid", &g_b_draw_grid);
                ImGui::Checkbox("Wireframe", &g_b_draw_wireframe);


                const char* traverse_types[] = { "AABB", "ZigZag" };
                const char* combo_preview_value = traverse_types[(int)g_traverse_type];
                if (ImGui::BeginCombo("Traverse Type:", combo_preview_value, 0))
                {
                    for (int n = 0; n < IM_ARRAYSIZE(traverse_types); n++)
                    {
                        const bool is_selected = (g_traverse_type == n);
                        if (ImGui::Selectable(traverse_types[n], is_selected))
                            g_traverse_type = (eTraverseType)n;

                        // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }

                ImGui::SliderFloat("Anim Freq:", &g_freq, 0.05f, 1.0f);
                ImGui::Text("%d", 1<<g_scale_idx);ImGui::SameLine();
                ImGui::SliderInt("Scale:", &g_scale_idx, 0, 7);

                ImGui::SliderFloat("+v0.x:", &g_v0_offset.x, 0, 50.0f);
                ImGui::SliderFloat("+v0.y:", &g_v0_offset.y, 0, 50.0f);

                ImGui::Checkbox("Exclusive", &g_b_draw_only_active_tri);ImGui::SameLine();
                if(ImGui::InputInt("Active Triangle:", &g_active_triangle)) {
                    g_active_triangle = (int)min((size_t)g_active_triangle, g_tris.size() - 1);
                }

                extern void draw_traversal_debugger_ui();
                draw_traversal_debugger_ui();

                // our convention is 0xAARRGGBB (unline ImGUI 0xAABBGGRR, so we swizzle here)
                if(!g_b_imgui_initialized || ImGui::ColorEdit3("clear color", (float*)&clear_color)) {
                    g_clear_color = IM_COL32(255*clear_color.z, 255*clear_color.y, 255*clear_color.x, 255*clear_color.w);                                                                 
                    printf("Clear Color: %x R:%.2f G:%.2f B:%.2f A:%.2f\n", g_clear_color, clear_color.x, clear_color.y, clear_color.z, clear_color.w);
                }
                if(!g_b_imgui_initialized || ImGui::ColorEdit4("tri color", (float*)&tri_color)) {
                    g_tri_color = IM_COL32(255*tri_color.z, 255*tri_color.y, 255*tri_color.x, 255*tri_color.w);                                                                 
                }

                if(!g_b_imgui_initialized || ImGui::ColorEdit4("outline color", (float*)&outline_color)) {
                    g_outline_color = IM_COL32(255*outline_color.z, 255*outline_color.y, 255*outline_color.x, 255*outline_color.w);                                                                 
                }

                if(g_sel_pix_x>=0 && g_sel_pix_y>=0 && g_sel_pix_x < g_tex_w && g_sel_pix_y < g_tex_h) {
                    u32 color = g_fb[(int)g_sel_pix_y*g_tex_w + (int)g_sel_pix_x];
                    FColor c = FColor::fromU32(color);
                    ImGui::ColorButton("PixColor##3c", ImVec4(c.r, c.g, c.b, c.a), 0, ImVec2(80, 80));
                    ImGui::SameLine();
                    ImGui::Text("R:%d G:%d B:%d A:%d", color&0xFF, (color>>8)&0xFF, (color>>16)&0xFF, (color>>24)&0xFF);
                    ImGui::SameLine();
                    ImGui::Text("U:%.3f V:%.3f W:%.3f", g_uvw_under_cursor.x, g_uvw_under_cursor.y, g_uvw_under_cursor.z);
                    ImGui::Text("Depth:%f hw(-zview):%f", g_pix_info_under_cursor.z, g_pix_info_under_cursor.hw);
                    for(int i=0; i<ATT_COUNT; i++) {
                        ImGui::Text("Attr%d: %.3f %.3f %.3f %.3f", i, g_pix_info_under_cursor.attr[i].x, g_pix_info_under_cursor.attr[i].y,
                                g_pix_info_under_cursor.attr[i].z, g_pix_info_under_cursor.attr[i].w);
                    }
                }

                ImGui::Text("Sel Pix: %.4f %.4f", g_sel_pix_x, g_sel_pix_y);
                ImGui::Text("Sampl Pix: %.4f %.4f", (int)g_sel_pix_x + sample_offset, (int)g_sel_pix_y + sample_offset);

                if(g_sel_pix_x>=0 && g_sel_pix_y>=0 && g_active_triangle < (int)g_tris.size() && g_active_triangle>=0)
                {
                    const TriSetup<S>& ts = g_tris[g_active_triangle];

                    struct {
                        Pt<S> pp;
                        const char* const name;
                    } PixData[] = {
                        {Pt<S>(toFixed(g_sel_pix_x), toFixed(g_sel_pix_y)), "Fixed Point"},
                        {Pt<S>(toFixed((int)g_sel_pix_x + sample_offset), toFixed((int)g_sel_pix_y + sample_offset)), "Pixel Sample Point"}
                    };

                    for(int i=0;i<2;++i) {
                        Pt<S> pix = PixData[i].pp;
                        float e0 = fromFP<2*FP_FracBits>(ts.e0(pix));
                        float e1 = fromFP<2*FP_FracBits>(ts.e1(pix));
                        float e2 = fromFP<2*FP_FracBits>(ts.e2(pix));
                        bool b0 = ts.inside0(pix);
                        bool b1 = ts.inside1(pix);
                        bool b2 = ts.inside2(pix);
                        bool b_inside = b0 && b1 && b2;
                        ImGui::Text("%s: %.4f %.4f", PixData[i].name, fromFixed(pix.x), fromFixed(pix.y));
                        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(b_inside?0:255, b_inside?255:0, 100, 255)); 
                        ImGui::Text("e0:%.2f/%d e1:%.2f/%d e2:%.2f/%d", e0, b0, e1, b1, e2, b2);
                        ImGui::PopStyleColor(); 
                    }
                }

                if (ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_None)) {
                    extern void draw_object_info(ObjList& o);
                    draw_object_info(g_objects);

                }

                if (ImGui::CollapsingHeader("Triangles", ImGuiTreeNodeFlags_None)) {
                    extern void draw_triangle_info(TriBuffer& tris);
                    draw_triangle_info(g_tris);
                }
                if (ImGui::CollapsingHeader("Triangles Before Clipping", ImGuiTreeNodeFlags_None)) {
                    extern void draw_triangle_info_f(TriBufferF& tris);
                    draw_triangle_info_f(g_tris_notclipped);
                }

                ImGui::EndTabItem();
            }
            if(ImGui::BeginTabItem("Projection")) {
                ImGui::DragFloat("left", &g_proj_params.left, 0.1f, 0, 10);
                ImGui::DragFloat("right", &g_proj_params.right, 0.1f, g_proj_params.left + 1, 100);
                ImGui::DragFloat("top", &g_proj_params.top, 0.1f, g_proj_params.bottom + 1, 100);
                ImGui::DragFloat("bottom", &g_proj_params.bottom, 0.1f, 0, 10);
                ImGui::DragFloat("near", &g_proj_params.near, 0.1f, 0, 1);
                ImGui::DragFloat("far", &g_proj_params.far, 0.1f, g_proj_params.near + 1, 10);
                if(ImGui::Button("Reset")) {
                    g_proj_params = g_proj_params_def;
        
                }
                ImGui::SeparatorText("Frustum:");
                ImGui::DragFloat("FOVY [deg]", &g_frustum.FOVY, 1, 10.0f, 170.0f);
                ImGui::DragFloat("Aspect [w/h]", &g_frustum.aspect_w_by_h, 0.1, 0.5f, 2.0f);
                ImGui::DragFloat("near##frustum_near", &g_frustum.near, 0.1f, .1f, 1.0f);
                ImGui::DragFloat("far##frustum_far", &g_frustum.far, 1, g_frustum.near + 10.0f, g_frustum.near + 100.0f);
                if(ImGui::Button("Reset##reset_frustum")) {
                    g_frustum = g_frustum_def;
        
                }

                ImGui::SeparatorText("Proj matrix:");
                if (ImGui::BeginTable("Proj", 4))
                {
                    for(int y=0;y<4;++y) {
                        ImGui::TableNextRow();
                        for(int x=0;x<4;++x) {
                            ImGui::TableSetColumnIndex(x);
                            ImGui::Text("%.4f", g_mproj.m[y*4 + x]);
                        }
                    }
                    ImGui::EndTable();
                }
                ImGui::EndTabItem();
            }
            if(ImGui::BeginTabItem("Shading")) {
                ImGui::RadioButton("Color", &g_shading_view_mode, eShadingViewMode::kColor); ImGui::SameLine();
                ImGui::RadioButton("Texture", &g_shading_view_mode, eShadingViewMode::kTexture);ImGui::SameLine();
                ImGui::RadioButton("Normals", &g_shading_view_mode, eShadingViewMode::kNormals);ImGui::SameLine();
                ImGui::RadioButton("Lighting", &g_shading_view_mode, eShadingViewMode::kLighting);ImGui::SameLine();
                ImGui::RadioButton("Shader", &g_shading_view_mode, eShadingViewMode::kShader);
                ImGui::Checkbox("Force checker", &g_b_force_checker);
                ImGui::Checkbox("UV override", &g_b_uv_override);
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }


        if(ImGui::Button("Save Scene")) {
            FILE* fp = fopen("scene.bin", "wb");
            if(fp) {
                const i32 count = (int)g_tris.size();
                fwrite(&count, sizeof(count), 1, fp);
                for(int i=0; i<count;++i) {
                    TriSetup<S> ts = g_tris[i];
                    fwrite(ts.v, sizeof(ts.v), 1, fp);
                    fwrite(ts.hz, sizeof(ts.hz), 1, fp);
                    fwrite(ts.hw, sizeof(ts.hw), 1, fp);
                    fwrite(ts.a, sizeof(ts.a), 1, fp);
                    fwrite(ts.b, sizeof(ts.b), 1, fp);
                    fwrite(ts.c, sizeof(ts.c), 1, fp);
                    fwrite(ts.t, sizeof(ts.t), 1, fp);
                }
                fclose(fp);
            }
        }

        ImGui::SameLine();

        if(ImGui::Button("Load Scene")) {
            FILE* fp = fopen("scene.bin", "rb");
            if(fp) {
                g_tris.reset();
                i32 count = 0;
                fread(&count, sizeof(count), 1, fp);
                for(int i=0; i<count;++i) {
                    //TriSetup<S> ts(Pt<float>(0,0), Pt<float>(0,0), Pt<float>(0,0));
                    TriSetup<S> ts(vec4(0,0,0,1), vec4(0,0,0,1), vec4(0,0,0,1));
                    fread(ts.v, sizeof(ts.v), 1, fp);
                    fread(ts.hz, sizeof(ts.hz), 1, fp);
                    fread(ts.hw, sizeof(ts.hw), 1, fp);
                    fread(ts.a, sizeof(ts.a), 1, fp);
                    fread(ts.b, sizeof(ts.b), 1, fp);
                    fread(ts.c, sizeof(ts.c), 1, fp);
                    fread(ts.t, sizeof(ts.t), 1, fp);
                    g_tris.push(ts);
                }
                fclose(fp);
            }
        }

        if(ImGui::Button("Save Objects")) {
            FILE* fp = fopen("objects.bin", "wb");
            if(fp) {
                const i32 count = (int)g_objects.size();
                fwrite(&count, sizeof(count), 1, fp);
                for(int i=0; i<count;++i) {
                    const Object& o = g_objects[i];

                    serialize(o.name_, fp);

                    const ModelRes* mres = find_res(o.model_);
                    serialize(from_charptr(mres->name), fp);
                    serialize(from_charptr(mres->path), fp);
                    const TextureRes* tres = find_res(o.texture_);
                    serialize(from_charptr(tres ? tres->name : 0), fp);
                    serialize(from_charptr(tres ? tres->path : 0), fp);

                    fwrite(&o.pos_, sizeof(o.pos_), 1, fp);
                    fwrite(&o.euler_, sizeof(o.euler_), 1, fp);
                    fwrite(&o.scale_, sizeof(o.scale_), 1, fp);
                    fwrite(&o.draw_flags, sizeof(o.draw_flags), 1, fp);
                }
                fclose(fp);
            }
        }

        ImGui::SameLine();
        static BufferT<DirList, int> dir_lists;
        //static DirList models_dl;
        const char* const models_base_dir[] = { "./data/halloween/Models/", "./data/test/", "./data/sci-fi/_models/", nullptr };
        if(ImGui::Button("Add Obj")) {
            ImGui::OpenPopup("dir_list");
            for(int i=0;i<dir_lists.size();++i) {
                destroy_dir_list(dir_lists[i]);
            }
            dir_lists.reset();
            int i=0;
            while(models_base_dir[i]) {
                dir_lists.push(make_dir_list(models_base_dir[i], "obj"));
                i++;
            }
        }
        if (ImGui::BeginPopup("dir_list")) {
            for(int d=0;d<dir_lists.size();++d) {
                ImGui::SeparatorText("------");
                DirList& models_dl = dir_lists[d];
                for (int i = 0; i < models_dl.entries.size(); i++) {
                    if (ImGui::Selectable(models_dl.entries[i].data)) {
                        Cut c = cut(models_dl.entries[i], '.');
                        Str s = alloc_str_from_arena(models_dl.arena, c.head.data, c.head.len + 1);
                        Str base = { models_base_dir[d], (ptrdiff_t)strlen(models_base_dir[d]) };
                        ObjModel* mdl = add_resource(g_models, s.data, alloc_concat_from_arena(models_dl.arena, base, models_dl.entries[i]).data);
                        Object o = {alloc_str_from_arena(g_string_arena, "objX"), mdl, nullptr, vec3(0, 0, -1), vec3(0), vec3(1) };
                        g_objects.push(o);
                    }
                }
            }
            ImGui::EndPopup();
        }

        ImGui::SameLine();

        if(ImGui::Button("Load Objects")) {
            FILE* fp = fopen("objects.bin", "rb");
            if(fp) {
                g_objects.reset();
                i32 count = 0;
                fread(&count, sizeof(count), 1, fp);
                for(int i=0; i<count;++i) {
                    Object o;
                    o.name_ = deserialize(fp);

                    Str model_name = deserialize(fp);
                    Str model_path = deserialize(fp);

                    Str tex_name = deserialize(fp);
                    Str tex_path = deserialize(fp);

                    o.model_ = add_resource(g_models, model_name.data, model_path.data);
                    o.texture_ = add_resource(g_textures, tex_name.data, tex_path.data); 

                    fread(&o.pos_, sizeof(o.pos_), 1, fp);
                    fread(&o.euler_, sizeof(o.euler_), 1, fp);
                    fread(&o.scale_, sizeof(o.scale_), 1, fp);
                    fread(&o.draw_flags, sizeof(o.draw_flags), 1, fp);

                    g_objects.push(o);
                }
                fclose(fp);
            }
        }



        if(ImGui::Button("Reset Scene")) {
            g_vertices.reset();
        }

        ImGui::SameLine();

        if(ImGui::Button("Clear Scene")) {
            g_tris.resize(0, true);
        }

        ImGui::Text("Freq:%.2fGHz r:%.4fms t:%.4fms f:%.4f\n", (double)g_stats.freq/1000000000, 
                1000*g_stats.tick2time(g_stats.render), 1000*g_stats.tick2time(g_stats.traverse - g_stats.render), 1000*g_stats.tick2time(g_stats.traverse));
        //printf("tp:%ld t:%ld (%ld) tickp:%ld tick:%ld (%ld)\n", g_stats.time_prev, g_stats.time, g_stats.time - g_stats.time_prev, g_stats.ticks_prev, g_stats.ticks, g_stats.ticks - g_stats.ticks_prev);
        //printf("Freq:%.2fGHz t:%.4fms r:%.4fms\n", (double)g_stats.freq/1000000000, 1000*g_stats.tick2time(g_stats.render), 1000*g_stats.tick2time(g_stats.traverse));

        ImGui::End();
    }

    g_b_imgui_initialized = true;

}

void draw_object_info(ObjList& ol) {
    ImGuiStyle& style = ImGui::GetStyle();
    ImGui::PushStyleVarY(ImGuiStyleVar_FramePadding, (float)(int)(style.FramePadding.y * 0.60f));
    ImGui::PushStyleVarY(ImGuiStyleVar_ItemSpacing, (float)(int)(style.ItemSpacing.y * 0.60f));

    int obj2remove = -1;
    for(int i=0; i<ol.size();++i) {
        Object& o = ol[i];
        // we know our names are null delimited
        if (ImGui::TreeNode((void*)(intptr_t)i, "%s %d %s", o.name_.data, i, g_active_obj == i ? "active":"")) {
            ImGui::SetNextItemWidth(80);
            ImGui::DragFloat("x", (float*)&o.pos_.x, 0.001f, -10, 10); ImGui::SameLine();
            ImGui::SetNextItemWidth(80);
            ImGui::DragFloat("y", (float*)&o.pos_.y, 0.001f, -10, 10);ImGui::SameLine();
            ImGui::SetNextItemWidth(80);
            ImGui::DragFloat("z", (float*)&o.pos_.z, 0.005f, -1.5*g_proj_params.near, -100);
            ImGui::DragFloat3("euler_", (float*)&o.euler_, 0.01f, 0, 2*M_PIf);
            ImGui::DragFloat3("scale", (float*)&o.scale_, 0.025f, 0.1f, 10);
            ImGui::Checkbox("blend", &o.draw_flags.b_blend); ImGui::SameLine();
            ImGui::DragFloat2("scroll uv", (float*)&o.draw_flags.scroll_uv);

            static BufferT<DirList,int> dir_lists;
            const char* const tex_base_dir[] = { "./data/halloween/Textures/tga/", "./data/sci-fi/_models/", nullptr};
            if (ImGui::Button("Texture..")) {
                ImGui::OpenPopup("select_texture");

                for(int i=0;i<dir_lists.size();++i) {
                    destroy_dir_list(dir_lists[i]);
                }
                dir_lists.reset();
                int i=0;
                while(tex_base_dir[i]) {
                    dir_lists.push(make_dir_list(tex_base_dir[i], "tga"));
                    i++;
                }
            }
            ImGui::SameLine();
            ImGui::TextUnformatted((o.texture_ && o.texture_->filename_.data) ? o.texture_->filename_.data : "<NULL>");

            if (ImGui::BeginPopup("select_texture")) {
                for(int d=0;d<dir_lists.size();++d) {
                    ImGui::SeparatorText("-----");
                    DirList& tex_dl = dir_lists[d];
                    for (int ti = 0; ti < tex_dl.entries.size(); ti++) {
                        if (ImGui::Selectable(tex_dl.entries[ti].data)) {
                            Cut c = cut(tex_dl.entries[ti], '.');
                            Str s = alloc_str_from_arena(tex_dl.arena, c.head.data, c.head.len + 1);
                            Str base = { tex_base_dir[d], (ptrdiff_t)strlen(tex_base_dir[d]) };
                            g_objects[i].texture_ = add_resource(g_textures, s.data, alloc_concat_from_arena(tex_dl.arena, base, tex_dl.entries[ti]).data);
                        }
                    }
                }
                ImGui::EndPopup();
            }

            if(ImGui::Button("Remove")) {
                obj2remove = i;
            }

            ImGui::TreePop();
        } // Tree Node
    }
    if(obj2remove != -1) {
        g_objects.remove_swap(obj2remove);
    }

    ImGui::PopStyleVar(2);
}

void draw_triangle_info(TriBuffer& tris) 
{
    ImGuiStyle& style = ImGui::GetStyle();
    ImGui::PushStyleVarY(ImGuiStyleVar_FramePadding, (float)(int)(style.FramePadding.y * 0.60f));
    ImGui::PushStyleVarY(ImGuiStyleVar_ItemSpacing, (float)(int)(style.ItemSpacing.y * 0.60f));

    for(int i=0; i<(int)tris.size();++i) {

        const TriSetup<S>& ts = tris[i];

        if (ImGui::TreeNode((void*)(intptr_t)&ts, "Triangle %d %s", i, g_active_triangle == i ? "active":"")) {

            ImGui::Text("Area inv: %f", ts.oo2A);

            static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV;
            if (ImGui::BeginTable("Setup", 4, flags))
            {
                // Submit columns name with TableSetupColumn() and call TableHeadersRow() to create a row with a header in each column.
                // (Later we will show how TableSetupColumn() has other uses, optional flags, sizing weight etc.)
                ImGui::TableSetupColumn("name");
                ImGui::TableSetupColumn("e0");
                ImGui::TableSetupColumn("e1");
                ImGui::TableSetupColumn("e2");
                ImGui::TableHeadersRow();

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("a");
                for (int column = 1; column < 4; column++)
                {
                    ImGui::TableSetColumnIndex(column);
                    ImGui::Text("%d (%.3f)", ts.a[column - 1], fromFixed(ts.a[column - 1]));
                }

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("b");
                for (int column = 1; column < 4; column++)
                {
                    ImGui::TableSetColumnIndex(column);
                    ImGui::Text("%d (%.3f)", ts.b[column - 1], fromFixed(ts.b[column - 1]));
                }

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("c");
                for (int column = 1; column < 4; column++)
                {
                    ImGui::TableSetColumnIndex(column);
                    ImGui::Text("%d (%.3f)", ts.c[column - 1], fromFixed(ts.c[column - 1]));
                }

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("t");
                for (int column = 1; column < 4; column++)
                {
                    ImGui::TableSetColumnIndex(column);
                    ImGui::Text("%d", ts.t[column - 1]);
                }

                ImGui::EndTable();
            }

            if (ImGui::BeginTable("Vertices", 3, flags))
            {
                ImGui::TableSetupColumn("name");
                ImGui::TableSetupColumn("x");
                ImGui::TableSetupColumn("y");
                ImGui::TableHeadersRow();

                for(int i=0;i<3;++i) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("v%d", i);
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%d (%.3f)", ts.v[i].x, fromFixed(ts.v[i].x));
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%d (%.3f)", ts.v[i].y, fromFixed(ts.v[i].y));
                }
                ImGui::EndTable();
            }

            ImGui::TreePop();
        } // Tree Node
    }
    ImGui::PopStyleVar(2);
}

// TODO: use draw_triangle_info for this as well
void draw_triangle_info_f(TriBufferF& tris) 
{
    ImGuiStyle& style = ImGui::GetStyle();
    ImGui::PushStyleVarY(ImGuiStyleVar_FramePadding, (float)(int)(style.FramePadding.y * 0.60f));
    ImGui::PushStyleVarY(ImGuiStyleVar_ItemSpacing, (float)(int)(style.ItemSpacing.y * 0.60f));

    for(int i=0; i<(int)tris.size();++i) {

        const TriSetup<float>& ts = tris[i];

        if (ImGui::TreeNode((void*)(intptr_t)&ts, "Triangle %d %s", i, g_active_triangle == i ? "active":"")) {

            ImGui::Text("Area inv: %f", ts.oo2A);

            static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV;
            if (ImGui::BeginTable("Setup", 4, flags))
            {
                // Submit columns name with TableSetupColumn() and call TableHeadersRow() to create a row with a header in each column.
                // (Later we will show how TableSetupColumn() has other uses, optional flags, sizing weight etc.)
                ImGui::TableSetupColumn("name");
                ImGui::TableSetupColumn("e0");
                ImGui::TableSetupColumn("e1");
                ImGui::TableSetupColumn("e2");
                ImGui::TableHeadersRow();

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("a");
                for (int column = 1; column < 4; column++)
                {
                    ImGui::TableSetColumnIndex(column);
                    ImGui::Text("%f (%.3f)", ts.a[column - 1], fromFixed(ts.a[column - 1]));
                }

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("b");
                for (int column = 1; column < 4; column++)
                {
                    ImGui::TableSetColumnIndex(column);
                    ImGui::Text("%f (%.3f)", ts.b[column - 1], fromFixed(ts.b[column - 1]));
                }

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("c");
                for (int column = 1; column < 4; column++)
                {
                    ImGui::TableSetColumnIndex(column);
                    ImGui::Text("%f (%.3f)", ts.c[column - 1], fromFixed(ts.c[column - 1]));
                }

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("t");
                for (int column = 1; column < 4; column++)
                {
                    ImGui::TableSetColumnIndex(column);
                    ImGui::Text("%d", ts.t[column - 1]);
                }

                ImGui::EndTable();
            }

            if (ImGui::BeginTable("Vertices", 4, flags))
            {
                ImGui::TableSetupColumn("name");
                ImGui::TableSetupColumn("x");
                ImGui::TableSetupColumn("y");
                ImGui::TableSetupColumn("z/w");
                ImGui::TableHeadersRow();

                for(int i=0;i<3;++i) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("v%d", i);
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%f (%.3f)", ts.v[i].x, fromFixed(ts.v[i].x));
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%f (%.3f)", ts.v[i].y, fromFixed(ts.v[i].y));
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("z:%.3f w:%.3f (%.3f)", ts.p[i].z, ts.p[i].w, ts.p[i].z/ts.p[i].w);
                }
                ImGui::EndTable();
            }

            if(i == 12) {
                printf("v0: %f %f\n", ts.v[0].x, ts.v[0].y);
                printf("v1: %f %f\n", ts.v[1].x, ts.v[1].y);
                printf("v2: %f %f\n", ts.v[2].x, ts.v[2].y);
                printf("Area0: %f\n", edge(ts.v[0], ts.v[1], ts.v[2]));
                printf("Area1: %f\n", edge(ts.v[1], ts.v[2], ts.v[0]));
                printf("Area2: %f\n", edge(ts.v[2], ts.v[0], ts.v[1]));

                Pt<double> v0d = Pt<double>(ts.v[0].x, ts.v[0].y);
                Pt<double> v1d = Pt<double>(ts.v[1].x, ts.v[1].y);
                Pt<double> v2d = Pt<double>(ts.v[2].x, ts.v[2].y);
                printf("Area0d: %f\n", edge(v0d, v1d, v2d));
                printf("Area1d: %f\n", edge(v1d, v2d, v0d));
                printf("Area2d: %f\n", edge(v2d, v0d, v1d));
            }

            ImGui::TreePop();
        } // Tree Node
    }
    ImGui::PopStyleVar(2);
}

void draw_traversal_debugger_ui() {

    if (ImGui::TreeNode((void*)(intptr_t)&g_traverse_dgb_ctx, "Tri Traversal Debugger")) {

        ImGui::SeparatorText("Traversal Debug");
        ImGui::Text("x:%d y:%d dir:%d step:%d", g_traverse_dgb_ctx.x, g_traverse_dgb_ctx.y, g_traverse_dgb_ctx.dir, g_traverse_dgb_ctx.cur_step);
        ImGui::Text("Ss:%x Sc:%x", g_traverse_dgb_ctx.Ss, g_traverse_dgb_ctx.Sc);
        ImGui::Text("WasInside:%d WasChDir:%d", g_traverse_dgb_ctx.b_was_inside, g_traverse_dgb_ctx.b_was_changing_dir);
        ImGui::Text("CD:%d AdvY:%d Mid:%d", g_traverse_dgb_ctx.b_change_dir, g_traverse_dgb_ctx.b_advance_line, g_traverse_dgb_ctx.b_just_crossed_mid_point);
        ImGui::InputInt("Debug Tri Idx:", &g_traverse_dgb_ctx.debug_triangle_idx, 1, 1);
        ImGui::InputInt("Num steps:", &g_traverse_dgb_ctx.num_steps_to_traverse, 1, 10);
        ImGui::InputInt("Break on step:", &g_traverse_dgb_ctx.break_on_step, 1, 10);
        ImGui::Checkbox("Break", &g_traverse_dgb_ctx.b_need_break);

        ImGui::TreePop();
    }
}

void on_exit() {
    delete g_obj;

    free(g_string_arena.mem);
    g_string_arena.capacity = 0;
    g_string_arena.size = 0;
}

#if 0

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;

float random (in vec2 st) {
    return fract(sin(dot(st.xy,
                         vec2(12.9898,78.233)))*
        43758.5453123);
}

// Based on Morgan McGuire @morgan3d
// https://www.shadertoy.com/view/4dS3Wd
float noise (in vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return (b-a)*u.x + a +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

#define OCTAVES 6
float fbm (in vec2 st) {
    // Initial values
    float value = 0.0;
    float amplitude = .5;
    float frequency = 0.;
    //
    // Loop of octaves
    for (int i = 0; i < OCTAVES; i++) {
        value += amplitude * noise(st);
        st *= 2.;
        amplitude *= .5;
    }
    return value;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec2 st = fragCoord/RESOLUTION.xy;
    //vec2 st = gl_FragCoord.xy/u_resolution.xy;
    st.x *= RESOLUTION.x/RESOLUTION.y;
    float orig_y = st.y;
    //st.y -= iTime;
    st.x = st.x + (fbm(st)-0.5 + 0.005*sin(7.0*(iTime + st.y)))*(1.0-orig_y);
    
    vec3 outer = vec3(1,0.5,0);
    vec3 inner = vec3(1,1,0);

    vec3 color = vec3(0.0);
    st = st * 3.0;
    st.y -= iTime/1.0;
    float v = fbm(st);
    
    float coeff = v * v * (3.0 - 2.0 * v);
    color += mix(outer, inner, coeff)*coeff*(1.0-orig_y);
    //color += vec3(coeff);

    fragColor = vec4(color,1.0);
}
#endif
