#pragma once
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)

#endif

#include <fstream>
#include <queue>

#include <string.h>

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif

namespace hnswlib {
    typedef size_t labeltype;

    template<typename T>
    static void writeBinaryPOD(std::ostream &out, const T &podRef) {
        out.write((char *) &podRef, sizeof(T));
    }

    template<typename T>
    static void readBinaryPOD(std::istream &in, T &podRef) {
        in.read((char *) &podRef, sizeof(T));
    }

    template<typename MTYPE>
    using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);


    template<typename MTYPE>
    class SpaceInterface {
    public:
        virtual ~SpaceInterface() {};
        virtual size_t get_data_size() = 0;
        virtual DISTFUNC<MTYPE> get_dist_func() = 0;
        virtual void *get_dist_func_param() = 0;
    };

    template<typename dist_t>
    class AlgorithmInterface {
    public:
        virtual ~AlgorithmInterface() {};
        virtual void addPoint(void *datapoint, labeltype label) = 0;
        virtual std::priority_queue<std::pair<dist_t, labeltype >> searchKnn(const void *, size_t) const = 0;
        virtual void saveIndex(const std::string &location) = 0;
    };


}

#include "space_l2.h"
#include "space_ip.h"
#include "bruteforce.h"
#include "hnswalg.h"
