#include <atomic>
#include <cstdio>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <set>
#include <fstream>
#include <unistd.h>
#include <condition_variable>
#include <queue>
#include <memory>
#include <chrono>
#include <cassert>
#include <sys/time.h>
#include <future>
#include <sstream>

#include "gflags/gflags.h"
#include "hnswlib/hnswlib.h"

DEFINE_string(fun, "", "function");
DEFINE_string(sp, "", "space");
DEFINE_string(idx, "", "index path");
DEFINE_string(km, "", "keymap path");
DEFINE_string(input, "", "input path");
DEFINE_string(qry, "", "query path");
DEFINE_string(res, "", "result path");
DEFINE_string(gt, "", "ground truth path");
DEFINE_string(atm, "8,16,32", "auto-tune M");
DEFINE_string(atefc, "500,1000,1500,2000", "auto-tune efc");
DEFINE_int32(dim, 0, "dim");
DEFINE_int32(m, 16, "M");
DEFINE_int32(efc, 200, "ef for construct");
DEFINE_int32(efs, 200, "ef for search");
DEFINE_int32(topk, 100, "top-k");
DEFINE_int32(tr, 20, "thread number");
DEFINE_int32(norm, 0, "renormal");
DEFINE_int32(verbose, 0, "evaluate verbose");
DEFINE_int32(ip2cos, 0, "redeuce ip to cos");

std::unique_ptr<hnswlib::SpaceInterface<float>> gSpace;
std::unique_ptr<std::vector<std::string>> gIdToKey;




// begin cutils

#define hassert(cond, fmt, ...)                     \
{                                                   \
    bool bCond = cond;                              \
    if (!bCond)                                     \
    {                                               \
        printf ( fmt "\n", ##__VA_ARGS__ );         \
    }                                               \
    assert(bCond);                                  \
}

namespace cutils {

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

class TimeDiff {
private:
	struct timeval m_startTime;
	struct timeval m_endTime;
public:
	TimeDiff() {gettimeofday(&m_startTime, NULL);}
	inline void Reset() {gettimeofday(&m_startTime, NULL);}
	inline void Stop() {gettimeofday(&m_endTime, NULL);}
    inline int ElapsedInSecond() {
        return (m_endTime.tv_sec - m_startTime.tv_sec);
    }
	inline int ElapsedInMillisecond() {
		return (m_endTime.tv_sec - m_startTime.tv_sec) * 1000 + 
			(m_endTime.tv_usec - m_startTime.tv_usec) / 1000;
    }
};

class AsyncWorker {
public:
    template <typename WorkerType, typename ...Args>
    static std::unique_ptr<AsyncWorker> Make(WorkerType func, Args&&... args) {
        return std::move(std::unique_ptr<AsyncWorker>(
                    new AsyncWorker(func, std::forward<Args>(args)...)));
    }

    template <typename WorkerType, typename ...Args>
    AsyncWorker(WorkerType func, Args... args) 
        : stop_(false) 
        , worker_(std::async(std::launch::async, 
                    func, std::forward<Args>(args)..., std::ref(stop_))) {
        assert(worker_.valid());
    }

    void Join() {
        worker_.get();
    }

    ~AsyncWorker() {
        stop_ = true;
        if (worker_.valid()) {
            worker_.get();
        }
    }

private:
    bool stop_;
    std::future<void> worker_;
};

using AsyncWorkerPtr = std::unique_ptr<AsyncWorker>;

template <typename EntryType>
class CQueue {
private:
    const size_t max_queue_size_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::unique_ptr<EntryType>> msg_;

public:
    explicit CQueue(size_t max_queue_size = 0) : 
        max_queue_size_(max_queue_size) {}

    void Push(std::unique_ptr<EntryType> item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            msg_.emplace(std::move(item));
            assert(nullptr != msg_.back());
            if (0 < max_queue_size_ && max_queue_size_ < msg_.size()) {
                msg_.pop();
            }
            assert(0 == max_queue_size_ || max_queue_size_ >= msg_.size());
        }
        cv_.notify_one();
    }

    void BatchPush(
            std::vector<std::unique_ptr<EntryType>>& vec_item, 
            size_t pop_batch_size = 1, 
            size_t pop_worker_cnt = 1) {
        if (true == vec_item.empty()) {
            return ;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& item : vec_item) {
                msg_.emplace(std::move(item));
                assert(nullptr != msg_.back());
            }
            for (size_t idx = 0; idx < vec_item.size(); ++idx) {
                if (0 == max_queue_size_ || max_queue_size_ >= msg_.size()) {
                    break;
                }

                msg_.pop();
            }
            assert(0 == max_queue_size_ || max_queue_size_ >= msg_.size());
        }

        assert(pop_batch_size > 0);
        size_t notify_cnt = 
            (vec_item.size() + pop_batch_size - 1) / pop_batch_size;
        assert(notify_cnt > 0);
        if (notify_cnt < pop_worker_cnt) {
            for (size_t idx = 0; idx < notify_cnt; ++idx) {
                cv_.notify_one();
            }
        } else {
            cv_.notify_all();
        }
    }

    std::unique_ptr<EntryType> Pop() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            while (msg_.empty()) {
                cv_.wait(lock, [&]() {
                        return !msg_.empty();
                        });
            }

            assert(false == msg_.empty());
            auto item = move(msg_.front());
            msg_.pop();
            return item;
        }
    }

    std::unique_ptr<EntryType> Pop(std::chrono::microseconds timeout) {
        auto time_point = std::chrono::system_clock::now() + timeout;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (false == 
                    cv_.wait_until(lock, time_point, 
                        [&]() {
                        return !msg_.empty();
                        })) {
                // timeout
                return nullptr;
            }

            assert(false == msg_.empty());
            auto item = move(msg_.front());
            msg_.pop();
            return item;
        }
    }

    std::vector<std::unique_ptr<EntryType>> BatchPop(
            size_t iMaxBatchSize, std::chrono::microseconds timeout) {
        auto time_point = std::chrono::system_clock::now() + timeout;
        std::vector<std::unique_ptr<EntryType>> vec;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (false == cv_.wait_until(lock, time_point, 
                        [&]() {
                        return !msg_.empty();
                        })) {
                return vec;
            }

            if (msg_.empty()) {
                return vec;
            }

            while (false == msg_.empty() && vec.size() < iMaxBatchSize) {
                assert(nullptr != msg_.front());
                auto item = std::move(msg_.front());
                msg_.pop();
                assert(nullptr != item);
                vec.push_back(std::move(item));
                assert(nullptr != vec.back());
            }
        }

        return std::move(vec);
    }

    std::vector<std::unique_ptr<EntryType>> BatchPop(size_t iMaxBatchSize) {
        std::vector<std::unique_ptr<EntryType>> vec;
        
        {
            std::unique_lock<std::mutex> lock(mutex_);
            while (msg_.empty()) {
                cv_.wait(lock, [&]() {
                        return !msg_.empty();
                        });
            }

            assert(false == msg_.empty());
            while (false == msg_.empty() && vec.size() < iMaxBatchSize) {
                assert(nullptr != msg_.front());
                auto item = std::move(msg_.front());
                msg_.pop();
                assert(nullptr != item);
                vec.push_back(std::move(item));
                assert(nullptr != vec.back());
            }

        }

        assert(false == vec.empty());
        return std::move(vec);
    }

    int BatchPopNoWait(
            size_t iMaxBatchSize, 
            std::vector<std::unique_ptr<EntryType>>& vec) {
        vec.clear();

        std::lock_guard<std::mutex> lock(mutex_);
        if (msg_.empty()) {
            return 1;
        }

        assert(false == msg_.empty());
        while (false == msg_.empty() && vec.size() < iMaxBatchSize) {
            assert(nullptr != msg_.front());
            auto item = std::move(msg_.front());
            msg_.pop();
            assert(nullptr != item);
            vec.push_back(std::move(item));
            assert(nullptr != vec.back());
        }

        return 0;
    }

    size_t Size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return msg_.size();
    }
};

}

// end cutils










int Access(const std::string& fn) {
    return access(fn.c_str(), F_OK);
}

void InitSpace() {
    if (gSpace != nullptr) {
        gSpace = nullptr;
    }
    assert(gSpace == nullptr);
    assert(FLAGS_dim > 0);

    if (FLAGS_sp == "ip") {
        gSpace = cutils::make_unique<hnswlib::InnerProductSpace>(FLAGS_dim);
    } else if (FLAGS_sp == "l2") {
        gSpace = cutils::make_unique<hnswlib::L2Space>(FLAGS_dim);
    } else {
        printf("Unsupport space %s\n", FLAGS_sp.c_str());
        assert(0);
    }
}

void Normalize(float* v, size_t dim) {
    float s = 0;
    if (dim % 16 == 0) {
        s = 1. - hnswlib::InnerProductSIMD16Ext(v, v, &dim);
    } else if (dim % 4 == 0) {
        s = 1. - hnswlib::InnerProductSIMD4Ext(v, v, &dim);
    } else {
        s = 1. - hnswlib::InnerProduct(v, v, &dim);
    }
    if (s > 0) {
        s = 1. / sqrt(std::max<float>(0., s));
        for (size_t i = 0; i < dim; ++i) v[i] *= s;
    }
}

using LineVector = std::vector<std::unique_ptr<std::string>>;
void LineHandler(
        cutils::CQueue<std::string>& que,
        int batch_size,
        std::vector<std::string>& tot_labels,
        std::vector<float>& tot_feats,
        std::mutex& wlock,
        bool& stop) {
    std::vector<std::string> labels;
    std::vector<float> feats;
    bool dic[256] = {0};
    dic[' '] = dic['\t'] = 1;
    while (!stop || que.Size() > 0) {
        auto vec = que.BatchPop(batch_size, std::chrono::milliseconds(10));
        labels.resize(vec.size());
        feats.resize(vec.size() * FLAGS_dim);
        for (size_t i = 0; i < vec.size(); ++i) {
            auto& line = *(vec[i]);
            size_t idx = 0;
            
            while (idx < line.size() && (int)line[idx] >= 0 && dic[(int)line[idx]]) idx++;
            assert(idx < line.size());
            int len = 0;
            while (idx + len < line.size()) {
                if ((int)line[idx + len] < 0) {
                    ++len;
                } else if (!dic[(int)line[idx + len]]) {
                    ++len;
                } else {
                    break;
                }
            }
            labels[i].assign(line.data() + idx, len);

            const char* pos = line.data() + idx + len;
            for (int j = 0; j < FLAGS_dim; ++j) {
                char* nxt = nullptr;
                feats[i * FLAGS_dim + j] = std::strtof(pos, &nxt);
                pos = nxt;
            }

            if (FLAGS_norm) {
                Normalize(&feats[i * FLAGS_dim], FLAGS_dim);
            }
        }

        if (vec.size() > 0) {
            std::lock_guard<std::mutex> guard(wlock);
            tot_labels.insert(tot_labels.end(), labels.begin(), labels.end());
            tot_feats.insert(tot_feats.end(), feats.begin(), feats.end());
        }
    }
}

void Downloader(
        cutils::CQueue<std::string>& que,
        int batch_size,
        const std::vector<std::string>& files,
        std::atomic<int>& idx_alloc,
        bool& stop) {
    while (true) {
        int idx = (idx_alloc++);
        if ((size_t)idx >= files.size()) break;

        const std::string& file = files[idx];
        std::ifstream reader(file);
        hassert(reader.good(), "Open %s failed", file.c_str());

        LineVector bufs;
        while (!reader.eof()) {
            auto line = cutils::make_unique<std::string>();
            std::getline(reader, *line);
            if (line->size() == 0) continue;

            bufs.push_back(std::move(line));
            if (bufs.size() >= (size_t)batch_size) {
                que.BatchPush(bufs, 100);
                bufs.clear();

                while (que.Size() > 1000000) {
                    usleep(100 * 1000);
                }
            }
        }
        if (bufs.size() > 0) {
            que.BatchPush(bufs, 100);
            bufs.clear();
        }
    }
}

void LoadData(
        const std::vector<std::string>& files,
        std::vector<std::string>& labels,
        std::vector<float>& feats) {
    std::mutex wlock;
    cutils::CQueue<std::string> que;
    std::vector<cutils::AsyncWorkerPtr> workers(FLAGS_tr);
    for (int i = 0; i < FLAGS_tr; ++i) {
        workers[i] = cutils::AsyncWorker::Make(
                &LineHandler, std::ref(que), 100,
                std::ref(labels), std::ref(feats), std::ref(wlock));
    }

    int reader_num = std::min<int>(10, files.size());
    std::vector<cutils::AsyncWorkerPtr> readers(reader_num);
    std::atomic<int> idx_alloc(0);
    for (int i = 0; i < reader_num; ++i) {
        readers[i] = cutils::AsyncWorker::Make(
                &Downloader, std::ref(que), 200, 
                std::cref(files), std::ref(idx_alloc));
    }

    readers.clear();
    workers.clear();
}

void Builder(
        hnswlib::HierarchicalNSW<float>* hnsw_index, 
        std::vector<std::string>& labels,
        std::vector<float>& feats,
        std::atomic<int>& idx_alloc,
        bool& stop) {
    while (true) {
        size_t idx = (idx_alloc++);
        if (idx >= labels.size()) break;

        uint32_t internal_id = hnsw_index->addPoint(
                feats.data() + FLAGS_dim * idx, (size_t)0, -1);
        gIdToKey->at(internal_id) = labels[idx];

        if (idx % 100000 == 0) {
            printf("Builder %ld %zu %zu\n", time(0), labels.size(), idx);
        }
    }
}

void ReadKeyMap() {
    std::ifstream rfile(FLAGS_km);
    hassert(rfile.good(), "Open %s failed", FLAGS_km.c_str());
    std::string line;
    int idx = 0;
    while (!rfile.eof()) {
        std::getline(rfile, line);
        if (line.size() == 0) continue;
        gIdToKey->at(idx++) = line;
    }
}

void SaveKeyMap() {
    std::ofstream wfile(FLAGS_km);
    hassert(wfile.good(), "Open %s failed", FLAGS_km.c_str());
    for (size_t i = 0; i < gIdToKey->size(); ++i) {
        wfile << gIdToKey->at(i) << std::endl;
    }
}

std::map<std::string, std::vector<std::pair<float, std::string>>> LoadResult(const char* fn) {
    std::map<std::string, std::vector<std::pair<float, std::string>>> res;
    std::ifstream rfile(fn);
    hassert(rfile.good(), "Open %s failed", fn);
    std::string line, tmp, qid, iid;
    float val;
    while (!rfile.eof()) {
        std::getline(rfile, line);
        if (line.size() == 0) continue;
        std::stringstream ss(line);
        ss >> tmp;
        ss >> qid >> tmp >> iid >> val;
        res[qid].push_back({val, iid});
    }
    for (auto& kv : res) {
        std::sort(kv.second.begin(), kv.second.end());
    }
    return res;
}

int dcmp(float v) {
    const static float eps = 1e-5;
    if (v < -eps) return -1;
    return v > eps ? 1 : 0;
}

void Evaluate(const char* gtfn, const char* fn) {
    auto gtres = LoadResult(gtfn);
    auto res = LoadResult(fn);
    size_t hit_by_label = 0, hit_by_score = 0, tot = 0;
    for (auto kv : gtres) {
        size_t label = 0, score = 0;
        auto& gv_list = kv.second;
        auto& ot_list = res[kv.first];
        tot += gv_list.size();
        
        std::set<std::string> s;
        for (auto& kv : gv_list) s.insert(kv.second);
        for (auto& kv : ot_list) label += (s.count(kv.second) ? 1 : 0);

        size_t a = 0, b = 0;
        while (a < gv_list.size() && b < ot_list.size()) {
            int d = dcmp(gv_list[a].first - ot_list[b].first);
            // printf("%d %zu %zu %.6f %.6f\n", d, a, b, gv_list[a].first, ot_list[b].first);
            if (d == 0) {
                score++;
                a++;
                b++;
            } else if (d < 0) {
                a++;
            } else {
                b++;
            }
        }

        hit_by_label += label;
        hit_by_score += score;
        if (FLAGS_verbose) {
            printf("[EVA] qid %s hit_by_label %.4f hit_by_score %.4f\n",
                    kv.first.c_str(),
                    (float)label / gv_list.size(),
                    (float)score / gv_list.size());
        }
    }
    printf("[EVA] gtfn %s fn %s hit_by_label %.4f hit_by_score %.4f\n", 
            gtfn, fn,
            100. * hit_by_label / tot, 
            100. * hit_by_score / tot);
}

int Run() {
    assert(FLAGS_idx.size() > 0);
    assert(gSpace != nullptr);

    cutils::TimeDiff td;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
    if (Access(FLAGS_idx) == 0 && Access(FLAGS_km) == 0) {
        td.Reset();
        index = cutils::make_unique<hnswlib::HierarchicalNSW<float>>(
                    gSpace.get(), FLAGS_idx.c_str());
        assert(index != nullptr);
        td.Stop();
        printf("Load index cost %d\n", td.ElapsedInSecond());

        td.Reset();
        gIdToKey = cutils::make_unique<std::vector<std::string>>(index->cur_element_count);
        ReadKeyMap();
        td.Stop();
        printf("Load keymap cost %d\n", td.ElapsedInSecond());
    } else {
        td.Reset();
        std::vector<std::string> labels;
        std::vector<float> feats;
        LoadData({FLAGS_input}, labels, feats);
        td.Stop();
        printf("LoadData %s cost %d\n", 
                FLAGS_input.c_str(), td.ElapsedInSecond());
        printf("labels %zu feats %zu\n", labels.size(), feats.size());

        if (labels.size() == 0) {
            printf("Input is null. %s\n", FLAGS_input.c_str());
            return -1;
        }

        float ip2cos_fac = 1.;
        if (FLAGS_ip2cos) {
            td.Reset();
            double max_len = 0;
            for (size_t i = 0; i < labels.size(); ++i) {
                float* v = &(feats[i * FLAGS_dim]);
                double s = 0;
                for (int k = 0; k < FLAGS_dim; ++k) {
                    s += v[k] * v[k];
                }
                max_len = std::max(max_len, s);
            }
            if (max_len > 1.) {
                ip2cos_fac = sqrt(max_len * 1.01);
                for (size_t i = 0; i < labels.size(); ++i) {
                    float* v = &(feats[i * FLAGS_dim]);
                    for (int k = 0; k < FLAGS_dim; ++k) {
                        v[k] /= ip2cos_fac;
                    }
                }
            }

            std::vector<float> new_feats((FLAGS_dim + 1) * labels.size());
            size_t e = 0;
            for (size_t i = 0; i < labels.size(); ++i) {
                float* v = &(feats[i * FLAGS_dim]);
                double s = 0;
                for (int k = 0; k < FLAGS_dim; ++k) {
                    s += v[k] * v[k];
                    new_feats[e++] = v[k];
                }
                assert(s <= 1.);
                new_feats[e++] = sqrt(std::max(0., 1. - s));
            }
            assert(e == (FLAGS_dim + 1) * labels.size());
            feats.swap(new_feats);
            FLAGS_dim += 1;
            gSpace = cutils::make_unique<hnswlib::InnerProductSpace>(FLAGS_dim);
            td.Stop();
            printf("ip2cos phase_1 cost %d ip2cos_fac %.6f\n", 
                    td.ElapsedInSecond(), ip2cos_fac);
        }

        td.Reset();
        gIdToKey = cutils::make_unique<std::vector<std::string>>(labels.size());
        index = cutils::make_unique<hnswlib::HierarchicalNSW<float>>(
                gSpace.get(), labels.size(), FLAGS_m, FLAGS_efc);
        std::vector<cutils::AsyncWorkerPtr> builders(FLAGS_tr);
        std::atomic<int> idx_alloc(0);
        for (int i = 0; i < FLAGS_tr; ++i) {
            builders[i] = cutils::AsyncWorker::Make(
                    &Builder, index.get(), std::ref(labels), 
                    std::ref(feats), std::ref(idx_alloc));
        }
        builders.clear();
        td.Stop();
        printf("Build index cost %d\n", td.ElapsedInSecond());

        if (FLAGS_ip2cos) {
            td.Reset();
            FLAGS_dim -= 1;
            // index will user space in query, can not desturct it
            // gSpace = cutils::make_unique<hnswlib::InnerProductSpace>(FLAGS_dim);
            index->DropLastDimension();
            for (size_t i = 0; i < labels.size(); ++i) {
                float* v = (float*)index->getDataByInternalId(i);
                for (int k = 0; k < FLAGS_dim; ++k) {
                    v[k] *= ip2cos_fac;
                }
            }
            td.Stop();
            printf("ip2cos phase_2 cost %d\n", td.ElapsedInSecond());
        }

        td.Reset();
        index->saveIndex(FLAGS_idx);
        td.Stop();
        printf("Save index cost %d s\n", td.ElapsedInSecond());

        td.Reset();
        SaveKeyMap();
        td.Stop();
        printf("Save keymap cost %d s\n", td.ElapsedInSecond());
    }
    assert(index != nullptr);
    printf("Index info: dim %zu cur_count %zu max_count %zu M %zu efc %zu\n",
            index->data_size_ / sizeof(float),
            index->cur_element_count, index->max_elements_,
            index->M_, index->ef_construction_);
    assert((size_t)FLAGS_dim == index->data_size_ / sizeof(float));

    std::string resfn;
    if (FLAGS_qry.size() > 0 && Access(FLAGS_qry) == 0) {
        td.Reset();
        std::vector<std::string> labels;
        std::vector<float> feats;
        LoadData({FLAGS_qry}, labels, feats);
        td.Stop();
        printf("LoadData %s cost %d\n", 
                FLAGS_qry.c_str(), td.ElapsedInSecond());
        printf("labels %zu feats %zu\n", labels.size(), feats.size());

        char fn[1024], line[1024];
        sprintf(fn, "%s_%zu_%zu_%d", FLAGS_res.c_str(), 
                index->M_, index->ef_construction_, FLAGS_efs);
        std::ofstream wfile(fn);
        hassert(wfile.good(), "open %s failed", fn);
        size_t tot_cost = 0;
        for (size_t i = 0; i < labels.size(); ++i) {
            td.Reset();
            std::vector<std::pair<float, uint32_t>> res;
            index->MMSimSearch(feats.data() + FLAGS_dim * i,
                    FLAGS_topk, FLAGS_efs, res);
            td.Stop();
            tot_cost += td.ElapsedInMillisecond();

            for (size_t j = 0; j < res.size(); ++j) {
                auto& kv = res[j];
                if (FLAGS_sp == "ip") {
                    kv.first = 1. - kv.first;
                }
                sprintf(line, "qid %s sim %s %.8f\n", 
                        labels[i].c_str(),
                        gIdToKey->at(kv.second).c_str(),
                        kv.first);
                wfile << line;
            }
        }

        resfn = fn;
        printf("[COST] Query %s avg cost %.2f ms\n", fn, (float)tot_cost / labels.size());
    }

    if (FLAGS_gt.size() > 0 && resfn.size() > 0) {
        Evaluate(FLAGS_gt.c_str(), resfn.c_str());
    }

    if (FLAGS_ip2cos) {
        // for autotune
        gSpace = cutils::make_unique<hnswlib::InnerProductSpace>(FLAGS_dim);
    }

    return 0;
}

void BruteForceWorker(
        const std::vector<float>& feats,
        const std::vector<float>& qry_feats,
        std::atomic<int>& idx_alloc,
        std::vector<std::vector<std::pair<float, uint32_t>>>& ans,
        bool& stop) {
    assert(feats.size() % FLAGS_dim == 0);
    assert(qry_feats.size() % FLAGS_dim == 0);
    while (true) {
        int id = (idx_alloc++);
        if ((size_t)id >= qry_feats.size() / FLAGS_dim) break;

        const float* query = qry_feats.data() + FLAGS_dim * id;
        int n = feats.size() / FLAGS_dim;
        std::priority_queue<std::pair<float, uint32_t>> pq;
        for (int i = 0; i < n; ++i) {
            float v = (gSpace->get_dist_func())(
                    query, feats.data() + FLAGS_dim * i, (void*)&FLAGS_dim);
            if (pq.size() < (size_t)FLAGS_topk) {
                pq.push({v, (uint32_t)i});
            } else if (v < pq.top().first) {
                pq.pop();
                pq.push({v, (uint32_t)i});
            }
        }
        auto& res = ans[id];
        res.resize(pq.size());
        for (size_t i = 0; i < res.size(); ++i) {
            res[res.size() - 1 - i] = pq.top();
            pq.pop();
        }
    }
}

void BruteForce() {
    cutils::TimeDiff td;

    std::vector<std::string> labels;
    std::vector<float> feats;
    {
        td.Reset();
        LoadData({FLAGS_input}, labels, feats);
        td.Stop();
        printf("LoadData %s cost %d\n", 
                FLAGS_input.c_str(), td.ElapsedInSecond());
        printf("labels %zu feats %zu\n", labels.size(), feats.size());
    }
    
    std::vector<std::string> qry_labels;
    std::vector<float> qry_feats;
    {
        td.Reset();
        LoadData({FLAGS_qry}, qry_labels, qry_feats);
        td.Stop();
        printf("LoadData %s cost %d\n", 
                FLAGS_qry.c_str(), td.ElapsedInSecond());
        printf("labels %zu feats %zu\n", qry_labels.size(), qry_feats.size());
    }

    std::atomic<int> idx_alloc(0);
    std::vector<std::vector<std::pair<float, uint32_t>>> ans(qry_labels.size());
    std::vector<cutils::AsyncWorkerPtr> workers(FLAGS_tr);
    for (int i = 0; i < FLAGS_tr; ++i) {
        workers[i] = cutils::AsyncWorker::Make(&BruteForceWorker,
                std::cref(feats), std::cref(qry_feats), std::ref(idx_alloc),
                std::ref(ans));
    }
    workers.clear();

    char line[1024];
    std::ofstream wfile(FLAGS_res);
    hassert(wfile.good(), "open %s failed", FLAGS_res.c_str());
    for (size_t i = 0; i < qry_labels.size(); ++i) {
        auto& res = ans[i];
        for (size_t j = 0; j < res.size(); ++j) {
            auto& kv = res[j];
            if (FLAGS_sp == "ip") {
                kv.first = 1. - kv.first;
            }
            sprintf(line, "qid %s sim %s %.8f\n", 
                    qry_labels[i].c_str(),
                    labels[kv.second].c_str(),
                    kv.first);
            wfile << line;
        }
    }
}

std::vector<int> SplitStr(std::string str) {
    std::vector<int> vec;
    for (auto& c : str) if (c == ',') c = ' ';
    std::stringstream ss(str);
    int x;
    while (ss >> x) vec.push_back(x);
    return vec;
}

void AutoTune() {
    std::vector<int> ms = SplitStr(FLAGS_atm);
    std::vector<int> efcs = SplitStr(FLAGS_atefc);
    std::string src_idx = FLAGS_idx;
    std::string src_km = FLAGS_km;
    for (auto m : ms) {
        FLAGS_m = m;
        for (auto efc : efcs) {
            FLAGS_efc = efc;
            for (int i = 2; i <= 6; ++i) {
                FLAGS_efs = FLAGS_efc * i / 4;
                char buf[1024];
                sprintf(buf, "%s_%d_%d", src_idx.c_str(), m, efc);
                FLAGS_idx = buf;
                sprintf(buf, "%s_%d_%d", src_km.c_str(), m, efc);
                FLAGS_km = buf;
                Run();
            }
        }
    }
}

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);

    std::string buf;
    buf.append("\n")
        .append(" -fun run ...\n")
        .append(" -fun bruteforce ...\n")
        .append(" -fun autotune ...\n")
        .append(" -fun evaluate ...\n");
    google::SetUsageMessage(buf);
    google::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_fun == "run") {
        InitSpace();
        Run();
    } else if (FLAGS_fun == "bruteforce") {
        InitSpace();
        BruteForce();
    } else if (FLAGS_fun == "autotune") {
        InitSpace();
        AutoTune();
    } else if (FLAGS_fun == "evaluate") {
        Evaluate(FLAGS_gt.c_str(), FLAGS_res.c_str());
    } else {
        google::ShowUsageWithFlagsRestrict(argv[0], "hnsw_tools");
    }

    return 0;
}

//gzrd_Lib_CPP_Version_ID--start
#ifndef GZRD_SVN_ATTR
#define GZRD_SVN_ATTR "0"
#endif
static char gzrd_Lib_CPP_Version_ID[] __attribute__((used))="$HeadURL: http://scm-gy.tencent.com/gzrd/gzrd_mail_rep/QQMailcore_proj/trunk/platform/simsearch/mmsimsvr/tools/hnsw_tools.cpp $ $Id: hnsw_tools.cpp 2852609 2018-11-29 11:44:58Z flashlin $ " GZRD_SVN_ATTR "__file__";
// gzrd_Lib_CPP_Version_ID--end

