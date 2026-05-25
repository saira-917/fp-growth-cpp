/*
FP-Growth Algorithm Implementation
Author: Saira Chowdhury

Implements frequent itemset mining using a tree-based FP-Growth approach.
Optimized for improved runtime and memory efficiency on large datasets.
*/

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cctype>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

struct Args {
    string infile; string outfile = "";
    long long minsup = -1; double minprob = -1;
    bool verbose = true; bool keep_singletons = true;
};

static void usage() {
    cerr << "FP-Growth � Frequent Itemset Mining\n"
        << "Options:\n"
        << "  -i <file>    input transactions file\n"
        << "  -s <minsup>  minimum support (absolute integer)\n"
        << "  -p <frac>    minimum support fraction (0�1]\n"
        << "  -o <file>    write itemsets to this file (default: stdout)\n"
        << "  --no-single  skip single-item sets\n"
        << "  -q           quiet\n";
}

// -------- memory usage (noop on Windows) --------
#ifdef _WIN32
static size_t get_rss_bytes() { return 0; }
#else
#include <unistd.h>
static size_t get_rss_bytes() {
    long rss_pages = 0; ifstream f("/proc/self/statm"); long x;
    if (f) { f >> x >> rss_pages; }
    return (size_t)rss_pages * (size_t)sysconf(_SC_PAGESIZE);
}
#endif

// ------------------- Load database -------------------
static vector<vector<int>> load_db(const string& path,
    vector<string>& id2item,
    unordered_map<string, int>& item2id,
    size_t& n_items) {
    ifstream in(path);
    if (!in) throw runtime_error("Cannot open input file: " + path);
    vector<vector<int>> db;
    auto get_or_add = [&](const string& tok) {
        auto it = item2id.find(tok);
        if (it != item2id.end()) return it->second;
        int id = (int)id2item.size();
        item2id[tok] = id; id2item.push_back(tok); return id;
        };
    string line;
    while (getline(in, line)) {
        vector<int> basket; basket.reserve(32);
        string cur;
        for (size_t i = 0; i <= line.size(); ++i) {
            char c = (i < line.size() ? line[i] : ' ');
            if (isspace(static_cast<unsigned char>(c)) || c == ',' || c == '\t') {
                if (!cur.empty()) { basket.push_back(get_or_add(cur)); cur.clear(); }
            }
            else cur.push_back(c);
        }
        sort(basket.begin(), basket.end());
        basket.erase(unique(basket.begin(), basket.end()), basket.end());
        if (!basket.empty()) db.push_back(move(basket));
    }
    n_items = id2item.size();
    return db;
}

// ------------------- FP-tree -------------------
struct FPNode {
    int item = -1; int cnt = 0;
    FPNode* parent = nullptr;
    unordered_map<int, FPNode*> children;
    FPNode* next = nullptr;
};

struct HeaderEntry {
    int item; int freq; FPNode* head = nullptr;
};

struct FPTree {
    FPNode* root;
    unordered_map<int, HeaderEntry> header;
    vector<HeaderEntry*> order_desc;
    FPTree() { root = new FPNode(); }
    ~FPTree() { free_node(root); }
    void free_node(FPNode* n) {
        if (!n) return;
        for (auto& kv : n->children) free_node(kv.second);
        delete n;
    }
};

static unique_ptr<FPTree> build_fptree(const vector<vector<int>>& db,
    const unordered_map<int, int>& global_freq,
    const vector<int>& order_items,
    long long minsup) {
    auto tree = make_unique<FPTree>();
    for (int it : order_items)
        tree->header[it] = HeaderEntry{ it, global_freq.at(it), nullptr };
    tree->order_desc.reserve(order_items.size());
    for (int it : order_items) tree->order_desc.push_back(&tree->header[it]);

    for (const auto& trans_raw : db) {
        vector<int> trans;
        for (int it : trans_raw) {
            auto g = global_freq.find(it);
            if (g != global_freq.end() && g->second >= minsup) trans.push_back(it);
        }
        if (trans.empty()) continue;
        sort(trans.begin(), trans.end(), [&](int a, int b) {
            int fa = global_freq.at(a), fb = global_freq.at(b);
            if (fa != fb) return fa > fb;
            return a < b;
            });
        FPNode* cur = tree->root;
        for (int it : trans) {
            FPNode* child;
            auto itc = cur->children.find(it);
            if (itc == cur->children.end()) {
                child = new FPNode();
                child->item = it; child->cnt = 1; child->parent = cur;
                cur->children[it] = child;
                auto& he = tree->header[it];
                child->next = he.head; he.head = child;
            }
            else { child = itc->second; child->cnt += 1; }
            cur = child;
        }
    }
    return tree;
}

static void mine_tree(const FPTree& tree, long long minsup,
    vector<int>& suffix, const vector<string>& id2item,
    const function<void(const vector<int>&, int)>& emit) {
    vector<pair<int, int>> items;
    for (auto he_ptr : tree.order_desc)
        items.emplace_back(he_ptr->item, he_ptr->freq);
    sort(items.begin(), items.end(), [&](auto& a, auto& b) {
        if (a.second != b.second) return a.second < b.second;
        return a.first < b.first;
        });

    for (auto& pr : items) {
        int item = pr.first, freq = pr.second;
        if (freq < minsup) continue;
        vector<int> newset = suffix; newset.push_back(item);
        emit(newset, freq);

        vector<vector<pair<int, int>>> cond_paths;
        for (FPNode* n = tree.header.at(item).head; n; n = n->next) {
            int count = n->cnt;
            vector<pair<int, int>> path;
            FPNode* p = n->parent;
            while (p && p->parent) {
                path.emplace_back(p->item, count);
                p = p->parent;
            }
            if (!path.empty()) cond_paths.push_back(move(path));
        }
        if (cond_paths.empty()) continue;

        unordered_map<int, int> cond_freq;
        for (const auto& path : cond_paths)
            for (auto& pp : path) cond_freq[pp.first] += pp.second;

        vector<int> cond_items;
        for (auto& kv : cond_freq)
            if (kv.second >= minsup) cond_items.push_back(kv.first);
        if (cond_items.empty()) continue;
        sort(cond_items.begin(), cond_items.end(), [&](int a, int b) {
            if (cond_freq.at(a) != cond_freq.at(b)) return cond_freq.at(a) > cond_freq.at(b);
            return a < b;
            });

        FPTree cond_tree;
        for (int it : cond_items) {
            cond_tree.header[it] = HeaderEntry{ it, cond_freq[it], nullptr };
            cond_tree.order_desc.push_back(&cond_tree.header[it]);
        }
        cond_tree.root = new FPNode();

        for (const auto& path : cond_paths) {
            vector<int> items_only; int count = 0;
            for (auto& pp : path) {
                int it = pp.first, c = pp.second;
                if (cond_freq[it] >= minsup) { items_only.push_back(it); count = c; }
            }
            if (items_only.empty()) continue;
            sort(items_only.begin(), items_only.end(), [&](int a, int b) {
                if (cond_freq.at(a) != cond_freq.at(b)) return cond_freq.at(a) > cond_freq.at(b);
                return a < b;
                });
            FPNode* cur = cond_tree.root;
            for (int it : items_only) {
                auto itc = cur->children.find(it);
                if (itc == cur->children.end()) {
                    FPNode* child = new FPNode();
                    child->item = it; child->cnt = count; child->parent = cur;
                    cur->children[it] = child;
                    auto& he = cond_tree.header[it];
                    child->next = he.head; he.head = child;
                    cur = child;
                }
                else {
                    itc->second->cnt += count;
                    cur = itc->second;
                }
            }
        }
        mine_tree(cond_tree, minsup, newset, id2item, emit);
    }
}

// ------------------- main -------------------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    Args args;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "-i" && i + 1 < argc) args.infile = argv[++i];
        else if (a == "-o" && i + 1 < argc) args.outfile = argv[++i];
        else if (a == "-s" && i + 1 < argc) args.minsup = stoll(argv[++i]);
        else if (a == "-p" && i + 1 < argc) args.minprob = stod(argv[++i]);
        else if (a == "-q") args.verbose = false;
        else if (a == "--no-single") args.keep_singletons = false;
        else { usage(); return 1; }
    }
    if (args.infile.empty()) { usage(); return 1; }

    auto t0 = chrono::high_resolution_clock::now();

    vector<string> id2item;
    unordered_map<string, int> item2id;
    size_t n_items = 0;
    auto db = load_db(args.infile, id2item, item2id, n_items);
    const size_t N = db.size();
    if (N == 0) { cerr << "Empty DB\n"; return 1; }

    if (args.minsup < 0) {
        if (args.minprob > 0) args.minsup = (long long)ceil(args.minprob * (double)N);
        else { cerr << "ERROR: set -s or -p\n"; return 1; }
    }

    unordered_map<int, int> freq;
    for (const auto& t : db) for (int it : t) ++freq[it];

    vector<int> frequent_items;
    for (auto& kv : freq) if (kv.second >= args.minsup) frequent_items.push_back(kv.first);
    if (frequent_items.empty()) { cerr << "No frequent items\n"; return 0; }

    auto tree = build_fptree(db, freq, frequent_items, args.minsup);

    vector<pair<vector<int>, int>> results;
    vector<int> suffix;
    auto emit = [&](const vector<int>& itemset, int sup) {
        if (!args.keep_singletons && itemset.size() == 1) return;
        results.emplace_back(itemset, sup);
        };
    mine_tree(*tree, args.minsup, suffix, id2item, emit);

    sort(results.begin(), results.end(), [&](auto& A, auto& B) {
        if (A.first.size() != B.first.size()) return A.first.size() > B.first.size();
        if (A.second != B.second) return A.second > B.second;
        vector<string> as, bs;
        for (int id : A.first) as.push_back(id2item[id]);
        for (int id : B.first) bs.push_back(id2item[id]);
        return as < bs;
        });

    ostream* out = &cout; ofstream fout;
    if (!args.outfile.empty()) { fout.open(args.outfile); out = &fout; }
    for (auto& r : results) {
        (*out) << r.second << '\t';
        vector<string> names;
        for (int id : r.first) names.push_back(id2item[id]);
        sort(names.begin(), names.end());
        for (size_t i = 0; i < names.size(); ++i) {
            if (i) (*out) << ' ';
            (*out) << names[i];
        }
        (*out) << '\n';
    }

    auto t1 = chrono::high_resolution_clock::now();
    double secs = chrono::duration<double>(t1 - t0).count();

    if (args.verbose) {
        cerr << fixed << setprecision(3);
        cerr << "\n==== SUMMARY ====\n";
        cerr << "transactions: " << N << "\n";
        cerr << "items:        " << id2item.size()
            << " (frequent: " << tree->header.size() << ")\n";
        cerr << "minsup:       " << args.minsup << "\n";
        size_t rss = get_rss_bytes();
        if (rss) cerr << "RSS memory:   " << (rss / (1024.0 * 1024.0)) << " MB\n";
        cerr << "runtime:      " << secs << " s\n";
    }
    return 0;
}
