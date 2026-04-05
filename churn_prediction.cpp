/**
 * churn_prediction.cpp
 *
 * Customer Churn Prediction — pure C++17 implementation
 * Mirrors the Python/scikit-learn notebook (churn_prediction.ipynb)
 *
 * Models   : Logistic Regression, Decision Tree, Random Forest, XGBoost (from scratch)
 * HPO      : Grid Search, Random Search, Bayesian Optimisation (GP-EI)
 * Imbalance: SMOTE (in-fold, like imblearn.pipeline)
 * CV       : 5-fold Stratified K-Fold
 * Metrics  : Accuracy, Precision, Recall, F1, AUC-ROC
 *
 * Zero external dependencies — all algorithms implemented from scratch in pure C++17.
 *
 * Build (example):
 *   g++ -O2 -std=c++17 churn_prediction.cpp -o churn_prediction
 *
 *   Or with CMakeLists.txt:   cmake -B build && cmake --build build
 *
 * Usage:
 *   ./churn_prediction customer_churn_dataset.csv
 */

// ─────────────────────────────────────────────────────────────────────────────
// Standard headers
// ─────────────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>


// ─────────────────────────────────────────────────────────────────────────────
// Global type aliases
// ─────────────────────────────────────────────────────────────────────────────
using Matrix    = std::vector<std::vector<double>>;
using Vector    = std::vector<double>;
using IVector   = std::vector<int>;
using StrVec    = std::vector<std::string>;
using HPValue   = std::variant<int, double, std::string>;
using HPConfig  = std::map<std::string, HPValue>;
using Objective = std::function<double(const HPConfig&)>;

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 1 — DATA LOADING
// ─────────────────────────────────────────────────────────────────────────────

struct DataFrame {
    StrVec columns;
    std::unordered_map<std::string, Vector>  num;   // numeric columns
    std::unordered_map<std::string, StrVec>  str;   // string/categorical columns
    std::size_t n_rows = 0;

    bool has(const std::string& c) const {
        return num.count(c) || str.count(c);
    }
    const Vector& get_num(const std::string& c) const { return num.at(c); }
    const StrVec& get_str(const std::string& c) const { return str.at(c); }
};

// Split a CSV line respecting quoted fields
static StrVec split_csv_line(const std::string& line) {
    StrVec tokens;
    std::string tok;
    bool in_q = false;
    for (char c : line) {
        if (c == '"') { in_q = !in_q; }
        else if (c == ',' && !in_q) { tokens.push_back(tok); tok.clear(); }
        else tok += c;
    }
    tokens.push_back(tok);
    return tokens;
}

// Try to parse a cell as double (handles "True"/"False" booleans)
static bool try_num(const std::string& s, double& out) {
    if (s.empty()) { out = std::numeric_limits<double>::quiet_NaN(); return true; }
    if (s == "True"  || s == "true")  { out = 1.0; return true; }
    if (s == "False" || s == "false") { out = 0.0; return true; }
    try {
        std::size_t pos;
        out = std::stod(s, &pos);
        return pos == s.size();
    } catch (...) { return false; }
}

DataFrame load_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);

    DataFrame df;
    std::string line;

    // Header
    if (!std::getline(f, line))
        throw std::runtime_error("Empty file: " + path);
    df.columns = split_csv_line(line);
    std::size_t ncols = df.columns.size();

    // Accumulate raw strings
    std::vector<StrVec> raw(ncols);
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto toks = split_csv_line(line);
        toks.resize(ncols, "");
        for (std::size_t j = 0; j < ncols; ++j)
            raw[j].push_back(toks[j]);
        ++df.n_rows;
    }

    // Infer column types
    for (std::size_t j = 0; j < ncols; ++j) {
        const std::string& col = df.columns[j];
        bool all_num = true;
        Vector nums(raw[j].size());
        for (std::size_t i = 0; i < raw[j].size(); ++i)
            if (!try_num(raw[j][i], nums[i])) { all_num = false; break; }
        if (all_num) df.num[col] = std::move(nums);
        else         df.str[col] = std::move(raw[j]);
    }

    std::cout << "Loaded " << df.n_rows << " rows × " << ncols << " cols\n";
    return df;
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 2 — FEATURE ENGINEERING
// ─────────────────────────────────────────────────────────────────────────────

// Parse "YYYY-MM-DD" → seconds since epoch (as double)
static double date_to_sec(const std::string& s) {
    std::tm tm{};
    std::istringstream ss(s);
    ss >> std::get_time(&tm, "%Y-%m-%d");
    if (ss.fail()) return 0.0;
    tm.tm_isdst = -1;
    return static_cast<double>(std::mktime(&tm));
}

static double date_col_to_days(const DataFrame& df, const std::string& col, std::size_t i) {
    if (df.str.count(col)) return date_to_sec(df.str.at(col)[i]) / 86400.0;
    if (df.num.count(col)) return df.num.at(col)[i];
    return 0.0;
}

void engineer_features(DataFrame& df) {
    std::size_t n = df.n_rows;

    // Reference date = max of Transaction_Date
    double ref_day = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        ref_day = std::max(ref_day, date_col_to_days(df, "Transaction_Date", i));
    if (ref_day == 0.0) ref_day = date_to_sec("2025-12-31") / 86400.0;

    const auto& balance      = df.get_num("Account_Balance");
    const auto& txn_amt      = df.get_num("Transaction_Amount");
    const auto& loan_amt     = df.get_num("Loan_Amount");
    const auto& income       = df.get_num("Annual_Income");
    const auto& n_txn        = df.get_num("Number_of_Transactions");
    const auto& cs_inter     = df.get_num("Customer_Service_Interactions");
    const auto& complaints   = df.get_num("Recent_Complaints");
    const auto& chg_balance  = df.get_num("Change_in_Account_Balance");
    const auto& satisfaction = df.get_num("Customer_Satisfaction_Score");

    Vector tenure(n), inactivity(n), txn_freq(n), avg_txn(n),
           bal_vol(n), loan_income(n), cmp_ratio(n), sat_gap(n), engagement(n);

    for (std::size_t i = 0; i < n; ++i) {
        double open_day  = date_col_to_days(df, "Account_Open_Date",    i);
        double last_day  = date_col_to_days(df, "Last_Transaction_Date", i);

        // a. Tenure / inactivity
        tenure[i]     = std::max(0.0, (ref_day - open_day) / 30.44);
        inactivity[i] = std::max(0.0,  ref_day - last_day);

        // b. Transaction
        txn_freq[i] = n_txn[i] / (tenure[i] + 1.0);
        avg_txn[i]  = txn_amt[i] / std::max(1.0, n_txn[i]);
        bal_vol[i]  = std::abs(chg_balance[i]) / (std::abs(balance[i]) + 1e-6);

        // c. Financial health
        loan_income[i] = loan_amt[i] / (income[i] + 1e-6);

        // d. Service quality
        cmp_ratio[i] = complaints[i] / (cs_inter[i] + 1.0);
        sat_gap[i]   = (6.0 - satisfaction[i]) * (complaints[i] + 1.0);

        // e. Composite engagement
        engagement[i] = txn_freq[i] * satisfaction[i] / (cmp_ratio[i] + 1.0);
    }

    // Clip Balance_Volatility_Index at 99th percentile (computed on full data)
    Vector sorted_vol = bal_vol;
    std::sort(sorted_vol.begin(), sorted_vol.end());
    double vol_cap = sorted_vol[static_cast<std::size_t>(0.99 * sorted_vol.size())];
    for (auto& v : bal_vol) v = std::min(v, vol_cap);

    df.num["Customer_Tenure_Months"]          = tenure;
    df.num["Inactivity_Period_Days"]           = inactivity;
    df.num["Transaction_Frequency"]            = txn_freq;
    df.num["Avg_Transaction_Size"]             = avg_txn;
    df.num["Balance_Volatility_Index"]         = bal_vol;
    df.num["Loan_to_Income_Ratio"]             = loan_income;
    df.num["Complaint_to_Interaction_Ratio"]   = cmp_ratio;
    df.num["Satisfaction_Gap"]                 = sat_gap;
    df.num["Engagement_Index"]                 = engagement;

    std::cout << "Engineered 9 features (reference day = " << ref_day << ")\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 3 — PREPROCESSING
// ─────────────────────────────────────────────────────────────────────────────

// 3.1  Standard Scaler
class StandardScaler {
public:
    Vector mean_, std_;

    Matrix fit_transform(const Matrix& X) {
        std::size_t n = X.size(), p = X[0].size();
        mean_.assign(p, 0.0); std_.assign(p, 0.0);
        for (const auto& row : X)
            for (std::size_t j = 0; j < p; ++j) mean_[j] += row[j];
        for (auto& m : mean_) m /= n;
        for (const auto& row : X)
            for (std::size_t j = 0; j < p; ++j) {
                double d = row[j] - mean_[j]; std_[j] += d * d;
            }
        for (std::size_t j = 0; j < p; ++j) {
            std_[j] = std::sqrt(std_[j] / n);
            if (std_[j] < 1e-10) std_[j] = 1.0;
        }
        return transform(X);
    }

    Matrix transform(const Matrix& X) const {
        Matrix out = X;
        for (auto& row : out)
            for (std::size_t j = 0; j < row.size(); ++j)
                row[j] = (row[j] - mean_[j]) / std_[j];
        return out;
    }
};

// 3.2  One-Hot Encoder  (drop='first' like sklearn)
class OneHotEncoder {
public:
    std::vector<StrVec> categories_;  // per feature, sorted, first dropped

    void fit(const std::vector<StrVec>& cat) {
        categories_.clear();
        for (const auto& col : cat) {
            std::set<std::string> u(col.begin(), col.end());
            StrVec sv(u.begin(), u.end());
            if (sv.size() > 1) sv.erase(sv.begin());  // drop first
            categories_.push_back(sv);
        }
    }

    Matrix transform(const std::vector<StrVec>& cat) const {
        std::size_t n = cat[0].size();
        std::size_t total = 0;
        for (const auto& c : categories_) total += c.size();

        Matrix out(n, Vector(total, 0.0));
        std::size_t off = 0;
        for (std::size_t f = 0; f < cat.size(); ++f) {
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t k = 0; k < categories_[f].size(); ++k)
                    if (cat[f][i] == categories_[f][k]) { out[i][off + k] = 1.0; break; }
            off += categories_[f].size();
        }
        return out;
    }
};

// Horizontal concatenation of two matrices (same row count)
static Matrix hconcat(const Matrix& A, const Matrix& B) {
    Matrix out(A.size());
    for (std::size_t i = 0; i < A.size(); ++i) {
        out[i] = A[i];
        out[i].insert(out[i].end(), B[i].begin(), B[i].end());
    }
    return out;
}

// 3.3  SMOTE (Synthetic Minority Over-sampling Technique)
class SMOTE {
    int k_;
    unsigned seed_;
public:
    SMOTE(int k = 5, unsigned seed = 42) : k_(k), seed_(seed) {}

    std::pair<Matrix, IVector> fit_resample(const Matrix& X, const IVector& y) {
        std::mt19937 rng(seed_);

        Matrix minority, majority;
        for (std::size_t i = 0; i < X.size(); ++i)
            (y[i] == 1 ? minority : majority).push_back(X[i]);

        int to_gen = (int)majority.size() - (int)minority.size();
        if (to_gen <= 0) return {X, y};

        std::size_t p = X[0].size();
        Matrix synthetic;
        synthetic.reserve(to_gen);

        std::uniform_int_distribution<int> ridx(0, (int)minority.size() - 1);
        std::uniform_real_distribution<double> alpha(0.0, 1.0);

        while ((int)synthetic.size() < to_gen) {
            int qi = ridx(rng);

            // k-NN in minority class
            std::vector<std::pair<double,int>> dists;
            dists.reserve(minority.size());
            for (int j = 0; j < (int)minority.size(); ++j) {
                if (j == qi) continue;
                double d = 0.0;
                for (std::size_t f = 0; f < p; ++f) {
                    double df_ = minority[qi][f] - minority[j][f];
                    d += df_ * df_;
                }
                dists.push_back({std::sqrt(d), j});
            }
            int k = std::min(k_, (int)dists.size());
            std::partial_sort(dists.begin(), dists.begin() + k, dists.end());

            std::uniform_int_distribution<int> kdist(0, k - 1);
            int ni = dists[kdist(rng)].second;

            Vector s = minority[qi];
            double a = alpha(rng);
            for (std::size_t f = 0; f < p; ++f)
                s[f] += a * (minority[ni][f] - s[f]);
            synthetic.push_back(s);
        }

        Matrix X_res = X;
        IVector y_res = y;
        for (const auto& s : synthetic) { X_res.push_back(s); y_res.push_back(1); }

        // Shuffle
        std::vector<std::size_t> perm(X_res.size());
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        Matrix X_out(X_res.size()); IVector y_out(y_res.size());
        for (std::size_t i = 0; i < perm.size(); ++i) {
            X_out[i] = X_res[perm[i]]; y_out[i] = y_res[perm[i]];
        }
        return {X_out, y_out};
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 4 — HELPERS (extract, split)
// ─────────────────────────────────────────────────────────────────────────────

struct ExtractedData {
    Matrix             X_num;
    std::vector<StrVec> X_cat;
    IVector            y;
};

ExtractedData extract(const DataFrame& df,
                      const StrVec& num_cols,
                      const StrVec& cat_cols,
                      const std::string& target) {
    ExtractedData e;
    std::size_t n = df.n_rows;

    // target
    const auto& yr = df.get_num(target);
    e.y.resize(n);
    for (std::size_t i = 0; i < n; ++i) e.y[i] = (int)yr[i];

    // numeric
    e.X_num.assign(n, Vector(num_cols.size(), 0.0));
    for (std::size_t j = 0; j < num_cols.size(); ++j) {
        const auto& col = df.get_num(num_cols[j]);
        for (std::size_t i = 0; i < n; ++i)
            e.X_num[i][j] = std::isnan(col[i]) ? 0.0 : col[i];
    }

    // categorical
    e.X_cat.resize(cat_cols.size());
    for (std::size_t j = 0; j < cat_cols.size(); ++j)
        e.X_cat[j] = df.get_str(cat_cols[j]);

    return e;
}

struct DataSplit {
    Matrix X_num_tr, X_num_te;
    std::vector<StrVec> X_cat_tr, X_cat_te;
    IVector y_tr, y_te;
};

// Stratified 80/20 split by explicit index lists
DataSplit make_split(const ExtractedData& e,
                     const std::vector<std::size_t>& tr_idx,
                     const std::vector<std::size_t>& te_idx) {
    DataSplit s;
    s.X_cat_tr.resize(e.X_cat.size());
    s.X_cat_te.resize(e.X_cat.size());

    for (std::size_t i : tr_idx) {
        s.X_num_tr.push_back(e.X_num[i]);
        s.y_tr.push_back(e.y[i]);
        for (std::size_t j = 0; j < e.X_cat.size(); ++j)
            s.X_cat_tr[j].push_back(e.X_cat[j][i]);
    }
    for (std::size_t i : te_idx) {
        s.X_num_te.push_back(e.X_num[i]);
        s.y_te.push_back(e.y[i]);
        for (std::size_t j = 0; j < e.X_cat.size(); ++j)
            s.X_cat_te[j].push_back(e.X_cat[j][i]);
    }
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 5 — METRICS
// ─────────────────────────────────────────────────────────────────────────────

struct Metrics {
    double accuracy = 0, precision = 0, recall = 0, f1 = 0, auc_roc = 0;
    double train_us = 0, pred_us = 0;

    void print(const std::string& lbl = "") const {
        if (!lbl.empty()) std::cout << "  [" << lbl << "]\n";
        std::cout << std::fixed << std::setprecision(4)
            << "    Accuracy  : " << accuracy  << "\n"
            << "    Precision : " << precision << "\n"
            << "    Recall    : " << recall    << "\n"
            << "    F1        : " << f1        << "\n"
            << "    AUC-ROC   : " << auc_roc   << "\n";
    }
};

static double auc_roc(const IVector& yt, const Vector& yp) {
    std::size_t n = yt.size();
    std::vector<std::pair<double,int>> s(n);
    for (std::size_t i = 0; i < n; ++i) s[i] = {yp[i], yt[i]};
    std::sort(s.rbegin(), s.rend());

    double pos = std::count(yt.begin(), yt.end(), 1);
    double neg = n - pos;
    if (pos == 0 || neg == 0) return 0.5;

    double auc = 0, fp = 0, tp = 0, pfp = 0, ptp = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (s[i].second == 1) { tp++; }
        else {
            if (tp != ptp) {
                auc += (fp - pfp) * (tp + ptp) / 2.0;
                pfp = fp; ptp = tp;
            }
            fp++;
        }
    }
    auc += (fp - pfp) * (tp + ptp) / 2.0;
    return auc / (pos * neg);
}

static Metrics compute_metrics(const IVector& yt, const IVector& yp, const Vector& ypr) {
    int tp = 0, fp = 0, fn = 0, tn = 0;
    for (std::size_t i = 0; i < yt.size(); ++i) {
        if (yp[i]==1 && yt[i]==1) tp++;
        else if (yp[i]==1 && yt[i]==0) fp++;
        else if (yp[i]==0 && yt[i]==1) fn++;
        else tn++;
    }
    Metrics m;
    m.accuracy  = (double)(tp+tn) / yt.size();
    m.precision = (tp+fp) ? (double)tp/(tp+fp) : 0.0;
    m.recall    = (tp+fn) ? (double)tp/(tp+fn) : 0.0;
    m.f1        = (m.precision+m.recall) > 0
                  ? 2.0*m.precision*m.recall/(m.precision+m.recall) : 0.0;
    m.auc_roc   = auc_roc(yt, ypr);
    return m;
}


// ─────────────────────────────────────────────────────────────────────────────
// SECTION 6 — ML MODELS
// ─────────────────────────────────────────────────────────────────────────────

class Classifier {
public:
    virtual ~Classifier() = default;
    virtual void fit(const Matrix& X, const IVector& y) = 0;
    virtual IVector predict(const Matrix& X) const = 0;
    virtual Vector  predict_proba(const Matrix& X) const = 0;
    virtual std::string name() const = 0;
};

// ── 6.1  Logistic Regression (gradient descent, L2) ─────────────────────────
class LogisticRegression : public Classifier {
    double C_; int max_iter_; double lr_;
    Vector w_; double b_ = 0.0;

    static double sigmoid(double x) { return 1.0/(1.0+std::exp(-x)); }
public:
    explicit LogisticRegression(double C=1.0, int max_iter=1000, double lr=0.1)
        : C_(C), max_iter_(max_iter), lr_(lr) {}

    void fit(const Matrix& X, const IVector& y) override {
        std::size_t n = X.size(), p = X[0].size();
        w_.assign(p, 0.0); b_ = 0.0;
        double lambda = 1.0 / (C_ * (double)n);

        for (int it = 0; it < max_iter_; ++it) {
            Vector gw(p, 0.0); double gb = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                double z = b_;
                for (std::size_t j = 0; j < p; ++j) z += w_[j]*X[i][j];
                double e = sigmoid(z) - y[i];
                for (std::size_t j = 0; j < p; ++j) gw[j] += e*X[i][j];
                gb += e;
            }
            for (std::size_t j = 0; j < p; ++j)
                w_[j] -= lr_*(gw[j]/n + lambda*w_[j]);
            b_ -= lr_*gb/n;
        }
    }

    Vector predict_proba(const Matrix& X) const override {
        Vector pr(X.size());
        for (std::size_t i = 0; i < X.size(); ++i) {
            double z = b_;
            for (std::size_t j = 0; j < w_.size(); ++j) z += w_[j]*X[i][j];
            pr[i] = sigmoid(z);
        }
        return pr;
    }

    IVector predict(const Matrix& X) const override {
        auto pr = predict_proba(X);
        IVector out(pr.size());
        for (std::size_t i = 0; i < pr.size(); ++i) out[i] = pr[i] >= 0.5 ? 1 : 0;
        return out;
    }

    std::string name() const override { return "LogisticRegression"; }
};

// ── 6.2  Decision Tree (CART, Gini/Entropy) ──────────────────────────────────
struct Node {
    int    feat    = -1;
    double thresh  = 0.0;
    int    left    = -1, right = -1;
    double leaf_p  = 0.0;   // class-1 probability at leaf
    bool   is_leaf = false;
};

class DecisionTree : public Classifier {
    int  max_depth_, min_split_, min_leaf_;
    bool use_entropy_;
    std::vector<Node> nodes_;

    double impurity(const IVector& y) const {
        if (y.empty()) return 0.0;
        double p = (double)std::count(y.begin(),y.end(),1) / y.size();
        if (use_entropy_) {
            if (p==0||p==1) return 0.0;
            return -p*std::log2(p)-(1-p)*std::log2(1-p);
        }
        return 1.0 - p*p - (1-p)*(1-p);
    }

    struct Split { int feat=-1; double thresh=0; double gain=-1e18;
                   std::vector<std::size_t> li, ri; };

    Split best_split(const Matrix& X, const IVector& y,
                     const std::vector<std::size_t>& idx) const {
        Split best;
        IVector ysub; for (std::size_t i:idx) ysub.push_back(y[i]);
        double pimp = impurity(ysub);
        std::size_t p = X[0].size();

        for (std::size_t f = 0; f < p; ++f) {
            std::vector<double> vals; for (std::size_t i:idx) vals.push_back(X[i][f]);
            std::sort(vals.begin(), vals.end());
            vals.erase(std::unique(vals.begin(),vals.end()),vals.end());

            for (std::size_t vi = 0; vi+1 < vals.size(); ++vi) {
                double t = (vals[vi]+vals[vi+1])/2.0;
                std::vector<std::size_t> li, ri;
                for (std::size_t i:idx) (X[i][f]<=t ? li : ri).push_back(i);
                if ((int)li.size()<min_leaf_ || (int)ri.size()<min_leaf_) continue;

                IVector yl, yr;
                for (std::size_t i:li) yl.push_back(y[i]);
                for (std::size_t i:ri) yr.push_back(y[i]);
                double n = (double)idx.size();
                double g = pimp - li.size()/n*impurity(yl) - ri.size()/n*impurity(yr);
                if (g > best.gain) best = {(int)f, t, g, li, ri};
            }
        }
        return best;
    }

    int build(const Matrix& X, const IVector& y,
              const std::vector<std::size_t>& idx, int depth) {
        int ni = (int)nodes_.size();
        nodes_.emplace_back();

        double pos = 0; for (std::size_t i:idx) pos += y[i];
        nodes_[ni].leaf_p = pos / idx.size();

        bool stop = (max_depth_>=0 && depth>=max_depth_)
                 || (int)idx.size()<min_split_
                 || pos==0 || pos==(double)idx.size();
        if (stop) { nodes_[ni].is_leaf = true; return ni; }

        auto sp = best_split(X, y, idx);
        if (sp.feat<0 || sp.li.empty() || sp.ri.empty()) {
            nodes_[ni].is_leaf = true; return ni;
        }
        nodes_[ni].feat   = sp.feat;
        nodes_[ni].thresh = sp.thresh;
        nodes_[ni].left   = build(X, y, sp.li, depth+1);
        nodes_[ni].right  = build(X, y, sp.ri, depth+1);
        return ni;
    }

    double pred_one(const Vector& x, int ni) const {
        if (nodes_[ni].is_leaf) return nodes_[ni].leaf_p;
        return x[nodes_[ni].feat] <= nodes_[ni].thresh
               ? pred_one(x, nodes_[ni].left)
               : pred_one(x, nodes_[ni].right);
    }

public:
    explicit DecisionTree(int max_depth=-1, int min_split=2,
                          int min_leaf=1, const std::string& crit="gini")
        : max_depth_(max_depth), min_split_(min_split), min_leaf_(min_leaf),
          use_entropy_(crit=="entropy") {}

    void fit(const Matrix& X, const IVector& y) override {
        nodes_.clear();
        std::vector<std::size_t> all(X.size());
        std::iota(all.begin(), all.end(), 0);
        build(X, y, all, 0);
    }

    Vector predict_proba(const Matrix& X) const override {
        Vector pr(X.size());
        for (std::size_t i=0;i<X.size();++i) pr[i] = pred_one(X[i], 0);
        return pr;
    }

    IVector predict(const Matrix& X) const override {
        auto pr = predict_proba(X);
        IVector out(pr.size());
        for (std::size_t i=0;i<pr.size();++i) out[i] = pr[i]>=0.5?1:0;
        return out;
    }

    std::string name() const override { return "DecisionTree"; }
};

// ── 6.3  Random Forest ───────────────────────────────────────────────────────
class RandomForest : public Classifier {
    int n_est_, max_depth_, min_split_;
    std::string max_feat_;
    unsigned seed_;

    struct TreeBundle {
        DecisionTree tree;
        std::vector<std::size_t> feat_idx;
    };
    std::vector<TreeBundle> forest_;

    std::size_t n_feat(std::size_t p) const {
        if (max_feat_=="sqrt") return std::max((std::size_t)1, (std::size_t)std::sqrt(p));
        if (max_feat_=="log2") return std::max((std::size_t)1, (std::size_t)std::log2(p)+1);
        return p;
    }

public:
    explicit RandomForest(int n_est=100, int max_depth=-1, int min_split=2,
                          std::string max_feat="sqrt", unsigned seed=42)
        : n_est_(n_est), max_depth_(max_depth), min_split_(min_split),
          max_feat_(std::move(max_feat)), seed_(seed) {}

    void fit(const Matrix& X, const IVector& y) override {
        forest_.clear();
        std::size_t n = X.size(), p = X[0].size();
        std::size_t nf = n_feat(p);
        std::mt19937 rng(seed_);
        std::uniform_int_distribution<std::size_t> rdist(0, n-1);

        for (int t = 0; t < n_est_; ++t) {
            // Bootstrap
            std::vector<std::size_t> boot(n);
            for (auto& b : boot) b = rdist(rng);

            // Feature subset
            std::vector<std::size_t> fidx(p);
            std::iota(fidx.begin(), fidx.end(), 0);
            std::shuffle(fidx.begin(), fidx.end(), rng);
            fidx.resize(nf);
            std::sort(fidx.begin(), fidx.end());

            // Build sub-matrices
            Matrix Xb(n, Vector(nf)); IVector yb(n);
            for (std::size_t i=0;i<n;++i) {
                for (std::size_t j=0;j<nf;++j) Xb[i][j] = X[boot[i]][fidx[j]];
                yb[i] = y[boot[i]];
            }

            DecisionTree dt(max_depth_, min_split_);
            dt.fit(Xb, yb);
            forest_.push_back({std::move(dt), fidx});
        }
    }

    Vector predict_proba(const Matrix& X) const override {
        std::size_t n = X.size();
        Vector pr(n, 0.0);
        for (const auto& tb : forest_) {
            std::size_t nf = tb.feat_idx.size();
            Matrix Xs(n, Vector(nf));
            for (std::size_t i=0;i<n;++i)
                for (std::size_t j=0;j<nf;++j) Xs[i][j] = X[i][tb.feat_idx[j]];
            auto p_ = tb.tree.predict_proba(Xs);
            for (std::size_t i=0;i<n;++i) pr[i] += p_[i];
        }
        for (auto& v : pr) v /= n_est_;
        return pr;
    }

    IVector predict(const Matrix& X) const override {
        auto pr = predict_proba(X);
        IVector out(pr.size());
        for (std::size_t i=0;i<pr.size();++i) out[i] = pr[i]>=0.5?1:0;
        return out;
    }

    std::string name() const override { return "RandomForest"; }
};

// ── 6.4  XGBoost — Gradient Boosted Trees (pure C++, from scratch) ───────────
//
//  Algorithm (mirrors the original XGBoost paper, Chen & Guestrin 2016):
//
//  Loss        : binary log-loss  L = -[y·log(p) + (1-y)·log(1-p)]
//  Gradient    : g_i  = p_i - y_i          (∂L/∂F, first-order)
//  Hessian     : h_i  = p_i·(1-p_i)        (∂²L/∂F², second-order)
//
//  Tree building (exact greedy):
//    Split gain = 0.5 · [ G_L²/(H_L+λ) + G_R²/(H_R+λ)
//                        - (G_L+G_R)²/(H_L+H_R+λ) ]  −  γ
//    Leaf weight w* = −G / (H + λ)
//
//  Regularisation : λ (L2 on leaf weights), γ (minimum gain to split)
//  Subsampling    : subsample_ (rows), colsample_ (columns per tree)
//  Shrinkage      : learning_rate (η)

struct XGBNode {
    int    feat    = -1;
    double thresh  = 0.0;
    int    left    = -1, right = -1;
    double weight  = 0.0;
    bool   is_leaf = false;
};

class XGBTree {
public:
    std::vector<XGBNode>       nodes_;
    std::vector<std::size_t>   feat_idx_;   // column subset used for this tree

    double predict_one(const Vector& x) const {
        int ni = 0;
        while (!nodes_[ni].is_leaf)
            ni = x[nodes_[ni].feat] <= nodes_[ni].thresh
                 ? nodes_[ni].left : nodes_[ni].right;
        return nodes_[ni].weight;
    }
};

class XGBoostClassifier : public Classifier {
    int    n_est_, max_depth_;
    double lr_, subsample_, colsample_, lambda_, gamma_;
    unsigned seed_;
    std::vector<XGBTree> trees_;
    double base_score_ = 0.0;

    static double sigmoid(double x) {
        // clamp to prevent exp overflow
        x = std::max(-35.0, std::min(35.0, x));
        return 1.0 / (1.0 + std::exp(-x));
    }

    // Recursively build one tree node.
    // Returns index of the new node inside tree.nodes_.
    int build_node(XGBTree& tree,
                   const Matrix& X,
                   const Vector& g, const Vector& h,
                   const std::vector<std::size_t>& idx,
                   int depth) {
        int ni = (int)tree.nodes_.size();
        tree.nodes_.emplace_back();

        // Aggregate gradients / hessians at this node
        double G = 0.0, H = 0.0;
        for (std::size_t i : idx) { G += g[i]; H += h[i]; }

        // Optimal leaf weight (closed-form solution)
        double w_opt = -G / (H + lambda_);

        bool force_leaf = (max_depth_ >= 0 && depth >= max_depth_)
                       || (int)idx.size() < 2;

        if (!force_leaf) {
            int    best_feat   = -1;
            double best_thresh = 0.0;
            double best_gain   = gamma_;   // split only if gain > γ

            for (std::size_t f : tree.feat_idx_) {
                // Sort samples by this feature value
                std::vector<std::pair<double, std::size_t>> sv;
                sv.reserve(idx.size());
                for (std::size_t i : idx) sv.push_back({X[i][f], i});
                std::sort(sv.begin(), sv.end());

                double GL = 0.0, HL = 0.0;
                for (std::size_t k = 0; k + 1 < sv.size(); ++k) {
                    GL += g[sv[k].second];
                    HL += h[sv[k].second];

                    // Skip tied feature values (no valid split point between them)
                    if (sv[k].first == sv[k+1].first) continue;

                    // Enforce minimum child hessian (≈ min_child_weight = 1)
                    double GR = G - GL, HR = H - HL;
                    if (HL < 1.0 || HR < 1.0) continue;

                    double gain = 0.5 * ( GL*GL / (HL + lambda_)
                                        + GR*GR / (HR + lambda_)
                                        - G*G  / (H  + lambda_) ) - gamma_;
                    if (gain > best_gain) {
                        best_gain   = gain;
                        best_feat   = (int)f;
                        best_thresh = (sv[k].first + sv[k+1].first) / 2.0;
                    }
                }
            }

            if (best_feat >= 0) {
                std::vector<std::size_t> li, ri;
                for (std::size_t i : idx)
                    (X[i][best_feat] <= best_thresh ? li : ri).push_back(i);

                tree.nodes_[ni].feat   = best_feat;
                tree.nodes_[ni].thresh = best_thresh;
                tree.nodes_[ni].left   = build_node(tree, X, g, h, li, depth + 1);
                tree.nodes_[ni].right  = build_node(tree, X, g, h, ri, depth + 1);
                return ni;
            }
        }

        // Leaf
        tree.nodes_[ni].is_leaf = true;
        tree.nodes_[ni].weight  = w_opt;
        return ni;
    }

public:
    explicit XGBoostClassifier(int n_est=100, int max_depth=5,
                                double lr=0.1,  double sub=1.0,
                                double col=1.0, double lambda=1.0,
                                double gamma=0.0, unsigned seed=42)
        : n_est_(n_est), max_depth_(max_depth), lr_(lr),
          subsample_(sub), colsample_(col),
          lambda_(lambda), gamma_(gamma), seed_(seed) {}

    void fit(const Matrix& X, const IVector& y) override {
        trees_.clear();
        std::size_t n = X.size(), p = X[0].size();
        std::mt19937 rng(seed_);

        // Base score = log-odds of positive-class prior
        double pos = (double)std::count(y.begin(), y.end(), 1);
        double neg = (double)n - pos;
        base_score_ = (pos > 0 && neg > 0) ? std::log(pos / neg) : 0.0;

        // Running raw scores  F_i  (before sigmoid)
        Vector F(n, base_score_);

        std::size_t n_col = std::max((std::size_t)1,
                            (std::size_t)std::round(colsample_ * (double)p));

        std::vector<std::size_t> all_rows(n), all_feats(p);
        std::iota(all_rows.begin(),  all_rows.end(),  0);
        std::iota(all_feats.begin(), all_feats.end(), 0);

        for (int t = 0; t < n_est_; ++t) {
            // ── Compute gradients & hessians ──
            Vector gv(n), hv(n);
            for (std::size_t i = 0; i < n; ++i) {
                double pi = sigmoid(F[i]);
                gv[i] = pi - (double)y[i];
                hv[i] = std::max(1e-6, pi * (1.0 - pi));
            }

            // ── Row subsampling ──
            std::vector<std::size_t> row_idx = all_rows;
            if (subsample_ < 1.0) {
                std::shuffle(row_idx.begin(), row_idx.end(), rng);
                row_idx.resize((std::size_t)(subsample_ * (double)n));
            }

            // ── Column subsampling (per tree) ──
            std::vector<std::size_t> col_idx = all_feats;
            if (n_col < p) {
                std::shuffle(col_idx.begin(), col_idx.end(), rng);
                col_idx.resize(n_col);
                std::sort(col_idx.begin(), col_idx.end());
            }

            // ── Build tree ──
            XGBTree tree;
            tree.feat_idx_ = col_idx;
            build_node(tree, X, gv, hv, row_idx, 0);
            trees_.push_back(tree);

            // ── Update F for all rows (not just sampled rows) ──
            for (std::size_t i = 0; i < n; ++i)
                F[i] += lr_ * tree.predict_one(X[i]);
        }
    }

    Vector predict_proba(const Matrix& X) const override {
        Vector pr(X.size());
        for (std::size_t i = 0; i < X.size(); ++i) {
            double score = base_score_;
            for (const auto& tree : trees_)
                score += lr_ * tree.predict_one(X[i]);
            pr[i] = sigmoid(score);
        }
        return pr;
    }

    IVector predict(const Matrix& X) const override {
        auto pr = predict_proba(X);
        IVector out(pr.size());
        for (std::size_t i = 0; i < pr.size(); ++i)
            out[i] = pr[i] >= 0.5 ? 1 : 0;
        return out;
    }

    std::string name() const override { return "XGBoost"; }
};

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 7 — STRATIFIED K-FOLD CV  (with in-fold SMOTE)
// ─────────────────────────────────────────────────────────────────────────────

// Full preprocessing pipeline applied inside each fold
static double cv_f1(Classifier& model,
                    const Matrix& Xnum, const std::vector<StrVec>& Xcat,
                    const IVector& y, int K=5, unsigned seed=42) {
    std::size_t n = y.size();

    // Build stratified fold assignment
    std::vector<std::size_t> i0, i1;
    for (std::size_t i=0;i<n;++i) (y[i]==0?i0:i1).push_back(i);
    std::mt19937 rng(seed);
    std::shuffle(i0.begin(),i0.end(),rng);
    std::shuffle(i1.begin(),i1.end(),rng);

    std::vector<int> fold(n);
    for (std::size_t i=0;i<i0.size();++i) fold[i0[i]] = (int)(i%K);
    for (std::size_t i=0;i<i1.size();++i) fold[i1[i]] = (int)(i%K);

    SMOTE smote(5, seed);
    double sum_f1 = 0.0;

    for (int k=0;k<K;++k) {
        std::vector<std::size_t> tr, va;
        for (std::size_t i=0;i<n;++i) (fold[i]==k ? va : tr).push_back(i);

        // Build fold matrices
        Matrix Xntr(tr.size(), Vector(Xnum[0].size()));
        Matrix Xnva(va.size(), Vector(Xnum[0].size()));
        std::vector<StrVec> Xctr(Xcat.size()), Xcva(Xcat.size());
        IVector ytr, yva;

        for (std::size_t i=0;i<tr.size();++i) {
            Xntr[i] = Xnum[tr[i]]; ytr.push_back(y[tr[i]]);
            for (std::size_t j=0;j<Xcat.size();++j) Xctr[j].push_back(Xcat[j][tr[i]]);
        }
        for (std::size_t i=0;i<va.size();++i) {
            Xnva[i] = Xnum[va[i]]; yva.push_back(y[va[i]]);
            for (std::size_t j=0;j<Xcat.size();++j) Xcva[j].push_back(Xcat[j][va[i]]);
        }

        // Fit scaler & encoder on training fold only
        StandardScaler sc; Matrix Xstr = sc.fit_transform(Xntr);
        Matrix Xsva = sc.transform(Xnva);
        OneHotEncoder enc; enc.fit(Xctr);
        Matrix Xotr = enc.transform(Xctr);
        Matrix Xova = enc.transform(Xcva);

        Matrix Xtr_full = hconcat(Xstr, Xotr);
        Matrix Xva_full = hconcat(Xsva, Xova);

        // SMOTE on training fold
        auto [Xs, ys] = smote.fit_resample(Xtr_full, ytr);

        model.fit(Xs, ys);
        IVector ypred = model.predict(Xva_full);

        int tp=0,fp=0,fn=0;
        for (std::size_t i=0;i<yva.size();++i) {
            if (ypred[i]==1&&yva[i]==1) tp++;
            else if (ypred[i]==1&&yva[i]==0) fp++;
            else if (ypred[i]==0&&yva[i]==1) fn++;
        }
        double pr_v = (tp+fp)?(double)tp/(tp+fp):0.0;
        double rc_v = (tp+fn)?(double)tp/(tp+fn):0.0;
        sum_f1 += (pr_v+rc_v)>0 ? 2.0*pr_v*rc_v/(pr_v+rc_v) : 0.0;
    }
    return sum_f1 / K;
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 8 — EVALUATE (train on full split, test on held-out)
// ─────────────────────────────────────────────────────────────────────────────

struct EvalResult {
    std::string model_name, phase;
    Metrics test_m;
    double cv_f1_mean = 0;
};

static EvalResult evaluate(Classifier& model,
                            const DataSplit& sp,
                            const std::string& phase,
                            int cv_k = 5) {
    // CV F1
    double cf1 = cv_f1(model, sp.X_num_tr, sp.X_cat_tr, sp.y_tr, cv_k);

    // Final fit on full training set
    StandardScaler sc; Matrix Xntr = sc.fit_transform(sp.X_num_tr);
    Matrix Xnte = sc.transform(sp.X_num_te);
    OneHotEncoder enc; enc.fit(sp.X_cat_tr);
    Matrix Xotr = enc.transform(sp.X_cat_tr);
    Matrix Xote = enc.transform(sp.X_cat_te);

    Matrix Xtr = hconcat(Xntr, Xotr);
    Matrix Xte = hconcat(Xnte, Xote);

    SMOTE smote(5,42);
    auto [Xs,ys] = smote.fit_resample(Xtr, sp.y_tr);

    auto t0 = std::chrono::high_resolution_clock::now();
    model.fit(Xs, ys);
    auto t1 = std::chrono::high_resolution_clock::now();
    IVector ypred = model.predict(Xte);
    auto t2 = std::chrono::high_resolution_clock::now();
    Vector yprob = model.predict_proba(Xte);

    EvalResult r;
    r.model_name = model.name();
    r.phase      = phase;
    r.cv_f1_mean = cf1;
    r.test_m     = compute_metrics(sp.y_te, ypred, yprob);
    r.test_m.train_us = std::chrono::duration<double,std::micro>(t1-t0).count() / Xs.size();
    r.test_m.pred_us  = std::chrono::duration<double,std::micro>(t2-t1).count() / Xte.size();
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 9 — HYPERPARAMETER OPTIMISATION
// ─────────────────────────────────────────────────────────────────────────────

// Unified HPO objective builder
static Objective make_objective(const std::string& mname,
                                const Matrix& Xnum,
                                const std::vector<StrVec>& Xcat,
                                const IVector& y) {
    return [mname, Xnum, Xcat, y](const HPConfig& cfg) -> double {
        auto get_d = [&](const std::string& k) -> double {
            return std::holds_alternative<double>(cfg.at(k))
                   ? std::get<double>(cfg.at(k))
                   : (double)std::get<int>(cfg.at(k));
        };
        auto get_i = [&](const std::string& k) -> int {
            return std::holds_alternative<int>(cfg.at(k))
                   ? std::get<int>(cfg.at(k))
                   : (int)std::get<double>(cfg.at(k));
        };
        auto get_s = [&](const std::string& k) -> std::string {
            return std::get<std::string>(cfg.at(k));
        };

        std::unique_ptr<Classifier> m;
        if (mname == "LogisticRegression") {
            m = std::make_unique<LogisticRegression>(get_d("C"));
        } else if (mname == "DecisionTree") {
            int md = cfg.count("max_depth") && std::holds_alternative<int>(cfg.at("max_depth"))
                     ? get_i("max_depth") : -1;
            m = std::make_unique<DecisionTree>(
                md, get_i("min_samples_split"),
                get_i("min_samples_leaf"), get_s("criterion"));
        } else if (mname == "RandomForest") {
            int md = cfg.count("max_depth") && std::holds_alternative<int>(cfg.at("max_depth"))
                     ? get_i("max_depth") : -1;
            m = std::make_unique<RandomForest>(
                get_i("n_estimators"), md,
                get_i("min_samples_split"), get_s("max_features"));
        } else {  // XGBoost
            m = std::make_unique<XGBoostClassifier>(
                get_i("n_estimators"), get_i("max_depth"),
                get_d("learning_rate"), get_d("subsample"),
                get_d("colsample_bytree"));
        }
        return cv_f1(*m, Xnum, Xcat, y, 5, 42);
    };
}

// Helper: print an HPConfig to stdout (avoids structured-binding lambda capture)
static void print_cfg(const HPConfig& cfg) {
    for (auto it = cfg.begin(); it != cfg.end(); ++it) {
        const std::string& key = it->first;
        const HPValue&     val = it->second;
        std::cout << key << "=";
        if      (std::holds_alternative<int>        (val)) std::cout << std::get<int>        (val);
        else if (std::holds_alternative<double>     (val)) std::cout << std::get<double>     (val);
        else if (std::holds_alternative<std::string>(val)) std::cout << std::get<std::string>(val);
        std::cout << " ";
    }
}

// ── 9.1  Grid Search ─────────────────────────────────────────────────────────
class GridSearchCV {
public:
    std::map<std::string,std::vector<HPValue>> grid_;
    HPConfig best_params_; double best_score_ = -1e18;

    explicit GridSearchCV(std::map<std::string,std::vector<HPValue>> g)
        : grid_(std::move(g)) {}

    void run(Objective obj, const std::string& model_name) {
        std::vector<std::string> keys;
        std::vector<std::size_t> sizes;
        for (const auto& [k,v]:grid_) { keys.push_back(k); sizes.push_back(v.size()); }

        std::vector<std::size_t> idx(keys.size(),0);
        int trial=0;
        while (true) {
            HPConfig cfg;
            for (std::size_t i=0;i<keys.size();++i) cfg[keys[i]] = grid_[keys[i]][idx[i]];

            std::cout << "  [GS " << model_name << " #" << ++trial << "] ";
            print_cfg(cfg);
            double s = obj(cfg);
            std::cout << "→ F1=" << std::fixed << std::setprecision(4) << s << "\n";

            if (s > best_score_) { best_score_=s; best_params_=cfg; }

            // increment
            int carry=1;
            for (int i=(int)keys.size()-1;i>=0&&carry;--i) {
                idx[i]+=carry;
                if (idx[i]>=sizes[i]) { idx[i]=0; carry=1; } else carry=0;
            }
            if (carry) break;
        }
        std::cout << "  Grid best F1=" << best_score_ << "\n";
    }
};

// ── 9.2  Random Search ───────────────────────────────────────────────────────
class RandomSearchCV {
public:
    std::map<std::string,std::vector<HPValue>> dists_;
    int n_iter_; unsigned seed_;
    HPConfig best_params_; double best_score_ = -1e18;

    RandomSearchCV(std::map<std::string,std::vector<HPValue>> d,
                   int n=50, unsigned seed=42)
        : dists_(std::move(d)), n_iter_(n), seed_(seed) {}

    void run(Objective obj, const std::string& model_name) {
        std::mt19937 rng(seed_);
        for (int t=0;t<n_iter_;++t) {
            HPConfig cfg;
            for (auto& [k,vals]:dists_) {
                std::uniform_int_distribution<std::size_t> d(0,vals.size()-1);
                cfg[k] = vals[d(rng)];
            }
            std::cout << "  [RS " << model_name << " #" << t+1 << "] ";
            print_cfg(cfg);
            double s = obj(cfg);
            std::cout << "→ F1=" << std::fixed << std::setprecision(4) << s << "\n";
            if (s > best_score_) { best_score_=s; best_params_=cfg; }
        }
        std::cout << "  Random best F1=" << best_score_ << "\n";
    }
};

// ── 9.3  Bayesian Optimisation (GP + Expected Improvement) ───────────────────
class BayesianOptimisation {
public:
    struct ParamSpec {
        std::string name, type;  // "real", "int", "cat"
        double lo=0, hi=1; bool log_scale=false;
        std::vector<HPValue> cats;
    };

    std::vector<ParamSpec> specs_;
    int n_iter_, n_init_;
    unsigned seed_;
    HPConfig best_params_; double best_score_ = -1e18;

    // Observed data for GP
    std::vector<Vector> X_obs_;
    Vector y_obs_;

    BayesianOptimisation(std::vector<ParamSpec> specs,
                          int n_iter=50, int n_init=10, unsigned seed=42)
        : specs_(std::move(specs)), n_iter_(n_iter), n_init_(n_init), seed_(seed) {}

    // Map config → normalised [0,1]^d
    Vector normalise(const HPConfig& cfg) const {
        Vector x(specs_.size());
        for (std::size_t i=0;i<specs_.size();++i) {
            const auto& sp = specs_[i];
            if (sp.type=="cat") {
                auto it = std::find(sp.cats.begin(),sp.cats.end(),cfg.at(sp.name));
                x[i] = (double)std::distance(sp.cats.begin(),it)
                      / std::max(1.0,(double)(sp.cats.size()-1));
            } else {
                double v = std::holds_alternative<double>(cfg.at(sp.name))
                           ? std::get<double>(cfg.at(sp.name))
                           : (double)std::get<int>(cfg.at(sp.name));
                if (sp.log_scale)
                    x[i] = (std::log(v)-std::log(sp.lo))/(std::log(sp.hi)-std::log(sp.lo));
                else
                    x[i] = (v-sp.lo)/(sp.hi-sp.lo);
                x[i] = std::clamp(x[i],0.0,1.0);
            }
        }
        return x;
    }

    // Sample random config
    HPConfig sample(std::mt19937& rng) const {
        HPConfig cfg;
        std::uniform_real_distribution<double> ud(0.0,1.0);
        for (const auto& sp : specs_) {
            if (sp.type=="cat") {
                std::uniform_int_distribution<std::size_t> ci(0,sp.cats.size()-1);
                cfg[sp.name] = sp.cats[ci(rng)];
            } else {
                double u = ud(rng), v;
                if (sp.log_scale)
                    v = std::exp(std::log(sp.lo)+u*(std::log(sp.hi)-std::log(sp.lo)));
                else
                    v = sp.lo + u*(sp.hi-sp.lo);
                cfg[sp.name] = (sp.type=="int") ? HPValue((int)std::round(v)) : HPValue(v);
            }
        }
        return cfg;
    }

    // RBF kernel  k(x,x') = exp(−||x−x'||² / (2·ℓ²))
    double kernel(const Vector& a, const Vector& b, double l=0.5) const {
        double d=0;
        for (std::size_t i=0;i<a.size();++i) { double di=a[i]-b[i]; d+=di*di; }
        return std::exp(-d/(2.0*l*l));
    }

    // GP posterior (mean, variance) at query point x
    std::pair<double,double> gp_posterior(const Vector& xq) const {
        std::size_t n = X_obs_.size();
        if (!n) return {0.0,1.0};
        double noise = 1e-4;

        // Build K  (n×n)
        std::vector<Vector> K(n, Vector(n));
        for (std::size_t i=0;i<n;++i)
            for (std::size_t j=0;j<n;++j)
                K[i][j] = kernel(X_obs_[i],X_obs_[j]) + (i==j?noise:0.0);

        Vector ks(n);
        for (std::size_t i=0;i<n;++i) ks[i] = kernel(xq, X_obs_[i]);

        // Gaussian elimination to solve K·α = y
        auto solve = [&](Vector rhs) -> Vector {
            auto A = K; // copy
            for (std::size_t col=0;col<n;++col) {
                std::size_t piv=col;
                for (std::size_t r=col+1;r<n;++r)
                    if (std::abs(A[r][col])>std::abs(A[piv][col])) piv=r;
                std::swap(A[col],A[piv]); std::swap(rhs[col],rhs[piv]);
                for (std::size_t r=col+1;r<n;++r) {
                    if (std::abs(A[col][col])<1e-12) continue;
                    double f=A[r][col]/A[col][col];
                    for (std::size_t c=col;c<n;++c) A[r][c]-=f*A[col][c];
                    rhs[r]-=f*rhs[col];
                }
            }
            Vector x(n,0.0);
            for (int i=(int)n-1;i>=0;--i) {
                if (std::abs(A[i][i])<1e-12) continue;
                x[i]=rhs[i];
                for (std::size_t j=i+1;j<n;++j) x[i]-=A[i][j]*x[j];
                x[i]/=A[i][i];
            }
            return x;
        };

        Vector alpha = solve(y_obs_);
        double mu=0; for (std::size_t i=0;i<n;++i) mu+=ks[i]*alpha[i];

        Vector v = solve(ks);
        double dot=0; for (std::size_t i=0;i<n;++i) dot+=ks[i]*v[i];
        double sigma2 = std::max(0.0, kernel(xq,xq)-dot);
        return {mu, sigma2};
    }

    // Expected Improvement
    double EI(const Vector& xq) const {
        if (X_obs_.empty()) return 1.0;
        double fbest = *std::max_element(y_obs_.begin(),y_obs_.end());
        auto [mu,sig2] = gp_posterior(xq);
        double sig = std::sqrt(sig2);
        if (sig<1e-10) return 0.0;
        double z = (mu-fbest)/sig;
        auto Phi  = [](double x){ return 0.5*std::erfc(-x*0.7071067811865476); };
        auto phi  = [](double x){ return std::exp(-0.5*x*x)/2.506628274631001; };
        return (mu-fbest)*Phi(z) + sig*phi(z);
    }

    void run(Objective obj, const std::string& model_name) {
        std::mt19937 rng(seed_);
        X_obs_.clear(); y_obs_.clear();

        auto eval_cfg = [&](const HPConfig& cfg, const std::string& tag, int t) {
            std::cout << "  [BO " << model_name << " " << tag
                      << " #" << t << "] ";
            print_cfg(cfg);
            double s = obj(cfg);
            std::cout << "→ F1=" << std::fixed << std::setprecision(4) << s << "\n";
            X_obs_.push_back(normalise(cfg));
            y_obs_.push_back(s);
            if (s > best_score_) { best_score_=s; best_params_=cfg; }
        };

        // Random initialisation
        for (int i=0;i<n_init_;++i) eval_cfg(sample(rng), "init", i+1);

        // GP-guided iterations
        for (int i=0;i<n_iter_-n_init_;++i) {
            // Sample 1000 candidates, pick max EI
            HPConfig best_cand;
            double best_ei = -1e18;
            for (int c=0;c<1000;++c) {
                HPConfig cand = sample(rng);
                double ei = EI(normalise(cand));
                if (ei > best_ei) { best_ei=ei; best_cand=cand; }
            }
            eval_cfg(best_cand, "BO", n_init_+i+1);
        }
        std::cout << "  Bayesian best F1=" << best_score_ << "\n";
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 10 — RESULT TABLE
// ─────────────────────────────────────────────────────────────────────────────

static void print_table(const std::vector<EvalResult>& results) {
    std::cout << "\n" << std::string(100,'=') << "\n";
    std::cout << std::left
              << std::setw(22) << "Phase"
              << std::setw(22) << "Model"
              << std::setw(10) << "CV-F1"
              << std::setw(10) << "Acc"
              << std::setw(10) << "Prec"
              << std::setw(10) << "Recall"
              << std::setw(10) << "F1"
              << std::setw(10) << "AUC"
              << "\n" << std::string(100,'-') << "\n";
    for (const auto& r : results) {
        std::cout << std::left
                  << std::setw(22) << r.phase
                  << std::setw(22) << r.model_name
                  << std::fixed << std::setprecision(4)
                  << std::setw(10) << r.cv_f1_mean
                  << std::setw(10) << r.test_m.accuracy
                  << std::setw(10) << r.test_m.precision
                  << std::setw(10) << r.test_m.recall
                  << std::setw(10) << r.test_m.f1
                  << std::setw(10) << r.test_m.auc_roc
                  << "\n";
    }
    std::cout << std::string(100,'=') << "\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 11 — MAIN
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const std::string DATA_PATH = (argc > 1) ? argv[1] : "customer_churn_dataset.csv";

    // ── Column definitions (mirror notebook) ──────────────────────────────────
    const StrVec NUM_BASE = {
        "Age","Account_Balance","Transaction_Amount","Loan_Amount","Credit_Score",
        "Annual_Income","Number_of_Transactions","Customer_Service_Interactions",
        "Recent_Complaints","Change_in_Account_Balance","Customer_Satisfaction_Score",
        "Is_Employed"
    };
    const StrVec CAT_COLS = {
        "Gender","Account_Type","Transaction_Type",
        "Marital_Status","Region","Account_Activity_Trend"
    };
    const StrVec NUM_FE = {   // baseline + 9 engineered
        "Age","Account_Balance","Transaction_Amount","Loan_Amount","Credit_Score",
        "Annual_Income","Number_of_Transactions","Customer_Service_Interactions",
        "Recent_Complaints","Change_in_Account_Balance","Customer_Satisfaction_Score",
        "Is_Employed",
        "Customer_Tenure_Months","Inactivity_Period_Days","Transaction_Frequency",
        "Avg_Transaction_Size","Balance_Volatility_Index","Loan_to_Income_Ratio",
        "Complaint_to_Interaction_Ratio","Satisfaction_Gap","Engagement_Index"
    };
    const std::string TARGET = "Churn_Label";

    std::cout << "═══════════════════════════════════════════════════\n"
              << "  Customer Churn Prediction — C++ Implementation\n"
              << "═══════════════════════════════════════════════════\n\n";

    // ── 1. Load data ──────────────────────────────────────────────────────────
    std::cout << "[1] Loading data\n";
    DataFrame df = load_csv(DATA_PATH);

    // ── 2. Feature engineering ────────────────────────────────────────────────
    std::cout << "\n[2] Feature engineering\n";
    engineer_features(df);

    // ── 3. Extract feature matrices ───────────────────────────────────────────
    std::cout << "\n[3] Extracting features\n";
    auto base_data = extract(df, NUM_BASE, CAT_COLS, TARGET);
    auto enr_data  = extract(df, NUM_FE,   CAT_COLS, TARGET);

    // ── 4. Stratified 80/20 train/test split ──────────────────────────────────
    std::cout << "\n[4] Stratified 80/20 split\n";
    std::size_t n = df.n_rows;
    std::vector<std::size_t> i0, i1;
    for (std::size_t i=0;i<n;++i) (base_data.y[i]==0?i0:i1).push_back(i);
    std::mt19937 rng(42);
    std::shuffle(i0.begin(),i0.end(),rng); std::shuffle(i1.begin(),i1.end(),rng);
    std::size_t te0=(std::size_t)(i0.size()*0.2), te1=(std::size_t)(i1.size()*0.2);
    std::vector<std::size_t> tr_idx, te_idx;
    for (std::size_t i=0;i<te0;++i)         te_idx.push_back(i0[i]);
    for (std::size_t i=te0;i<i0.size();++i) tr_idx.push_back(i0[i]);
    for (std::size_t i=0;i<te1;++i)         te_idx.push_back(i1[i]);
    for (std::size_t i=te1;i<i1.size();++i) tr_idx.push_back(i1[i]);
    std::cout << "Train=" << tr_idx.size() << "  Test=" << te_idx.size() << "\n";

    auto base_sp = make_split(base_data, tr_idx, te_idx);
    auto enr_sp  = make_split(enr_data,  tr_idx, te_idx);

    std::vector<EvalResult> all_res;

    // ══════════════════════════════════════════════════════════════════════════
    // PHASE 1 — BASELINE MODELS
    // ══════════════════════════════════════════════════════════════════════════
    std::cout << "\n[Phase 1] Baseline models (original 12 numeric + 6 categorical features)\n"
              << std::string(70,'-') << "\n";

    for (auto& [mk, mdl] : std::initializer_list<std::pair<std::string,std::unique_ptr<Classifier>>>{
        {"LR_base",  std::make_unique<LogisticRegression>(1.0)},
        {"DT_base",  std::make_unique<DecisionTree>(5,2,1,"gini")},
        {"RF_base",  std::make_unique<RandomForest>(100,-1,2,"sqrt")},
        {"XGB_base", std::make_unique<XGBoostClassifier>(100,5,0.1,1.0,1.0)}})
    {
        std::cout << "  → " << mk << "\n";
        auto r = evaluate(*mdl, base_sp, "Phase1-Baseline");
        all_res.push_back(r);
        std::cout << "     CV-F1=" << r.cv_f1_mean
                  << "  Test-F1=" << r.test_m.f1
                  << "  AUC=" << r.test_m.auc_roc << "\n";
    }

    // ══════════════════════════════════════════════════════════════════════════
    // PHASE 2 — ENRICHED FEATURES
    // ══════════════════════════════════════════════════════════════════════════
    std::cout << "\n[Phase 2] Enriched features (+9 engineered features)\n"
              << std::string(70,'-') << "\n";

    for (auto& [mk, mdl] : std::initializer_list<std::pair<std::string,std::unique_ptr<Classifier>>>{
        {"LR_enr",  std::make_unique<LogisticRegression>(1.0)},
        {"DT_enr",  std::make_unique<DecisionTree>(5,2,1,"gini")},
        {"RF_enr",  std::make_unique<RandomForest>(100,-1,2,"sqrt")},
        {"XGB_enr", std::make_unique<XGBoostClassifier>(100,5,0.1,1.0,1.0)}})
    {
        std::cout << "  → " << mk << "\n";
        auto r = evaluate(*mdl, enr_sp, "Phase2-Enriched");
        all_res.push_back(r);
        std::cout << "     CV-F1=" << r.cv_f1_mean
                  << "  Test-F1=" << r.test_m.f1
                  << "  AUC=" << r.test_m.auc_roc << "\n";
    }

    // ══════════════════════════════════════════════════════════════════════════
    // PHASE 3 — HYPERPARAMETER OPTIMISATION
    // ══════════════════════════════════════════════════════════════════════════
    std::cout << "\n[Phase 3] Hyperparameter Optimisation on enriched features\n"
              << std::string(70,'-') << "\n";

    const StrVec MODELS = {"LogisticRegression","DecisionTree","RandomForest","XGBoost"};

    // Helper: build model from best HPConfig
    auto build_model = [](const std::string& mname, const HPConfig& p)
        -> std::unique_ptr<Classifier>
    {
        auto gi = [&](const std::string& k) -> int {
            return std::holds_alternative<int>(p.at(k))
                   ? std::get<int>(p.at(k)) : (int)std::get<double>(p.at(k));
        };
        auto gd = [&](const std::string& k) -> double {
            return std::holds_alternative<double>(p.at(k))
                   ? std::get<double>(p.at(k)) : (double)std::get<int>(p.at(k));
        };
        auto gs = [&](const std::string& k) -> std::string {
            return std::get<std::string>(p.at(k));
        };
        if (mname=="LogisticRegression")
            return std::make_unique<LogisticRegression>(gd("C"));
        if (mname=="DecisionTree") {
            int md = (p.count("max_depth")&&std::holds_alternative<int>(p.at("max_depth")))
                     ? gi("max_depth") : -1;
            return std::make_unique<DecisionTree>(md, gi("min_samples_split"),
                                                  gi("min_samples_leaf"), gs("criterion"));
        }
        if (mname=="RandomForest") {
            int md = (p.count("max_depth")&&std::holds_alternative<int>(p.at("max_depth")))
                     ? gi("max_depth") : -1;
            return std::make_unique<RandomForest>(gi("n_estimators"), md,
                                                  gi("min_samples_split"), gs("max_features"));
        }
        // XGBoost
        return std::make_unique<XGBoostClassifier>(
            gi("n_estimators"), gi("max_depth"),
            gd("learning_rate"), gd("subsample"), gd("colsample_bytree"));
    };

    // ── 3a  Grid Search ───────────────────────────────────────────────────────
    std::cout << "\n--- Grid Search ---\n";

    auto make_gs_grid = [](const std::string& mn)
        -> std::map<std::string,std::vector<HPValue>>
    {
        using V = std::vector<HPValue>;
        if (mn=="LogisticRegression")
            return {{"C", V{HPValue(0.01),HPValue(0.1),HPValue(1.0),HPValue(10.0)}}};
        if (mn=="DecisionTree")
            return {{"max_depth",        V{HPValue(3),HPValue(5),HPValue(7),HPValue(10)}},
                    {"min_samples_split",V{HPValue(2),HPValue(5),HPValue(10)}},
                    {"min_samples_leaf", V{HPValue(1),HPValue(2),HPValue(5)}},
                    {"criterion",        V{HPValue(std::string("gini")),HPValue(std::string("entropy"))}}};
        if (mn=="RandomForest")
            return {{"n_estimators",     V{HPValue(50),HPValue(100),HPValue(200)}},
                    {"max_depth",        V{HPValue(5),HPValue(10)}},
                    {"min_samples_split",V{HPValue(2),HPValue(5)}},
                    {"max_features",     V{HPValue(std::string("sqrt")),HPValue(std::string("log2"))}}};
        // XGBoost
        return {{"n_estimators",    V{HPValue(50),HPValue(100),HPValue(200)}},
                {"max_depth",       V{HPValue(3),HPValue(5),HPValue(7)}},
                {"learning_rate",   V{HPValue(0.01),HPValue(0.1),HPValue(0.3)}},
                {"subsample",       V{HPValue(0.7),HPValue(1.0)}},
                {"colsample_bytree",V{HPValue(0.8),HPValue(1.0)}}};
    };

    for (const auto& mn : MODELS) {
        std::cout << "\nGrid Search — " << mn << "\n";
        auto obj = make_objective(mn, enr_sp.X_num_tr, enr_sp.X_cat_tr, enr_sp.y_tr);
        GridSearchCV gs(make_gs_grid(mn));
        gs.run(obj, mn);

        auto mdl = build_model(mn, gs.best_params_);
        auto r = evaluate(*mdl, enr_sp, "HPO-GridSearch");
        all_res.push_back(r);
        std::cout << "  Final Test: F1=" << r.test_m.f1
                  << "  AUC=" << r.test_m.auc_roc << "\n";
    }

    // ── 3b  Random Search ─────────────────────────────────────────────────────
    std::cout << "\n--- Random Search ---\n";

    auto make_rs_dists = [](const std::string& mn)
        -> std::map<std::string,std::vector<HPValue>>
    {
        using V = std::vector<HPValue>;
        if (mn=="LogisticRegression") {
            V cv; for (double v:{0.001,0.003,0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0,100.0})
                       cv.push_back(HPValue(v));
            return {{"C", cv}};
        }
        if (mn=="DecisionTree")
            return {{"max_depth",        V{HPValue(3),HPValue(5),HPValue(7),HPValue(10),HPValue(15)}},
                    {"min_samples_split",V{HPValue(2),HPValue(3),HPValue(5),HPValue(7),HPValue(10),HPValue(15),HPValue(20)}},
                    {"min_samples_leaf", V{HPValue(1),HPValue(2),HPValue(3),HPValue(5),HPValue(7),HPValue(10)}},
                    {"criterion",        V{HPValue(std::string("gini")),HPValue(std::string("entropy"))}}};
        if (mn=="RandomForest") {
            V ne; for (int v:{50,75,100,125,150,175,200,250,300}) ne.push_back(HPValue(v));
            return {{"n_estimators",     ne},
                    {"max_depth",        V{HPValue(5),HPValue(10),HPValue(15)}},
                    {"min_samples_split",V{HPValue(2),HPValue(3),HPValue(5),HPValue(7),HPValue(10)}},
                    {"max_features",     V{HPValue(std::string("sqrt")),HPValue(std::string("log2"))}}};
        }
        // XGBoost
        V ne; for (int v:{50,75,100,125,150,175,200,250,300}) ne.push_back(HPValue(v));
        V lr; for (double v:{0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.5}) lr.push_back(HPValue(v));
        return {{"n_estimators",    ne},
                {"max_depth",       V{HPValue(3),HPValue(4),HPValue(5),HPValue(6),HPValue(7),HPValue(8),HPValue(10)}},
                {"learning_rate",   lr},
                {"subsample",       V{HPValue(0.6),HPValue(0.7),HPValue(0.8),HPValue(1.0)}},
                {"colsample_bytree",V{HPValue(0.6),HPValue(0.8),HPValue(1.0)}}};
    };

    for (const auto& mn : MODELS) {
        std::cout << "\nRandom Search — " << mn << "\n";
        auto obj = make_objective(mn, enr_sp.X_num_tr, enr_sp.X_cat_tr, enr_sp.y_tr);
        RandomSearchCV rs(make_rs_dists(mn), 50, 42);
        rs.run(obj, mn);

        auto mdl = build_model(mn, rs.best_params_);
        auto r = evaluate(*mdl, enr_sp, "HPO-RandomSearch");
        all_res.push_back(r);
        std::cout << "  Final Test: F1=" << r.test_m.f1
                  << "  AUC=" << r.test_m.auc_roc << "\n";
    }

    // ── 3c  Bayesian Optimisation ─────────────────────────────────────────────
    std::cout << "\n--- Bayesian Optimisation (GP-EI) ---\n";

    auto make_bo_specs = [](const std::string& mn)
        -> std::vector<BayesianOptimisation::ParamSpec>
    {
        if (mn=="LogisticRegression")
            return {{"C","real",0.001,100.0,true,{}}};
        if (mn=="DecisionTree")
            return {{"max_depth",       "int",  3,15,false,{}},
                    {"min_samples_split","int",  2,20,false,{}},
                    {"min_samples_leaf", "int",  1,10,false,{}},
                    {"criterion","cat",  0,1,false,
                     {HPValue(std::string("gini")),HPValue(std::string("entropy"))}}};
        if (mn=="RandomForest")
            return {{"n_estimators",    "int",  50,300,false,{}},
                    {"max_depth",       "int",  3, 20, false,{}},
                    {"min_samples_split","int",  2, 10, false,{}},
                    {"max_features","cat",0,1,false,
                     {HPValue(std::string("sqrt")),HPValue(std::string("log2"))}}};
        // XGBoost
        return {{"n_estimators",   "int",  50,300,false,{}},
                {"max_depth",      "int",   3, 10,false,{}},
                {"learning_rate",  "real",0.001,0.5,true,{}},
                {"subsample","cat",0,3,false,
                 {HPValue(0.6),HPValue(0.7),HPValue(0.8),HPValue(1.0)}},
                {"colsample_bytree","cat",0,2,false,
                 {HPValue(0.6),HPValue(0.8),HPValue(1.0)}}};
    };

    for (const auto& mn : MODELS) {
        std::cout << "\nBayesian Optimisation — " << mn << "\n";
        auto obj = make_objective(mn, enr_sp.X_num_tr, enr_sp.X_cat_tr, enr_sp.y_tr);
        BayesianOptimisation bo(make_bo_specs(mn), 50, 10, 42);
        bo.run(obj, mn);

        auto mdl = build_model(mn, bo.best_params_);
        auto r = evaluate(*mdl, enr_sp, "HPO-Bayesian");
        all_res.push_back(r);
        std::cout << "  Final Test: F1=" << r.test_m.f1
                  << "  AUC=" << r.test_m.auc_roc << "\n";
    }

    // ── Final results table ───────────────────────────────────────────────────
    std::cout << "\n\n===== FULL RESULTS SUMMARY =====";
    print_table(all_res);

    return 0;
}
