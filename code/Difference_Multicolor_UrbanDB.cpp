#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// ===================== Math & utils =====================

int lowerBound(const vector<double> &a, double x) {
  int lo = 0, hi = a.size();
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (a[mid] < x)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

int upperBound(const vector<double> &a, double x) {
  int lo = 0, hi = a.size();
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (a[mid] <= x)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

int intersectionSize(int aL, int aR, int bL, int bR) {
  int L = max(aL, bL), R = min(aR, bR);
  return (L <= R) ? (R - L + 1) : 0;
}

double tversky(int aL, int aR, int bL, int bR, double alpha, double beta) {
  int inter = intersectionSize(aL, aR, bL, bR);
  int sizeA = (aR - aL + 1);
  int sizeB = (bR - bL + 1);
  int aMinusB = sizeA - inter;
  int bMinusA = sizeB - inter;

  double denom = inter + (alpha * aMinusB) + (beta * bMinusA);
  return (denom == 0.0) ? 0.0 : ((double)inter) / denom;
}

// ===================== Data Structures =====================

struct Result {
  double leftVal, rightVal;
  double similarity;
  Result(double l, double r, double s)
      : leftVal(l), rightVal(r), similarity(s) {}
};

struct Data {
  int n;
  vector<double> keys;
  int m;
  vector<string> pairNames;
  set<string> uniqueColors;
  vector<vector<int>> prefix;
  vector<vector<double>> points;

  Data(int n, vector<double> keys, int m, vector<string> pairNames,
       set<string> uniqueColors, vector<vector<int>> prefix,
       vector<vector<double>> points)
      : n(n), keys(move(keys)), m(m), pairNames(move(pairNames)),
        uniqueColors(move(uniqueColors)), prefix(move(prefix)),
        points(move(points)) {}
};

struct RawRow {
  double key;
  int color;
  bool operator<(const RawRow &other) const { return key < other.key; }
};

// ===================== "Range tree" =====================

class RangeTree {
  struct Node {
    vector<double> lo, hi;
    int dim;
    double split;
    unique_ptr<Node> left, right;
    vector<int> idxs;
    bool isLeaf = false;
  };

  int dims;
  unique_ptr<Node> root;
  const vector<vector<double>> &P;

  unique_ptr<Node> build(vector<int> &idx, int loi, int hii, int depth) {
    int n = hii - loi;
    auto node = make_unique<Node>();
    node->lo.assign(dims, numeric_limits<double>::infinity());
    node->hi.assign(dims, -numeric_limits<double>::infinity());

    for (int k = loi; k < hii; k++) {
      const auto &pt = P[idx[k]];
      for (int d = 0; d < dims; d++) {
        if (pt[d] < node->lo[d])
          node->lo[d] = pt[d];
        if (pt[d] > node->hi[d])
          node->hi[d] = pt[d];
      }
    }

    if (n <= 32) {
      node->isLeaf = true;
      node->idxs.assign(idx.begin() + loi, idx.begin() + hii);
      return node;
    }

    int dim = depth % dims;
    int mid = loi + n / 2;

    // Use standard library for nth_element (faster & safer than manual
    // implementation)
    nth_element(idx.begin() + loi, idx.begin() + mid, idx.begin() + hii,
                [&](int a, int b) { return P[a][dim] < P[b][dim]; });

    node->dim = dim;
    node->split = P[idx[mid]][dim];
    node->left = build(idx, loi, mid, depth + 1);
    node->right = build(idx, mid, hii, depth + 1);
    return node;
  }

  bool overlaps(Node *node, const vector<double> &qlo,
                const vector<double> &qhi) {
    for (int d = 0; d < dims; d++) {
      if (node->hi[d] < qlo[d] || node->lo[d] > qhi[d])
        return false;
    }
    return true;
  }

  bool contains(const vector<double> &pt, const vector<double> &qlo,
                const vector<double> &qhi) {
    for (int d = 0; d < dims; d++) {
      if (pt[d] < qlo[d] || pt[d] > qhi[d])
        return false;
    }
    return true;
  }

  void rangeQuery(Node *node, const vector<double> &lo,
                  const vector<double> &hi, vector<int> &out) {
    if (!node)
      return;
    if (!overlaps(node, lo, hi))
      return;

    if (node->isLeaf) {
      for (int id : node->idxs) {
        if (contains(P[id], lo, hi))
          out.push_back(id);
      }
      return;
    }
    rangeQuery(node->left.get(), lo, hi, out);
    rangeQuery(node->right.get(), lo, hi, out);
  }

public:
  RangeTree(const vector<vector<double>> &points, int dims)
      : dims(dims), P(points) {
    int n = points.size();
    vector<int> all(n);
    for (int i = 0; i < n; i++)
      all[i] = i;
    root = build(all, 0, n, 0);
  }

  void query(const vector<double> &lo, const vector<double> &hi,
             vector<int> &out) {
    rangeQuery(root.get(), lo, hi, out);
  }
};

// ===================== Query logic =====================

unique_ptr<Result> queryBest(const Data &data, RangeTree &tree, int epsilon,
                             double startVal, double endVal) {
  int n = data.n;
  if (n == 0)
    return nullptr;

  int L0 = lowerBound(data.keys, min(startVal, endVal));
  int R0 = upperBound(data.keys, max(startVal, endVal)) - 1;
  if (L0 < 0)
    L0 = 0;
  if (R0 >= n)
    R0 = n - 1;
  if (L0 > R0)
    return nullptr;

  double bestT = -1.0;
  int bestL = -1, bestR = -1;
  double alpha = 0.5, beta = 0.5;

  vector<double> lo(data.m), hi(data.m);
  vector<int> hits;

  for (int p = 0; p <= n - 1; p++) {
    const auto &center = data.points[p];
    for (int d = 0; d < data.m; d++) {
      lo[d] = center[d] - epsilon;
      hi[d] = center[d] + epsilon;
    }

    hits.clear();
    tree.query(lo, hi, hits);

    for (int id : hits) {
      int R = id;
      if (R <= p)
        continue;
      int L = p + 1;

      double t = tversky(L, R, L0 + 1, R0 + 1, alpha, beta);

      if (t > bestT) {
        bestT = t;
        bestL = L;
        bestR = R;
      } else if (t == bestT && t >= 0) {
        int wBest = bestR - bestL, wCur = R - L;
        if (wCur < wBest ||
            (wCur == wBest && (L < bestL || (L == bestL && R < bestR)))) {
          bestL = L;
          bestR = R;
        }
      }
    }
  }
  if (bestL == -1)
    return nullptr;
  return make_unique<Result>(data.keys[bestL - 1], data.keys[bestR - 1], bestT);
}

unique_ptr<Result> queryBestBruteForce(const Data &data, int epsilon,
                                       double startVal, double endVal) {
  int n = data.n;
  if (n == 0)
    return nullptr;

  int L0 = lowerBound(data.keys, min(startVal, endVal));
  int R0 = upperBound(data.keys, max(startVal, endVal)) - 1;
  if (L0 < 0)
    L0 = 0;
  if (R0 >= n)
    R0 = n - 1;
  if (L0 > R0)
    return nullptr;

  double bestT = -1.0;
  int bestL = -1, bestR = -1;
  double alpha = 0.5, beta = 0.5;

  for (int L = 1; L <= n; L++) {
    for (int R = L; R <= n; R++) {
      bool fair = true;
      for (int d = 0; d < data.m; d++) {
        double diff = abs(data.points[R][d] - data.points[L - 1][d]);
        if (diff > epsilon) {
          fair = false;
          break;
        }
      }

      if (fair) {
        double t = tversky(L, R, L0 + 1, R0 + 1, alpha, beta);
        if (t > bestT) {
          bestT = t;
          bestL = L;
          bestR = R;
        } else if (t == bestT && t >= 0) {
          int wBest = bestR - bestL, wCur = R - L;
          if (wCur < wBest ||
              (wCur == wBest && (L < bestL || (L == bestL && R < bestR)))) {
            bestL = L;
            bestR = R;
          }
        }
      }
    }
  }

  if (bestL == -1)
    return nullptr;
  return make_unique<Result>(data.keys[bestL - 1], data.keys[bestR - 1], bestT);
}

// ===================== Data loading =====================

// ===================== Data loading =====================

Data readCSV(const string &filename, int expectedColors) {
  vector<RawRow> rawRows;
  set<int> colorSet;
  map<string, int> raceToInt;
  int nextColorId = 1;

  ifstream br(filename);
  if (!br.is_open()) {
    cerr << "Error: Could not open file " << filename << endl;
    exit(1);
  }

  string line;
  while (getline(br, line)) {
    // Trim whitespace
    line.erase(line.find_last_not_of(" \n\r\t") + 1);
    line.erase(0, line.find_first_not_of(" \n\r\t"));
    if (line.empty())
      continue;

    // Replace tabs and commas with spaces for easy parsing
    for (char &c : line) {
      if (c == ',' || c == '\t')
        c = ' ';
    }

    stringstream ss(line);
    vector<string> parts;
    string part;
    while (ss >> part)
      parts.push_back(part);

    // CHANGED: Require at least 3 columns for this new dataset
    if (parts.size() < 3)
      continue;

    try {
      double key = stod(parts[0]);
      // CHANGED: Use index 2 (the 3rd column) for the color/category
      string raceStr = parts[2];

      if (raceStr == "NA" || raceStr == "species" || raceStr == "race")
        continue;

      int color;
      if (raceToInt.count(raceStr)) {
        color = raceToInt[raceStr];
      } else {
        color = nextColorId++;
        raceToInt[raceStr] = color;
      }

      if (color >= 1 && color <= expectedColors) {
        rawRows.push_back({key, color});
        colorSet.insert(color);
      }
    } catch (const invalid_argument &) {
      // Silently ignore headers
    }
  }

  if (rawRows.empty()) {
    return Data(0, {}, 0, {}, {}, {}, {});
  }

  sort(rawRows.begin(), rawRows.end());

  vector<int> colors(colorSet.begin(), colorSet.end());
  vector<string> pairNames;
  vector<pair<int, int>> pairIndices;

  map<int, string> intToRace;
  for (const auto &entry : raceToInt) {
    intToRace[entry.second] = entry.first;
  }

  for (size_t i = 0; i < colors.size(); i++) {
    for (size_t j = i + 1; j < colors.size(); j++) {
      int cA = colors[i];
      int cB = colors[j];
      pairNames.push_back(intToRace[cA] + "-" + intToRace[cB]);
      pairIndices.push_back({cA, cB});
    }
  }

  int n = rawRows.size();
  int m = pairNames.size();
  vector<double> keys(n);
  vector<vector<int>> prefix(m, vector<int>(n + 1, 0));
  vector<vector<double>> points(n + 1, vector<double>(m, 0.0));
  vector<int> currentSums(m, 0);

  set<string> uniqueColorsStr;
  for (int c : colors) {
    uniqueColorsStr.insert(intToRace[c]);
  }

  for (int t = 0; t < n; t++) {
    const auto &row = rawRows[t];
    keys[t] = row.key;

    for (int d = 0; d < m; d++) {
      int cA = pairIndices[d].first;
      int cB = pairIndices[d].second;

      if (row.color == cA) {
        currentSums[d]++;
      } else if (row.color == cB) {
        currentSums[d]--;
      }

      prefix[d][t + 1] = currentSums[d];
      points[t + 1][d] = currentSums[d];
    }
  }

  return Data(n, keys, m, pairNames, uniqueColorsStr, prefix, points);
}

// ===================== Main Execution =====================

int main(int argc, char *argv[]) {
  if (argc < 3) {
    cout << "Usage: ./E1_Penguin <input.csv> <number_of_colors>\n";
    return 0;
  }

  string csvFile = argv[1];
  int numberOfColors = stoi(argv[2]);

  auto t0 = chrono::high_resolution_clock::now();
  Data data = readCSV(csvFile, numberOfColors);

  if (data.m == 0) {
    cout << "Not enough valid color rows found to generate pairwise columns.\n";
    return 0;
  }

  if (data.uniqueColors.size() != numberOfColors) {
    cout << "number_of_colors mismatch: file has " << data.uniqueColors.size()
         << " unique colors but arg was " << numberOfColors << "\n";
    return 0;
  }

  RangeTree tree(data.points, data.m);
  auto t1 = chrono::high_resolution_clock::now();
  long long durationTree =
      chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
  cout << "Tree built in " << durationTree << " ms\n";

  // Interactive query loop
  string line;
  while (true) {
    cout << "Enter x and range [start end] (enter -1 -1 -1 to exit):\n";
    if (!getline(cin, line))
      break;

    line.erase(line.find_last_not_of(" \n\r\t") + 1);
    line.erase(0, line.find_first_not_of(" \n\r\t"));
    if (line.empty())
      continue;

    stringstream ss(line);
    vector<string> parts;
    string part;
    while (ss >> part)
      parts.push_back(part);

    if (parts.size() < 3) {
      cout << "Please provide all three values: <epsilon> <start> <end>\n";
      continue;
    }

    int epsilon;
    double startVal, endVal;

    try {
      epsilon = stoi(parts[0]);
      startVal = stod(parts[1]);
      endVal = stod(parts[2]);
    } catch (const invalid_argument &) {
      cout << "Invalid format. Epsilon must be an integer, start and end must "
              "be numbers.\n";
      continue;
    }

    if (epsilon == -1 && startVal == -1.0 && endVal == -1.0)
      break;

    // --- Range Tree Query ---
    auto q0 = chrono::high_resolution_clock::now();
    auto resTree = queryBest(data, tree, epsilon, startVal, endVal);
    auto q1 = chrono::high_resolution_clock::now();
    long long dQ = chrono::duration_cast<chrono::milliseconds>(q1 - q0).count();

    // --- Brute Force Query ---
    auto b0 = chrono::high_resolution_clock::now();
    auto resBF = queryBestBruteForce(data, epsilon, startVal, endVal);
    auto b1 = chrono::high_resolution_clock::now();
    long long dB = chrono::duration_cast<chrono::milliseconds>(b1 - b0).count();

    // Print Tree results
    if (!resTree) {
      cout << "Tree Best range = [NA,NA] (similarity = 0.0)\n";
    } else {
      cout << "Tree Best range = [" << resTree->leftVal << ","
           << resTree->rightVal << "] (similarity = " << resTree->similarity
           << ")\n";
    }
    cout << "Tree Query executed in " << dQ << " ms\n\n";

    // Print Brute Force results
    if (!resBF) {
      cout << "BruteForce Best range = [NA,NA] (similarity = 0.0)\n";
    } else {
      cout << "BruteForce Best range = [" << resBF->leftVal << ","
           << resBF->rightVal << "] (similarity = " << resBF->similarity
           << ")\n";
    }
    cout << "BruteForce Query executed in " << dB << " ms\n\n";
  }

  return 0;
}