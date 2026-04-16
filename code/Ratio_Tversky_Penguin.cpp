#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// --- Data Structures ---

struct DatasetPoint {
  double value;
  std::string color;
  int id;
  DatasetPoint(double v, std::string c, int i) : value(v), color(c), id(i) {}
};

struct Point {
  std::vector<double> coords;
  int original_index;

  Point(double cr, double cb, int index) : original_index(index) {
    coords.push_back(cr);
    coords.push_back(cb);
    coords.push_back(static_cast<double>(index));
  }
};

struct RangeTreeNode;

struct RangeTreeNode {
  int level;
  double split_val;
  int subtree_size;
  RangeTreeNode *left = nullptr;
  RangeTreeNode *right = nullptr;
  RangeTreeNode *next_tree = nullptr;
  Point *curr_point = nullptr;
  Point *min_point = nullptr;

  RangeTreeNode(int l, double sv, int ss, RangeTreeNode *lt, RangeTreeNode *rt,
                RangeTreeNode *nt, Point *cp, Point *mp)
      : level(l), split_val(sv), subtree_size(ss), left(lt), right(rt),
        next_tree(nt), curr_point(cp), min_point(mp) {}
};

struct QueryBox {
  std::vector<std::pair<double, double>> bounds;
};

struct NeighborTracker {
  std::vector<Point *> predecessors;
  std::vector<Point *> successors;
  const int q_j;
  const size_t k = 2;

  NeighborTracker(int query_j) : q_j(query_j) {}

  void consider(Point *p) {
    if (p->original_index <= q_j) {
      predecessors.push_back(p);
      std::sort(predecessors.begin(), predecessors.end(),
                [](Point *a, Point *b) {
                  return a->original_index > b->original_index;
                });
      if (predecessors.size() > k) {
        predecessors.pop_back();
      }
    } else {
      successors.push_back(p);
      std::sort(successors.begin(), successors.end(), [](Point *a, Point *b) {
        return a->original_index < b->original_index;
      });
      if (successors.size() > k) {
        successors.pop_back();
      }
    }
  }

  std::vector<Point *> get_results() {
    std::vector<Point *> results;
    results.insert(results.end(), predecessors.begin(), predecessors.end());
    results.insert(results.end(), successors.begin(), successors.end());
    return results;
  }
};

// --- Helper Functions ---

// *** MODIFIED: Custom parser for sorted_penguins.csv (Value -> String Species)
// ***
std::vector<DatasetPoint> read_custom_dataset(const std::string &filename) {
  std::vector<DatasetPoint> dataset;
  std::ifstream file_handler(filename);

  if (!file_handler.is_open()) {
    std::cerr << "Error: Failed to open file " << filename << std::endl;
    return dataset;
  }

  std::string line;

  // Skip the header line ("bill_length_mm    species")
  std::getline(file_handler, line);

  while (std::getline(file_handler, line)) {
    if (line.empty())
      continue;

    // Convert any commas to spaces to handle both CSV and space/tab-separated
    std::replace(line.begin(), line.end(), ',', ' ');

    std::stringstream ss(line);
    double value;
    std::string species;

    // Read: [Value] [Species String]
    if (ss >> value >> species) {
      // *** MODIFIED: If species is Adelie put red, otherwise blue ***
      std::string color_str = (species == "Adelie") ? "red" : "blue";
      dataset.emplace_back(value, color_str, 0);
    }
  }

  // The algorithm requires data to be sorted in ascending order by value
  std::sort(dataset.begin(), dataset.end(),
            [](const DatasetPoint &a, const DatasetPoint &b) {
              return a.value < b.value;
            });

  // Re-assign correct sequential IDs after sorting
  for (size_t i = 0; i < dataset.size(); ++i) {
    dataset[i].id = static_cast<int>(i);
  }

  return dataset;
}

void delete_tree(RangeTreeNode *node) {
  if (!node)
    return;
  delete_tree(node->left);
  delete_tree(node->right);
  delete_tree(node->next_tree);
  delete node;
}

bool min_all(Point *p1, Point *p2) {
  if (!p1)
    return false;
  if (!p2)
    return true;
  for (size_t i = 0; i < p1->coords.size(); ++i) {
    if (p1->coords[i] < p2->coords[i])
      return true;
    if (p1->coords[i] > p2->coords[i])
      return false;
  }
  return false;
}

bool inRangeQ(Point *p, const QueryBox &Q, int dims_to_check) {
  for (int i = 0; i < dims_to_check; ++i) {
    if (p->coords[i] < Q.bounds[i].first || p->coords[i] > Q.bounds[i].second) {
      return false;
    }
  }
  return true;
}

// --- Tree Building and Querying Functions ---

RangeTreeNode *build_tree(std::vector<Point *> &pts, int level,
                          int total_levels) {
  if (pts.empty() || level >= total_levels)
    return nullptr;

  int dim = level;
  sort(pts.begin(), pts.end(),
       [&](Point *a, Point *b) { return a->coords[dim] < b->coords[dim]; });

  int mid = pts.size() / 2;
  Point *curr = pts[mid];
  double split_val = curr->coords[dim];

  std::vector<Point *> left_pts(pts.begin(), pts.begin() + mid);
  std::vector<Point *> right_pts(pts.begin() + mid + 1, pts.end());

  RangeTreeNode *left = build_tree(left_pts, level, total_levels);
  RangeTreeNode *right = build_tree(right_pts, level, total_levels);
  RangeTreeNode *next = build_tree(pts, level + 1, total_levels);

  Point *min_p = curr;
  if (left && min_all(left->min_point, min_p))
    min_p = left->min_point;
  if (right && min_all(right->min_point, min_p))
    min_p = right->min_point;
  if (next && min_all(next->min_point, min_p))
    min_p = next->min_point;

  return new RangeTreeNode(level, split_val, (int)(pts.size()), left, right,
                           next, curr, min_p);
}

void find_closest_fair_js(RangeTreeNode *node, const QueryBox &Q, int q_j,
                          int level, int total_levels,
                          NeighborTracker &tracker) {
  if (!node)
    return;

  if (level == total_levels - 1) {
    if (inRangeQ(node->curr_point, Q, total_levels - 1)) {
      tracker.consider(node->curr_point);
    }

    if (static_cast<double>(q_j) < node->split_val) {
      find_closest_fair_js(node->left, Q, q_j, level, total_levels, tracker);
      if (tracker.successors.size() < tracker.k) {
        find_closest_fair_js(node->right, Q, q_j, level, total_levels, tracker);
      }
    } else {
      find_closest_fair_js(node->right, Q, q_j, level, total_levels, tracker);
      if (tracker.predecessors.size() < tracker.k) {
        find_closest_fair_js(node->left, Q, q_j, level, total_levels, tracker);
      }
    }
    return;
  }

  double high = Q.bounds[level].second;
  if (high < node->split_val) {
    find_closest_fair_js(node->left, Q, q_j, level, total_levels, tracker);
  } else {
    if (node->left && node->left->next_tree) {
      find_closest_fair_js(node->left->next_tree, Q, q_j, level + 1,
                           total_levels, tracker);
    }

    if (inRangeQ(node->curr_point, Q, total_levels - 1)) {
      tracker.consider(node->curr_point);
    }

    find_closest_fair_js(node->right, Q, q_j, level, total_levels, tracker);
  }
}

double calculate_tversky_index(int start1, int end1, int start2, int end2,
                               double alpha = 0.5, double beta = 0.5) {
  int intersection_start = std::max(start1, start2);
  int intersection_end = std::min(end1, end2);
  double intersection_size =
      std::max(0, intersection_end - intersection_start + 1);

  double size1 = end1 - start1 + 1;
  double size2 = end2 - start2 + 1;

  double x_minus_y = size1 - intersection_size;
  double y_minus_x = size2 - intersection_size;

  double denominator =
      intersection_size + (alpha * x_minus_y) + (beta * y_minus_x);

  if (denominator == 0)
    return 1.0;
  return intersection_size / denominator;
}

void find_best_fair_range_brute_force(
    int query_i, int query_j, double epsilon,
    const std::vector<double> &cumulative_red,
    const std::vector<double> &cumulative_blue) {
  int best_i = -1;
  int best_j = -1;
  double max_similarity = -1.0;
  const int n = cumulative_red.size();
  double lower_bound = 1.0 / (1.0 + epsilon);
  double upper_bound = 1.0 + epsilon;

  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      double red_score =
          cumulative_red[j] - (i > 0 ? cumulative_red[i - 1] : 0.0);
      double blue_score =
          cumulative_blue[j] - (i > 0 ? cumulative_blue[i - 1] : 0.0);

      bool is_fair = false;
      if (blue_score == 0.0) {
        if (red_score == 0.0)
          is_fair = true;
      } else {
        double ratio = red_score / blue_score;
        if (ratio >= lower_bound && ratio <= upper_bound) {
          is_fair = true;
        }
      }

      if (is_fair) {
        double similarity = calculate_tversky_index(query_i, query_j, i, j);
        if (similarity > max_similarity) {
          max_similarity = similarity;
          best_i = i;
          best_j = j;
        }
      }
    }
  }

  std::cout << "\n--- Brute-Force Results ---" << std::endl;
  if (best_j != -1) {
    std::cout << "Best Fair Range Found Index: [" << best_i << ", " << best_j
              << "]" << std::endl;
    std::cout << "Tversky Index: " << max_similarity << std::endl;
  } else {
    std::cout << "No fair range could be found using the brute-force method."
              << std::endl;
  }
}

// --- Main Application Logic ---

int main() {
  const int DIMS = 3;

  // *** TARGET THE PENGUINS DATASET ***
  std::string filename = "../dataset/penguin/sorted_penguins.csv";
  std::vector<DatasetPoint> dataset = read_custom_dataset(filename);

  if (dataset.empty()) {
    std::cerr << "Dataset could not be loaded or is empty. Exiting."
              << std::endl;
    return 1;
  }
  const int DATASET_SIZE = dataset.size();
  std::cout << "Successfully loaded and sorted " << DATASET_SIZE
            << " records from " << filename << std::endl;

  // Define explicit weights (Adjust these back to 1.0 / 1.0 if you don't want
  // weighting)
  const double WEIGHT_RED = 1.0;
  const double WEIGHT_BLUE = 1.0;

  std::vector<double> cumulative_red(DATASET_SIZE, 0.0);
  std::vector<double> cumulative_blue(DATASET_SIZE, 0.0);

  for (int i = 0; i < DATASET_SIZE; ++i) {
    double prev_red = (i > 0) ? cumulative_red[i - 1] : 0.0;
    double prev_blue = (i > 0) ? cumulative_blue[i - 1] : 0.0;

    if (dataset[i].color == "red") {
      cumulative_red[i] = prev_red + WEIGHT_RED;
      cumulative_blue[i] = prev_blue;
    } else {
      cumulative_red[i] = prev_red;
      cumulative_blue[i] = prev_blue + WEIGHT_BLUE;
    }
  }

  double epsilon = 0.1;
  std::cout << "--- Fairness Range Optimizer ---" << std::endl;

  // *** MODIFIED: Bounds reflecting the bill_length_mm values ***
  double start_val = 40.0;
  double end_val = 80.0;

  std::cout << "\nSearching for query range corresponding to values between "
            << start_val << " and " << end_val << std::endl;
  int query_i = -1;
  int query_j = -1;

  for (int i = 0; i < DATASET_SIZE; ++i) {
    if (dataset[i].value >= start_val && dataset[i].value <= end_val) {
      if (query_i == -1) {
        query_i = i;
      }
      query_j = i;
    }
  }

  if (query_i == -1) {
    std::cerr << "Error: No data points found in the specified range. Exiting."
              << std::endl;
    return 1;
  }

  std::cout << "\nInitial Query Range (by index) is [" << query_i << ", "
            << query_j << "]" << std::endl;
  std::cout << "Initial Query Range (by values) is [" << dataset[query_i].value
            << ", " << dataset[query_j].value << "]" << std::endl;

  std::vector<Point> points_data;
  points_data.reserve(DATASET_SIZE);

  double one_plus_epsilon = 1.0 + epsilon;
  for (int i = 0; i < DATASET_SIZE; ++i) {
    double cr = cumulative_red[i] - cumulative_blue[i] * one_plus_epsilon;
    double cb = cumulative_blue[i] - cumulative_red[i] * one_plus_epsilon;
    points_data.emplace_back(cr, cb, i);
  }

  std::vector<Point *> points_ptrs;
  points_ptrs.reserve(DATASET_SIZE);
  for (auto &p : points_data) {
    points_ptrs.push_back(&p);
  }

  auto build_start = std::chrono::high_resolution_clock::now();
  std::cout << "\nBuilding 3D Layered Range Tree..." << std::endl;
  RangeTreeNode *root = build_tree(points_ptrs, 0, DIMS);
  auto build_end = std::chrono::high_resolution_clock::now();
  auto build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      build_end - build_start);
  std::cout << "Tree built in " << build_duration.count() << "ms." << std::endl;

  auto tree_query_start = std::chrono::high_resolution_clock::now();

  int best_i_tree = -1;
  int best_j_tree = -1;
  double max_similarity_tree = -1.0;

  for (int i = 0; i < DATASET_SIZE; ++i) {
    // Correct baseline for index 0 to ensure Tree and Brute-Force match
    double max_cr = (i > 0) ? points_data[i - 1].coords[0] : 0.0;
    double max_cb = (i > 0) ? points_data[i - 1].coords[1] : 0.0;

    QueryBox Q;
    Q.bounds.push_back({-std::numeric_limits<double>::max(), max_cr});
    Q.bounds.push_back({-std::numeric_limits<double>::max(), max_cb});

    NeighborTracker tracker(query_j);
    find_closest_fair_js(root, Q, query_j, 0, DIMS, tracker);
    std::vector<Point *> candidates = tracker.get_results();

    for (const auto &candidate_point : candidates) {
      int j = candidate_point->original_index;
      if (j >= i) {
        double similarity = calculate_tversky_index(query_i, query_j, i, j);
        if (similarity > max_similarity_tree) {
          max_similarity_tree = similarity;
          best_i_tree = i;
          best_j_tree = j;
        }
      }
    }
  }
  auto tree_query_end = std::chrono::high_resolution_clock::now();
  auto tree_query_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(tree_query_end -
                                                            tree_query_start);

  std::cout << "\n--- 3D Range-Tree Results ---" << std::endl;
  if (best_j_tree != -1) {
    std::cout << "Best Fair Range Index Found: [" << best_i_tree << ", "
              << best_j_tree << "]" << std::endl;
    std::cout << "Tversky Index: " << max_similarity_tree << std::endl;
    std::cout << "Corresponding values: [" << dataset[best_i_tree].value << ", "
              << dataset[best_j_tree].value << "]" << std::endl;
  } else {
    std::cout << "No fair range could be found." << std::endl;
  }
  std::cout << "Optimized 3D search took: " << tree_query_duration.count()
            << " ms" << std::endl;

  delete_tree(root);
  root = nullptr;

  auto brute_start = std::chrono::high_resolution_clock::now();
  find_best_fair_range_brute_force(query_i, query_j, epsilon, cumulative_red,
                                   cumulative_blue);
  auto brute_end = std::chrono::high_resolution_clock::now();
  auto brute_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      brute_end - brute_start);
  std::cout << "Brute-force search took: " << brute_duration.count() << " ms"
            << std::endl;

  return 0;
}