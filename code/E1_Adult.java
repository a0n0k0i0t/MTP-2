import java.io.*;
import java.util.*;

/**
 * E1.java — Interactive range search using an axis-aligned "range tree" over
 * m-dimensional prefix vectors built from pairwise cumulative columns.
 *
 * Usage:
 * java E1 <input.csv> <number_of_colors>
 */
public class E1_Adult {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.out.println("Usage: java E1 <input.csv> <number_of_colors>");
            return;
        }

        final String csvFile = args[0];
        final int numberOfColors = Integer.parseInt(args[1]);

        long t0 = System.nanoTime();

        // Pass the expected number of colors to the reader to filter out garbage data
        Data data = readCSV(csvFile, numberOfColors);

        if (data.m == 0) {
            System.out.println("Not enough valid color rows found to generate pairwise columns.");
            return;
        }

        if (data.uniqueColors.size() != numberOfColors) {
            System.out.println("number_of_colors mismatch: file has " + data.uniqueColors.size() +
                    " unique colors (Found: " + data.uniqueColors + ") but arg was " + numberOfColors);
            return;
        }

        RangeTree tree = new RangeTree(data.points, data.m);
        long t1 = System.nanoTime();
        System.out.println("Tree built in " + ((t1 - t0) / 1_000_000) + " ms");

        // Interactive query loop
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        while (true) {
            System.out.print("Enter x and range [start end] (enter -1 -1 -1 to exit):\n");
            String line = br.readLine();
            if (line == null)
                break;
            line = line.trim();
            if (line.isEmpty())
                continue;
            String[] parts = line.split("\\s+");
            if (parts.length < 3) {
                System.out.println("Please provide all three values: <epsilon> <start> <end>");
                continue;
            }

            int epsilon;
            double startVal = 0.0, endVal = 0.0;

            try {
                // Epsilon is an integer, but start and end must be parsed as doubles
                epsilon = Integer.parseInt(parts[0]);
                startVal = Double.parseDouble(parts[1]);
                endVal = Double.parseDouble(parts[2]);
            } catch (NumberFormatException ex) {
                System.out.println("Invalid format. Epsilon must be an integer, start and end must be numbers.");
                continue;
            }

            if (epsilon == -1 && startVal == -1.0 && endVal == -1.0)
                break;

            // --- Range Tree Query ---
            long q0 = System.nanoTime();
            Result resTree = queryBest(data, tree, epsilon, startVal, endVal);
            long q1 = System.nanoTime();

            // --- Brute Force Query ---
            long b0 = System.nanoTime();
            Result resBF = queryBestBruteForce(data, epsilon, startVal, endVal);
            long b1 = System.nanoTime();

            // Print Tree results
            if (resTree == null) {
                System.out.println("Tree Best range = [NA,NA] (similarity = 0.0)");
            } else {
                System.out.println("Tree Best range = [" + resTree.leftVal + "," + resTree.rightVal + "] (similarity = "
                        + resTree.similarity + ")");
            }
            System.out.println("Tree Query executed in " + ((q1 - q0) / 1_000_000) + " ms\n");

            // Print Brute Force results
            if (resBF == null) {
                System.out.println("BruteForce Best range = [NA,NA] (similarity = 0.0)");
            } else {
                System.out.println("BruteForce Best range = [" + resBF.leftVal + "," + resBF.rightVal
                        + "] (similarity = " + resBF.similarity + ")");
            }
            System.out.println("BruteForce Query executed in " + ((b1 - b0) / 1_000_000) + " ms\n");
        }
    }

    // ===================== Query logic =====================

    static final class Result {
        final double leftVal, rightVal;
        final double similarity;

        Result(double l, double r, double s) {
            leftVal = l;
            rightVal = r;
            similarity = s;
        }
    }

    static Result queryBest(Data data, RangeTree tree, int epsilon, double startVal, double endVal) {
        int n = data.n;
        if (n == 0)
            return null;

        int L0 = lowerBound(data.keys, Math.min(startVal, endVal));
        int R0 = upperBound(data.keys, Math.max(startVal, endVal)) - 1;
        if (L0 < 0)
            L0 = 0;
        if (R0 >= n)
            R0 = n - 1;
        if (L0 > R0)
            return null;

        double bestT = -1.0;
        int bestL = -1, bestR = -1;

        double alpha = 0.5;
        double beta = 0.5;

        for (int p = 0; p <= n - 1; p++) {
            double[] center = data.points[p];
            double[] lo = new double[data.m];
            double[] hi = new double[data.m];
            for (int d = 0; d < data.m; d++) {
                lo[d] = center[d] - epsilon;
                hi[d] = center[d] + epsilon;
            }

            List<Integer> hits = new ArrayList<>();
            tree.rangeQuery(lo, hi, hits);

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
                    if (wCur < wBest || (wCur == wBest && (L < bestL || (L == bestL && R < bestR)))) {
                        bestL = L;
                        bestR = R;
                    }
                }
            }
        }
        if (bestL == -1)
            return null;

        return new Result(data.keys[bestL - 1], data.keys[bestR - 1], bestT);
    }

    // --- BRUTE FORCE METHOD ---
    static Result queryBestBruteForce(Data data, int epsilon, double startVal, double endVal) {
        int n = data.n;
        if (n == 0)
            return null;

        int L0 = lowerBound(data.keys, Math.min(startVal, endVal));
        int R0 = upperBound(data.keys, Math.max(startVal, endVal)) - 1;
        if (L0 < 0)
            L0 = 0;
        if (R0 >= n)
            R0 = n - 1;
        if (L0 > R0)
            return null;

        double bestT = -1.0;
        int bestL = -1, bestR = -1;

        double alpha = 0.5;
        double beta = 0.5;

        for (int L = 1; L <= n; L++) {
            for (int R = L; R <= n; R++) {
                boolean fair = true;

                for (int d = 0; d < data.m; d++) {
                    double diff = Math.abs(data.points[R][d] - data.points[L - 1][d]);
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
                        if (wCur < wBest || (wCur == wBest && (L < bestL || (L == bestL && R < bestR)))) {
                            bestL = L;
                            bestR = R;
                        }
                    }
                }
            }
        }

        if (bestL == -1)
            return null;
        return new Result(data.keys[bestL - 1], data.keys[bestR - 1], bestT);
    }

    // ===================== Data loading =====================

    static final class Data {
        final int n;
        final double[] keys;
        final int m;
        final List<String> pairNames;
        final Set<String> uniqueColors;
        final int[][] prefix;
        final double[][] points;

        Data(int n, double[] keys, int m, List<String> pairNames, Set<String> uniqueColors,
                int[][] prefix, double[][] points) {
            this.n = n;
            this.keys = keys;
            this.m = m;
            this.pairNames = pairNames;
            this.uniqueColors = uniqueColors;
            this.prefix = prefix;
            this.points = points;
        }
    }

    static class RawRow implements Comparable<RawRow> {
        double key;
        int color;

        RawRow(double key, int color) {
            this.key = key;
            this.color = color;
        }

        @Override
        public int compareTo(RawRow other) {
            return Double.compare(this.key, other.key);
        }
    }

    static Data readCSV(String filename, int expectedColors) throws IOException {
        List<RawRow> rawRows = new ArrayList<>();
        Set<Integer> colorSet = new TreeSet<>();

        // Dynamically map race strings (e.g. "White", "Black") to internal integer IDs
        Map<String, Integer> raceToInt = new LinkedHashMap<>();
        int nextColorId = 1;

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty())
                    continue;

                // Split by comma for standard CSV parsing
                String[] parts = line.split(",");
                if (parts.length < 3)
                    continue;

                try {
                    // Index 0 is age (double)
                    double key = Double.parseDouble(parts[0].trim());
                    // Index 2 is race (String)
                    String raceStr = parts[2].trim();

                    // Skip the 'NA' string values generated in previous steps or headers
                    if (raceStr.equals("NA") || raceStr.equalsIgnoreCase("race"))
                        continue;

                    int color;
                    if (raceToInt.containsKey(raceStr)) {
                        color = raceToInt.get(raceStr);
                    } else {
                        color = nextColorId++;
                        raceToInt.put(raceStr, color);
                    }

                    // Only process up to the expected number of colors/races
                    if (color >= 1 && color <= expectedColors) {
                        rawRows.add(new RawRow(key, color));
                        colorSet.add(color);
                    }
                } catch (NumberFormatException e) {
                    // Silently ignore headers or completely malformed lines
                }
            }
        }

        if (rawRows.isEmpty()) {
            return new Data(0, new double[0], 0, new ArrayList<>(), new LinkedHashSet<>(), new int[0][0],
                    new double[0][0]);
        }

        Collections.sort(rawRows);

        List<Integer> colors = new ArrayList<>(colorSet);
        List<String> pairNames = new ArrayList<>();
        List<int[]> pairIndices = new ArrayList<>();

        // Create reverse lookup to reconstruct the pair names (e.g. "White-Black")
        Map<Integer, String> intToRace = new HashMap<>();
        for (Map.Entry<String, Integer> entry : raceToInt.entrySet()) {
            intToRace.put(entry.getValue(), entry.getKey());
        }

        for (int i = 0; i < colors.size(); i++) {
            for (int j = i + 1; j < colors.size(); j++) {
                int cA = colors.get(i);
                int cB = colors.get(j);
                pairNames.add(intToRace.get(cA) + "-" + intToRace.get(cB));
                pairIndices.add(new int[] { cA, cB });
            }
        }

        int n = rawRows.size();
        int m = pairNames.size();
        double[] keys = new double[n];

        int[][] prefix = new int[m][n + 1];
        double[][] points = new double[n + 1][m];
        int[] currentSums = new int[m];

        Set<String> uniqueColorsStr = new LinkedHashSet<>();
        for (int c : colors) {
            uniqueColorsStr.add(intToRace.get(c));
        }

        for (int t = 0; t < n; t++) {
            RawRow row = rawRows.get(t);
            keys[t] = row.key;

            for (int d = 0; d < m; d++) {
                int cA = pairIndices.get(d)[0];
                int cB = pairIndices.get(d)[1];

                if (row.color == cA) {
                    currentSums[d]++;
                } else if (row.color == cB) {
                    currentSums[d]--;
                }

                prefix[d][t + 1] = currentSums[d];
                points[t + 1][d] = currentSums[d];
            }
        }

        return new Data(n, keys, m, pairNames, uniqueColorsStr, prefix, points);
    }

    // ===================== "Range tree" =====================

    static final class RangeTree {
        final int dims;
        final Node root;
        final double[][] P;

        RangeTree(double[][] points, int dims) {
            this.dims = dims;
            this.P = points;
            int n = points.length;
            int[] all = new int[n];
            for (int i = 0; i < n; i++)
                all[i] = i;
            this.root = build(all, 0, n, 0);
        }

        final class Node {
            double[] lo, hi;
            int dim;
            double split;
            Node left, right;
            int[] idxs;
            boolean isLeaf;
        }

        Node build(int[] idx, int loi, int hii, int depth) {
            int n = hii - loi;
            Node node = new Node();
            node.lo = new double[dims];
            node.hi = new double[dims];
            Arrays.fill(node.lo, Double.POSITIVE_INFINITY);
            Arrays.fill(node.hi, Double.NEGATIVE_INFINITY);
            for (int k = loi; k < hii; k++) {
                double[] pt = P[idx[k]];
                for (int d = 0; d < dims; d++) {
                    if (pt[d] < node.lo[d])
                        node.lo[d] = pt[d];
                    if (pt[d] > node.hi[d])
                        node.hi[d] = pt[d];
                }
            }
            if (n <= 32) {
                node.isLeaf = true;
                node.idxs = Arrays.copyOfRange(idx, loi, hii);
                return node;
            }
            int dim = depth % dims;
            int mid = loi + n / 2;
            nthElement(idx, loi, mid, hii, dim);
            node.dim = dim;
            node.split = P[idx[mid]][dim];
            node.left = build(idx, loi, mid, depth + 1);
            node.right = build(idx, mid, hii, depth + 1);
            return node;
        }

        void nthElement(int[] a, int lo, int mid, int hi, int dim) {
            int l = lo, r = hi - 1;
            while (true) {
                int i = l, j = r;
                double pivot = P[a[(l + r) >>> 1]][dim];
                while (i <= j) {
                    while (P[a[i]][dim] < pivot)
                        i++;
                    while (P[a[j]][dim] > pivot)
                        j--;
                    if (i <= j) {
                        int t = a[i];
                        a[i] = a[j];
                        a[j] = t;
                        i++;
                        j--;
                    }
                }
                if (j < mid)
                    l = i;
                else
                    r = j;
                if (l >= mid && r <= mid)
                    return;
            }
        }

        void rangeQuery(double[] lo, double[] hi, List<Integer> out) {
            rangeQuery(root, lo, hi, out);
        }

        void rangeQuery(Node node, double[] lo, double[] hi, List<Integer> out) {
            if (node == null)
                return;
            if (!overlaps(node, lo, hi))
                return;
            if (node.isLeaf) {
                for (int id : node.idxs)
                    if (contains(P[id], lo, hi))
                        out.add(id);
                return;
            }
            rangeQuery(node.left, lo, hi, out);
            rangeQuery(node.right, lo, hi, out);
        }

        boolean overlaps(Node node, double[] qlo, double[] qhi) {
            for (int d = 0; d < dims; d++)
                if (node.hi[d] < qlo[d] || node.lo[d] > qhi[d])
                    return false;
            return true;
        }

        boolean contains(double[] pt, double[] qlo, double[] qhi) {
            for (int d = 0; d < dims; d++)
                if (pt[d] < qlo[d] || pt[d] > qhi[d])
                    return false;
            return true;
        }
    }

    // ===================== Math & utils =====================

    static int lowerBound(double[] a, double x) {
        int lo = 0, hi = a.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (a[mid] < x)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }

    static int upperBound(double[] a, double x) {
        int lo = 0, hi = a.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (a[mid] <= x)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }

    static double tversky(int aL, int aR, int bL, int bR, double alpha, double beta) {
        int inter = intersectionSize(aL, aR, bL, bR);

        int sizeA = (aR - aL + 1);
        int sizeB = (bR - bL + 1);

        int aMinusB = sizeA - inter;
        int bMinusA = sizeB - inter;

        double denom = inter + (alpha * aMinusB) + (beta * bMinusA);
        return (denom == 0.0) ? 0.0 : ((double) inter) / denom;
    }

    static int intersectionSize(int aL, int aR, int bL, int bR) {
        int L = Math.max(aL, bL), R = Math.min(aR, bR);
        return (L <= R) ? (R - L + 1) : 0;
    }
}