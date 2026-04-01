import gurobipy as gp
from gurobipy import GRB
import time
import sys
import traceback

def build_model(points, input_box_indices, Wr, Wb, epsilon, d=2):
    n = len(points)
    
    # --- Coordinate compression ---
    sorted_coords = []
    for k in range(d):
        vals = sorted(set(p['coords'][k] for p in points))
        sorted_coords.append(vals)

    coord_maps = [{v: i for i, v in enumerate(vals)} for vals in sorted_coords]

    point_ranks = [[0]*d for _ in range(n)]
    for i in range(n):
        for k in range(d):
            point_ranks[i][k] = coord_maps[k][points[i]['coords'][k]]

    # --- Model ---
    model = gp.Model("FairRangeFeasibility")
    model.setParam("OutputFlag", 0)

    # --- Boundary variables ---
    l_vars, r_vars, pos_L, pos_R = [], [], [], []
    for k in range(d):
        m = len(sorted_coords[k])
        l = model.addVars(m, vtype=GRB.BINARY, name=f"L_{k}")
        r = model.addVars(m, vtype=GRB.BINARY, name=f"R_{k}")
        l_vars.append(l)
        r_vars.append(r)

        model.addConstr(l.sum() == 1)
        model.addConstr(r.sum() == 1)

        posL_expr = gp.quicksum(j * l[j] for j in range(m))
        posR_expr = gp.quicksum(j * r[j] for j in range(m))
        
        posL_var = model.addVar(vtype=GRB.INTEGER, name=f"posL_val_{k}")
        posR_var = model.addVar(vtype=GRB.INTEGER, name=f"posR_val_{k}")
        
        model.addConstr(posL_var == posL_expr)
        model.addConstr(posR_var == posR_expr)
        model.addConstr(posL_var <= posR_var)

        pos_L.append(posL_var)
        pos_R.append(posR_var)

    # --- Output indicator ---
    O = model.addVars(n, vtype=GRB.BINARY, name="O")

    # --- Dimension satisfaction indicators ---
    Z = {}
    M = max(len(vals) for vals in sorted_coords)

    for i in range(n):
        for k in range(d):
            Z[i, k] = model.addVar(vtype=GRB.BINARY, name=f"Z_{i}_{k}")

            r_ik = point_ranks[i][k]
            model.addConstr(r_ik >= pos_L[k] - M * (1 - Z[i, k]))
            model.addConstr(r_ik <= pos_R[k] + M * (1 - Z[i, k]))

            y_L_ik = model.addVar(vtype=GRB.BINARY, name=f"yL_{i}_{k}")
            y_R_ik = model.addVar(vtype=GRB.BINARY, name=f"yR_{i}_{k}")

            model.addConstr(pos_L[k] >= r_ik + 1 - M * (1 - y_L_ik))
            model.addConstr(pos_R[k] <= r_ik - 1 + M * (1 - y_R_ik))
            model.addConstr(y_L_ik + y_R_ik + Z[i, k] >= 1)

            model.addConstr(O[i] <= Z[i, k])

        model.addConstr(O[i] >= gp.quicksum(Z[i, k] for k in range(d)) - (d - 1))

    # --- Similarity bookkeeping ---
    S_ino = gp.LinExpr()
    S_io = gp.LinExpr()
    S_oi = gp.LinExpr()

    count_R = gp.LinExpr()
    count_B = gp.LinExpr()

    for i in range(n):
        in_I = (i in input_box_indices)

        if in_I:
            S_ino += O[i]
            S_io += (1 - O[i])
        else:
            S_oi += O[i]

        if points[i]['color_id'] == 1:
            count_R += O[i]
        else:
            count_B += O[i]

    # --- Fairness ---
    diff = Wb * count_B - Wr * count_R
    model.addConstr(diff <= epsilon)
    model.addConstr(diff >= -epsilon)
    
    return model, S_ino, S_io, S_oi, l_vars, r_vars, sorted_coords


def solve_fair_range_binary_search_warm_start(points, input_box_indices, alpha, beta, Wr, Wb, epsilon, d=2):
    """
    Performs binary search on gamma to find the maximum gamma for which a feasible box exists.
    Uses warm-starting by maintaining the Gurobi model across iterations.
    Precision: 2 decimal points.
    """
    print("Building ILP Model once for Warm Start...")
    model, S_ino, S_io, S_oi, l_vars, r_vars, sorted_coords = build_model(
        points, input_box_indices, Wr, Wb, epsilon, d)

    low = 0.0
    high = 1.0
    best_gamma = 0.0
    best_box = None
    similarity_constraint = None
    
    print("Starting Binary Search for Gamma...")
    # Precision limit: 0.005 for approx 2 decimal points stability
    while high - low > 0.005: 
        mid = (low + high) / 2
        
        # Remove the constraint from the previous iteration if it exists
        if similarity_constraint is not None:
             model.remove(similarity_constraint)
             
        # Formulate the new LHS with the updated 'mid' value
        lhs = (1 - mid) * S_ino - mid * alpha * S_io - mid * beta * S_oi
        
        # Add the brand new constraint and store its reference
        similarity_constraint = model.addConstr(lhs >= 0, name="SimilarityConstraint")
        
        # Set Objective to 0 for feasibility check
        model.setObjective(0, GRB.MAXIMIZE)
        
        # Inject MIP Start if we have a best_box (helps when mid is lower than best_box's physical gamma)
        if best_box is not None:
             for k in range(d):
                  for j in range(len(sorted_coords[k])):
                       l_vars[k][j].Start = GRB.UNDEFINED
                       r_vars[k][j].Start = GRB.UNDEFINED
                  
                  l_val = best_box[k][0]
                  r_val = best_box[k][1]
                  
                  if l_val in sorted_coords[k]:
                      l_idx = sorted_coords[k].index(l_val)
                      l_vars[k][l_idx].Start = 1
                      
                  if r_val in sorted_coords[k]:
                      r_idx = sorted_coords[k].index(r_val)
                      r_vars[k][r_idx].Start = 1

        model.optimize()
        
        if model.Status == GRB.OPTIMAL:
            # Extract solution
            new_box = []
            for k in range(d):
                l_idx = -1
                r_idx = -1
                for j in range(len(sorted_coords[k])):
                     if l_vars[k][j].X > 0.5:
                         l_idx = j
                     if r_vars[k][j].X > 0.5:
                         r_idx = j
                
                if l_idx == -1: l_idx = 0
                if r_idx == -1: r_idx = len(sorted_coords[k]) - 1
                
                new_box.append((sorted_coords[k][l_idx], sorted_coords[k][r_idx]))
                
            best_gamma = mid
            best_box = new_box
            low = mid
            print(f"  Gamma {mid:.4f} is FEASIBLE.")
        else:
            high = mid
            print(f"  Gamma {mid:.4f} is INFEASIBLE.")
            
    return best_gamma, best_box

# --- File Parsing ---
def read_input_file(filename="points.txt"):
    """
    Reads input in the format specified by bfsmp_fread_copy.cpp
    """
    try:
        with open(filename, 'r') as f:
            tokens = f.read().split()
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None, None, None, None
        
    iterator = iter(tokens)
    try:
        n = int(next(iterator))
        d = int(next(iterator))
        t = int(next(iterator))
        
        points = []
        # Read Points Coords
        for i in range(n):
            coords = []
            for _ in range(d + t):
                coords.append(float(next(iterator)))
            points.append({'coords': coords, 'index': i})
            
        # Read Colors
        for i in range(n):
            points[i]['color_id'] = int(next(iterator))
            
        # Read Query Box
        qbox_bounds = []
        for _ in range(d):
            min_v = float(next(iterator))
            max_v = float(next(iterator))
            qbox_bounds.append((min_v, max_v))
            
        return points, d, t, qbox_bounds
        
    except StopIteration:
        print("Error: Unexpected end of file.")
        return None, None, None, None
    except ValueError:
        print("Error: Invalid number format.")
        return None, None, None, None

# --- Main Block ---
if __name__ == "__main__":
    
    filename = "points.txt"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        
    print(f"Reading data from {filename}...")
    points, d, t, qbox = read_input_file(filename)
    
    if points is None:
        sys.exit(1)
        
    print(f"Loaded {len(points)} points. Dimensions: d={d}, t={t}.")
    
    input_indices = set()
    for i, p in enumerate(points):
        in_box = True
        for k in range(d):
            if not (qbox[k][0] <= p['coords'][k] <= qbox[k][1]):
                in_box = False
                break
        if in_box:
            input_indices.add(i)
            
    print(f"Query Box I contains {len(input_indices)} points.")
    
    alpha = 1.0
    beta = 1.0
    Wr = 1.0
    Wb = 1.0
    epsilon = 0.0
    
    print(f"Solving ILP with Warm-Start Binary Search (alpha={alpha}, beta={beta}, epsilon={epsilon})...")
    
    start_time = time.time()
    try:
        gamma, box = solve_fair_range_binary_search_warm_start(points, input_indices, alpha, beta, Wr, Wb, epsilon, d=d)
    except:
        traceback.print_exc()
        gamma = 0
        box = None
        
    end_time = time.time()
    
    print(f"Time taken to find fair range: {end_time - start_time:.4f} seconds")
    print(f"Optimal Gamma (Binary Search): {gamma:.4f}")
    
    if box:
        print("Optimal Box Bounds:")
        for k in range(d):
            print(f"  Dim {k}: [{box[k][0]:.6f}, {box[k][1]:.6f}]")

        # --- Jaccard Similarity Calculation ---
        output_indices = set()
        for i, p in enumerate(points):
             in_box = True
             for k in range(d):
                 if not (box[k][0] <= p['coords'][k] <= box[k][1]):
                     in_box = False
                     break
             if in_box:
                 output_indices.add(i)
        
        intersection_count = len(input_indices.intersection(output_indices))
        union_count = len(input_indices.union(output_indices))
        
        jaccard = intersection_count / union_count if union_count > 0 else 0.0
        print(f"Jaccard Similarity: {jaccard:.4f}")
    else:
        print("No feasible box found.")
