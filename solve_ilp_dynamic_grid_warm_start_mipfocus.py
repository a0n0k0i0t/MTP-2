import gurobipy as gp
from gurobipy import GRB
import time
import sys
import traceback
import bisect

def solve_fair_range_Dinkleback(points, input_box_indices, qbox_bounds, alpha, beta, Wr, Wb, epsilon, d=2, delta_base=2, use_dynamic_grid=True, start_gamma=0.0, warm_start_box=None):
    """
    Performs Dinkleback search on gamma to find the maximum gamma for which a feasible box exists.
    Precision: 2 decimal points.
    """
    feasible_gamma= start_gamma
    # Dummy value for previous
    previous_gamma = start_gamma - 1.0
    best_box = warm_start_box

    print(f"Starting Binary Search for Gamma... (Dynamic Grid: {use_dynamic_grid})")
    # Precision limit: 0.005 for approx 2 decimal points stability
    print(f"Gamma : {feasible_gamma}")
    while abs(feasible_gamma - previous_gamma) > 0.005: 
        # Try to find a feasible solution for gamma = mid
        box, new_gamma = solve_feasibility_for_gamma(
            points, input_box_indices, qbox_bounds, alpha, beta, Wr, Wb, epsilon, 
            feasible_gamma, d, delta_base, use_dynamic_grid, warm_start_box=best_box
        )
        
        if box is not None:
            previous_gamma = feasible_gamma
            feasible_gamma = new_gamma
            best_box = box
            print(f"Gamma {feasible_gamma:.4f} is FEASIBLE.")
        else:
            print(f"Something went wrong in Dinkleback optimization!\nINFEASIBLE.")
            break
            
    return feasible_gamma, best_box

def solve_feasibility_for_gamma(points, input_box_indices, qbox_bounds, alpha, beta, Wr, Wb, epsilon, feasible_gamma, d=2, delta_base=2, use_dynamic_grid=True, warm_start_box=None):
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

    # --- Dynamic Grid Generation ---
    allowed_L_vars = []
    allowed_R_vars = []
    for k in range(d):
        m = len(sorted_coords[k])
        if not use_dynamic_grid:
            allowed_L_vars.append(list(range(m)))
            allowed_R_vars.append(list(range(m)))
            continue

        L_in = qbox_bounds[k][0]
        R_in = qbox_bounds[k][1]
        
        L_rank = bisect.bisect_left(sorted_coords[k], L_in)
        if L_rank >= m: L_rank = m - 1
        R_rank = bisect.bisect_right(sorted_coords[k], R_in) - 1
        if R_rank < 0: R_rank = 0

        # L valid ranks
        allowed_L = set([0, int(L_rank)])
        step = 1
        while L_rank - step > 0:
            allowed_L.add(int(L_rank - step))
            step = max(step + 1, int(step * delta_base))
        step = 1
        while L_rank + step < R_rank:
            allowed_L.add(int(L_rank + step))
            step = max(step + 1, int(step * delta_base))
        
        # R valid ranks
        allowed_R = set([m - 1, int(R_rank)])
        step = 1
        while R_rank + step < m - 1:
            allowed_R.add(int(R_rank + step))
            step = max(step + 1, int(step * delta_base))
        step = 1
        while R_rank - step > L_rank:
            allowed_R.add(int(R_rank - step))
            step = max(step + 1, int(step * delta_base))
            
        allowed_L_vars.append(sorted(list(allowed_L)))
        allowed_R_vars.append(sorted(list(allowed_R)))

    # --- Model ---
    model = gp.Model("FairRangeFeasibility")
    model.setParam("OutputFlag", 0)
    model.setParam("MIPFocus", 1)  # Focus mostly on finding feasible solutions fast
    model.setParam("Heuristics", 0.5)  # Spend 50% of time in node heuristics

    # --- Boundary variables ---
    l_vars, r_vars, pos_L, pos_R = [], [], [], []
    for k in range(d):
        l = {}
        for j in allowed_L_vars[k]:
            l[j] = model.addVar(vtype=GRB.BINARY, name=f"L_{k}_{j}")
        r = {}
        for j in allowed_R_vars[k]:
            r[j] = model.addVar(vtype=GRB.BINARY, name=f"R_{k}_{j}")
            
        if warm_start_box is not None:
            # Check if exactly matching value exists in allowed_L 
            val_L = warm_start_box[k][0]
            if val_L in sorted_coords[k]:
                idx_L = sorted_coords[k].index(val_L)
                if idx_L in l:
                    l[idx_L].Start = 1
            
            val_R = warm_start_box[k][1]
            if val_R in sorted_coords[k]:
                idx_R = sorted_coords[k].index(val_R)
                if idx_R in r:
                    r[idx_R].Start = 1
        
        l_vars.append(l)
        r_vars.append(r)

        model.addConstr(gp.quicksum(l.values()) == 1)
        model.addConstr(gp.quicksum(r.values()) == 1)

        posL_expr = gp.quicksum(j * l[j] for j in allowed_L_vars[k])
        posR_expr = gp.quicksum(j * r[j] for j in allowed_R_vars[k])
        
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
    M = 2*max(len(vals) for vals in sorted_coords)

    for i in range(n):
        for k in range(d):
            Z[i, k] = model.addVar(vtype=GRB.BINARY, name=f"Z_{i}_{k}")

            r_ik = point_ranks[i][k]
            # Forward Implication: Z=1 => Inside [L, R]
            model.addConstr(r_ik >= pos_L[k] - M * (1 - Z[i, k]))
            model.addConstr(r_ik <= pos_R[k] + M * (1 - Z[i, k]))

            # Reverse Implication: Inside => Z=1
            y_L_ik = model.addVar(vtype=GRB.BINARY, name=f"yL_{i}_{k}")
            y_R_ik = model.addVar(vtype=GRB.BINARY, name=f"yR_{i}_{k}")

            # yL=1 => pos_L > r_ik (strictly left)
            model.addConstr(pos_L[k] >= r_ik + 1 - M * (1 - y_L_ik))

            # yR=1 => pos_R < r_ik (strictly right)
            model.addConstr(pos_R[k] <= r_ik - 1 + M * (1 - y_R_ik))

            # If not strictly left and not strictly right, must be inside (Z=1)
            model.addConstr(y_L_ik + y_R_ik + Z[i, k] >= 1)

            model.addConstr(O[i] <= Z[i, k])

        model.addConstr(O[i] >= gp.quicksum(Z[i, k] for k in range(d)) - (d - 1))

    # --- Similarity bookkeeping ---
    S_ino = gp.LinExpr()
    S_io = gp.LinExpr()
    S_oi = gp.LinExpr()

    count_R = gp.LinExpr()
    count_B = gp.LinExpr()

    count = 0
    for i in range(n):
        in_I = (i in input_box_indices)

        if in_I:
            count += 1
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

    # --- Similarity Feasibility Check ---
    model.setObjective((1- feasible_gamma) * S_ino - feasible_gamma * alpha * S_io - feasible_gamma * beta * S_oi, GRB.MAXIMIZE)
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        # Extract solution
        best_box = []
        for k in range(d):
            l_idx = -1
            r_idx = -1
            for j in allowed_L_vars[k]:
                 if l_vars[k][j].X > 0.5:
                     l_idx = j
            for j in allowed_R_vars[k]:
                 if r_vars[k][j].X > 0.5:
                     r_idx = j
            
            if l_idx == -1: l_idx = 0
            if r_idx == -1: r_idx = len(sorted_coords[k]) - 1
            
            best_box.append((sorted_coords[k][l_idx], sorted_coords[k][r_idx]))
        inter = S_ino.getValue()
        inoto = S_io.getValue()
        onoti = S_oi.getValue()
        
        similarity = inter/(inter + alpha * inoto + beta * onoti) if (inter + alpha * inoto + beta * onoti) > 0 else 0
        return best_box, similarity
    else:
        return None, 0.0

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
    delta_base = 2
    
    print(f"Solving ILP with Dynamic Grid Dinkleback (alpha={alpha}, beta={beta}, epsilon={epsilon}, delta={delta_base})...")
    
    start_time = time.time()
    try:
        print("--- PHASE 1: DYNAMIC GRID ---")
        gamma_approx, box_approx = solve_fair_range_Dinkleback(points, input_indices, qbox, alpha, beta, Wr, Wb, epsilon, d=d, delta_base=delta_base, use_dynamic_grid=True)
        
        print("\n--- PHASE 2: EXACT SEARCH WARM START ---")
        if box_approx is not None:
            print(f"Using Phase 1 result (Gamma: {gamma_approx:.4f}) as warm start for Exact Search.")
            gamma, box = solve_fair_range_Dinkleback(points, input_indices, qbox, alpha, beta, Wr, Wb, epsilon, d=d, delta_base=1, use_dynamic_grid=False, start_gamma=gamma_approx, warm_start_box=box_approx)
        else:
            print("Phase 1 failed. Running exact full space from scratch...")
            gamma, box = solve_fair_range_Dinkleback(points, input_indices, qbox, alpha, beta, Wr, Wb, epsilon, d=d, delta_base=1, use_dynamic_grid=False)
    except:
        traceback.print_exc()
        gamma = 0
        box = None
        
    end_time = time.time()
    
    print(f"Time taken to find fair range: {end_time - start_time:.4f} seconds")
    print(f"Optimal Gamma: {gamma:.4f}")
    
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
