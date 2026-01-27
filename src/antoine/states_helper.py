import numpy as np

def column_heights(board):
    """
    Retourne la hauteur (nombre de cellules remplies) par colonne (sans compter padding).
    On calcule l'index du premier bloc (non-zero) depuis le haut -> hauteur = H - idx.
    """
    H, W = board.shape
    heights = []
    for j in range(W):
        if board[0, j] == 1:  # padding/borne
            continue
        i = 2
        while i < H and board[i, j] == 0:
            i += 1
        heights.append(H - i)  # si colonne vide => H - H = 0
    return np.array(heights, dtype=np.int32)

def holes_count(board):
    """Nombre de trous: case vide avec au-dessus une case non vide."""
    H, W = board.shape
    holes = 0
    for i in range(1, H):
        for j in range(W):
            if board[0, j] == 1:
                continue
            if board[i, j] == 0 and board[i - 1, j] != 0:
                holes += 1
    return holes

def holes_depth(board):
    """
    Somme des "profondeurs" de trous:
    pour chaque trou, combien de blocs au-dessus dans la colonne.
    """
    H, W = board.shape
    depth_sum = 0
    for j in range(W):
        if board[0, j] == 1:
            continue
        filled_seen = 0
        for i in range(H):
            if board[i, j] != 0:
                filled_seen += 1
            else:
                if filled_seen > 0:
                    depth_sum += filled_seen
    return depth_sum

def bumpiness(heights):
    return int(np.sum(np.abs(heights[:-1] - heights[1:])))

def row_transitions(board):
    """
    Transitions vide/plein par ligne (sans padding), classique en Tetris heuristique.
    """
    H, W = board.shape
    trans = 0
    # colonnes "jouables" (sans padding)
    playable_cols = [j for j in range(W) if board[0, j] != 1]
    for i in range(H):
        prev = 1  # bord gauche considéré rempli
        for j in playable_cols:
            cur = 1 if board[i, j] != 0 else 0
            if cur != prev:
                trans += 1
            prev = cur
        # bord droit considéré rempli
        if prev == 0:
            trans += 1
    return trans

def col_transitions(board):
    """
    Transitions vide/plein par colonne (sans padding).
    """
    H, W = board.shape
    trans = 0
    playable_cols = [j for j in range(W) if board[0, j] != 1]
    for j in playable_cols:
        prev = 1  # bord haut considéré rempli
        for i in range(H):
            cur = 1 if board[i, j] != 0 else 0
            if cur != prev:
                trans += 1
            prev = cur
        # bord bas considéré rempli
        if prev == 0:
            trans += 1
    return trans

def wells(heights):
    """
    Somme de profondeurs de "puits" : si une colonne est plus basse que ses voisines,
    on ajoute la différence (ou plus fin avec cumul). Version simple.
    """
    w = 0
    n = len(heights)
    for i in range(n):
        left = heights[i-1] if i-1 >= 0 else heights[i] + 10
        right = heights[i+1] if i+1 < n else heights[i] + 10
        m = min(left, right)
        if heights[i] < m:
            w += (m - heights[i])
    return int(w)


def make_info_from_obs(board):
    h = holes_count(board)
    hs = column_heights(board)

    agg_h = int(np.sum(hs))
    max_h = int(np.max(hs)) if len(hs) else 0
    bump = bumpiness(hs) if len(hs) >= 2 else 0
    rtrans = row_transitions(board)
    ctrans = col_transitions(board)
    well = wells(hs)
    hdepth = holes_depth(board)

    # Features brutes
    x = np.array(
        [1.0, h, agg_h, max_h, bump, rtrans, ctrans, well, hdepth],
        dtype=np.float32
    )

    # Dimensions "jouables"
    H, W_total = board.shape
    n_cols = int(np.sum(board[0, :] != 1))  # colonnes sans padding

    # --- SCALES (normalisation systématique) ---
    # 1) holes: typiquement faible; on prend ~10% des cases comme échelle.
    holes_scale = max(1.0, 0.10 * H * n_cols)          # ex: 0.1*20*10=20

    # 2) aggregate height: max théorique = H*n_cols
    agg_h_scale = max(1.0, float(H * n_cols))          # ex: 200

    # 3) max height: max = H
    max_h_scale = max(1.0, float(H))                   # ex: 20

    # 4) bumpiness: max ~ (n_cols-1)*H
    bump_scale = max(1.0, float((n_cols - 1) * H))     # ex: 180

    # 5) row transitions: max ~ H*(2*n_cols + 2) (en comptant les bords)
    rtrans_scale = max(1.0, float(H * (2 * n_cols + 2)))  # ex: 440

    # 6) col transitions: max ~ n_cols*(2*H + 2)
    ctrans_scale = max(1.0, float(n_cols * (2 * H + 2)))  # ex: 420

    # 7) wells (version simple): ordre de grandeur max ~ H*n_cols
    wells_scale = max(1.0, float(H * n_cols))          # ex: 200

    # 8) holes depth: max théorique ~ n_cols * H*(H-1)/2 ; on prend 50% comme échelle pratique
    max_depth_theoretical = float(n_cols * (H * (H - 1) / 2.0))  # ex: 1900
    hdepth_scale = max(1.0, 0.50 * max_depth_theoretical)        # ex: 950

    scales = np.array(
        [1.0, holes_scale, agg_h_scale, max_h_scale, bump_scale,
         rtrans_scale, ctrans_scale, wells_scale, hdepth_scale],
        dtype=np.float32
    )

    return x / scales