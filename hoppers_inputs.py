import numpy as np
from hopperstournament import P1, P2, actions, in_camp, opponent, pieces_of, result, winner

GOAL = {P1: (9, 9), P2: (0, 0)}


def canon(player, r, c):
    """Square (r, c) as `player` sees it: player 2 sees the board rotated."""
    return (9 - r, 9 - c) if player == P2 else (r, c)


def encode_board(state, moves):
    """(a) 400 numbers per move: my pieces, their pieces, origin, destination."""
    me = state.to_move
    board = np.array(state.board, dtype=np.int8)
    if me == P2:
        board = board[::-1, ::-1]
    X = np.zeros((len(moves), 400), dtype=np.float32)
    X[:, :100] = (board == me).ravel()
    X[:, 100:200] = (board == opponent(me)).ravel()
    for i, a in enumerate(moves):
        r0, c0 = canon(me, *a.origin)
        r1, c1 = canon(me, *a.destination)
        X[i, 200 + 10 * r0 + c0] = 1.0
        X[i, 300 + 10 * r1 + c1] = 1.0
    return X


def cheb(a, b):
    """Single steps between two squares (pieces also move diagonally)."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _targets(p):
    sr, sc = GOAL[opponent(p)]
    sign = 1 if p == P1 else -1
    cells = [(r, c) for r in range(10) for c in range(10) if in_camp(opponent(p), r, c)]
    cells.sort(key=lambda x: (-cheb(x, (sr, sc)), sign * (x[0] - x[1])))
    return cells


_TARGETS = {P1: _targets(P1), P2: _targets(P2)}


def travel(state, p):
    """Steps player p still needs to fill the target camp (the course heuristic)."""
    gr, gc = GOAL[p]
    sign = 1 if p == P1 else -1
    remaining = sorted(pieces_of(state, p), key=lambda x: (cheb(x, (gr, gc)), sign * (x[0] - x[1])))
    total = 0
    for t in _TARGETS[p]:
        if not remaining:
            break
        best_i, best_d = 0, None
        for i, cell in enumerate(remaining):
            d = cheb(cell, t)
            if best_d is None or d < best_d:
                best_i, best_d = i, d
            if d == 0:
                break
        total += best_d
        remaining.pop(best_i)
    return total


FEATURES = [
    "gain",             # steps this move saves in my remaining travel (the course heuristic)
    "advance",          # how much closer to the goal corner the piece gets
    "is_hop",           # 1 if it jumps, 0 if it steps
    "hops",             # number of jumps in the chain (0 for a step)
    "leaves_home",      # 1 if the piece leaves my starting camp
    "enters_target",    # 1 if the piece enters the camp I must fill
    "moves_in_target",  # 1 if the piece was already in the target camp
    "behind",           # how far ahead this piece is of my LAST piece (0 = it is my last piece)
    "dest_goal_dist",   # steps from the destination to the goal corner
    "dest_off_diag",    # distance of the destination from the main diagonal (0 = on it)
]


def encode_features(state, moves):
    """(b) 10 numbers per move, in the order of FEATURES.

    Everything is seen from the player to move ("me"), with the board rotated for
    player 2, so my goal corner is always (9, 9) and my home corner (0, 0).
    A handy number: r + c, the "diagonal" of a square. It is 0 at my home corner
    and 18 at my goal corner, so a bigger r + c means further forward.
      - my home camp (where I start)       = the 15 squares with r + c <= 4
      - the target camp (where I must go)  = the 15 squares with r + c >= 14
    """
    me = state.to_move

    # Computed once per position (they do not depend on the move):
    base = travel(state, me)               # steps I still need to fill the target camp, before moving
    last = min(sum(canon(me, r, c))        # diagonal of my most-behind piece
               for r, c in pieces_of(state, me))

    X = np.zeros((len(moves), len(FEATURES)), dtype=np.float32)   # one row per move
    for i, a in enumerate(moves):
        r0, c0 = canon(me, *a.origin)          # where the piece starts (my point of view)
        r1, c1 = canon(me, *a.destination)     # where the piece ends
        d0, d1 = r0 + c0, r1 + c1              # diagonal before and after the move
        hop = a.is_hop()

        gain = base - travel(result(state, a), me)   # steps the move saves me (the course heuristic);
                                                     # it can be negative if the move goes backwards
        advance = d1 - d0                            # how many diagonals the piece moves forward
        is_hop = float(hop)                          # 1 = jump, 0 = single step
        hops = len(a.path) - 1 if hop else 0         # jumps in the chain (path = origin + each landing square)
        leaves_home = float(d0 <= 4 < d1)            # 1 if the piece starts in my home camp and leaves it
        enters_target = float(d0 < 14 <= d1)         # 1 if the piece was outside the target camp and gets in
        moves_in_target = float(d0 >= 14)            # 1 if the piece was already in the target camp
        behind = d0 - last                           # how far ahead this piece is of my last piece
                                                     # (0 = it IS my last piece, the one that may get stuck)
        dest_goal_dist = max(9 - r1, 9 - c1)         # single steps from the destination to the goal corner
        dest_off_diag = abs(r1 - c1)                 # distance from the destination to the main diagonal
                                                     # (0 = on the diagonal, the shortest road to the goal)

        X[i] = (gain, advance, is_hop, hops, leaves_home,
                enters_target, moves_in_target, behind, dest_goal_dist, dest_off_diag)
    return X


ENCODERS = {"a": encode_board, "b": encode_features}
