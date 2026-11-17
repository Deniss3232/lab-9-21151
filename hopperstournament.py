"""
Hoppers - game definition.

Reference implementation of the game functions in the style of CS50 AI
"Lecture 0" (initial_state, player, actions, result, terminal, winner,
utility). Nothing in this file knows anything about any agent, so any agent
can be plugged on top of it.

Rules implemented (from the Hoppers rulebook):

  - 10x10 board. Each player owns a 15-square triangular Corner Camp.
    P1 (player 1) starts top-left:     row + col <= 4
    P2 (player 2) starts bottom-right: row + col >= 14

  - On a turn a player either STEPS or HOPS one of their pieces.

      step: move to an adjacent vacant square, in any of the 8 directions
            (orthogonal or diagonal).

      hop:  jump over an adjacent piece, in any of the 8 directions, landing
            on the vacant square directly beyond it. The piece hopped over
            may be friendly or enemy, and is never removed from the board.

  - Hops may be chained. Taking further hops is OPTIONAL: the player may stop
    after any hop in the chain. Every landing square along a chain is
    therefore a legal destination in its own right.

  - A chain never lands twice on the same square (this is what makes the
    chain search terminate).

  - Stepping and hopping work exactly the same inside a Corner Camp as
    outside it.

  - A player wins by fully occupying the opponent's Corner Camp. Following
    the rulebook's anti-blocking note, a camp counts as full even if some of
    the pieces occupying it belong to the player who started there, so a
    sitting piece cannot make the game unwinnable. At least one of the
    occupying pieces must belong to the invader, otherwise the defender would
    "win" for the invader by simply staying home.

Board and coordinate convention (matches the representation handed out to
the class):

        0   1   2   3   4   5   6   7   8   9
    0  P1  P1  P1  P1  P1   0   0   0   0   0
    1  P1  P1  P1  P1   0   0   0   0   0   0
    ...
    9   0   0   0   0   0  P2  P2  P2  P2  P2

  - position  = (row, col), row 0 at the top, col 0 at the left
  - board     = tuple of 10 tuples of 10 ints
  - cell      = EMPTY (0), P1 (1) or P2 (2)
  - state     = State(board, to_move)
  - action    = Action(origin, destination, path)

States are immutable and hashable, so `result` cannot mutate its argument and
states can be used as dictionary keys (useful for transposition tables).
"""

from typing import Dict, List, NamedTuple, Optional, Tuple

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

SIZE = 10
PIECES_PER_PLAYER = 15
CAMP_SIZE = 15

EMPTY = 0
P1 = 1
P2 = 2

#: The eight directions a piece may step or hop in.
DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

#: The corner each player is racing towards (the far corner of the enemy camp).
GOAL_CORNER: Dict[int, Tuple[int, int]] = {P1: (9, 9), P2: (0, 0)}

Position = Tuple[int, int]
Board = Tuple[Tuple[int, ...], ...]


class Action(NamedTuple):
    """A legal move.

    origin      square the piece starts on
    destination square the piece ends on
    path        full route taken, origin first and destination last.
                A step has a path of length 2; a chain of n hops has a path
                of length n + 1. The path is what lets a UI animate the move
                and lets a human read back what the agent did.
    """

    origin: Position
    destination: Position
    path: Tuple[Position, ...]

    def is_hop(self) -> bool:
        """True when this move is a hop (or a chain of hops)."""
        r0, c0 = self.origin
        r1, c1 = self.path[1]
        return abs(r1 - r0) == 2 or abs(c1 - c0) == 2

    def __str__(self) -> str:
        route = " -> ".join(f"({r},{c})" for r, c in self.path)
        return route


class State(NamedTuple):
    """A complete game position: the board plus whose turn it is."""

    board: Board
    to_move: int


# --------------------------------------------------------------------------
# Board helpers
# --------------------------------------------------------------------------


def opponent(p: int) -> int:
    """The other player."""
    return P2 if p == P1 else P1


def in_bounds(row: int, col: int) -> bool:
    """True when (row, col) is on the board."""
    return 0 <= row < SIZE and 0 <= col < SIZE


def in_camp(player: int, row: int, col: int) -> bool:
    """True when (row, col) belongs to `player`'s own Corner Camp."""
    if player == P1:
        return row + col <= 4
    return row + col >= 14


def camp_cells(player: int) -> Tuple[Position, ...]:
    """Every square of `player`'s own Corner Camp."""
    return tuple(
        (r, c)
        for r in range(SIZE)
        for c in range(SIZE)
        if in_camp(player, r, c)
    )


def target_camp(player: int) -> int:
    """The camp `player` is trying to occupy: the opponent's."""
    return opponent(player)


def pieces_of(state: State, player: int) -> List[Position]:
    """Every square currently holding a piece of `player`."""
    board = state.board
    return [
        (r, c)
        for r in range(SIZE)
        for c in range(SIZE)
        if board[r][c] == player
    ]


# --------------------------------------------------------------------------
# The game functions
# --------------------------------------------------------------------------


def initial_state() -> State:
    """The starting position: 15 pieces per player, each in their own camp.

    P1 moves first.
    """
    board = []
    for r in range(SIZE):
        row = []
        for c in range(SIZE):
            if in_camp(P1, r, c):
                row.append(P1)
            elif in_camp(P2, r, c):
                row.append(P2)
            else:
                row.append(EMPTY)
        board.append(tuple(row))
    return State(tuple(board), P1)


def player(state: State) -> int:
    """Which player moves next in `state`."""
    return state.to_move


def actions(state: State) -> List[Action]:
    """Every legal move for the player to move.

    Includes single steps and hops, with chained hops expanded so that each
    landing square along a chain appears as its own action (the rulebook makes
    continuing a chain optional).
    """
    result_actions: List[Action] = []
    for origin in pieces_of(state, state.to_move):
        result_actions.extend(actions_for_piece(state, origin))
    return result_actions


def actions_for_piece(state: State, origin: Position) -> List[Action]:
    """Every legal move for the single piece standing on `origin`.

    Useful on its own for a UI that highlights one piece's options.
    """
    board = state.board
    r0, c0 = origin
    if not in_bounds(r0, c0) or board[r0][c0] == EMPTY:
        return []

    moves: List[Action] = []
    seen = set()

    # Steps: any adjacent vacant square.
    for dr, dc in DIRECTIONS:
        r, c = r0 + dr, c0 + dc
        if in_bounds(r, c) and board[r][c] == EMPTY:
            seen.add((r, c))
            moves.append(Action(origin, (r, c), (origin, (r, c))))

    # Hops. Lift the moving piece off the board first so that it cannot hop
    # over the square it just vacated.
    lifted = [list(row) for row in board]
    lifted[r0][c0] = EMPTY

    def chain(at: Position, path: Tuple[Position, ...]) -> None:
        ar, ac = at
        for dr, dc in DIRECTIONS:
            over_r, over_c = ar + dr, ac + dc
            land_r, land_c = ar + 2 * dr, ac + 2 * dc
            if not in_bounds(land_r, land_c):
                continue
            if lifted[over_r][over_c] == EMPTY:      # nothing to hop over
                continue
            if lifted[land_r][land_c] != EMPTY:      # landing square taken
                continue
            landing = (land_r, land_c)
            if landing in path:                      # never revisit a square
                continue

            new_path = path + (landing,)
            if landing not in seen:
                seen.add(landing)
                moves.append(Action(origin, landing, new_path))
            chain(landing, new_path)

    chain(origin, (origin,))
    return moves


def result(state: State, action: Action) -> State:
    """The state reached by playing `action`. `state` is never mutated.

    Raises ValueError if the action is not legal in `state`, so a referee can
    detect an agent trying to cheat.
    """
    board = state.board
    r0, c0 = action.origin
    r1, c1 = action.destination

    if not in_bounds(r0, c0) or not in_bounds(r1, c1):
        raise ValueError(f"action off the board: {action}")
    if board[r0][c0] != state.to_move:
        raise ValueError(
            f"no piece of player {state.to_move} on {action.origin}"
        )
    if board[r1][c1] != EMPTY:
        raise ValueError(f"destination {action.destination} is occupied")

    legal = {(a.origin, a.destination) for a in actions_for_piece(state, action.origin)}
    if (action.origin, action.destination) not in legal:
        raise ValueError(f"illegal move: {action}")

    new_board = [list(row) for row in board]
    new_board[r0][c0] = EMPTY
    new_board[r1][c1] = state.to_move
    return State(
        tuple(tuple(row) for row in new_board),
        opponent(state.to_move),
    )


def camp_occupancy(state: State, camp_owner: int) -> Tuple[int, int]:
    """How full `camp_owner`'s camp is.

    Returns (occupied, by_invader): how many of the camp's 15 squares hold a
    piece, and how many of those belong to the invader (the other player).
    """
    board = state.board
    invader = opponent(camp_owner)
    occupied = 0
    by_invader = 0
    for r, c in camp_cells(camp_owner):
        cell = board[r][c]
        if cell != EMPTY:
            occupied += 1
            if cell == invader:
                by_invader += 1
    return occupied, by_invader


def winner(state: State) -> Optional[int]:
    """The winning player, or None if nobody has won yet.

    A player wins by fully occupying the opponent's camp. The camp counts as
    full even if some squares are held by pieces of the player who started
    there (the rulebook's anti-blocking note), but at least one occupying
    piece must be the invader's.
    """
    for invader in (P1, P2):
        occupied, by_invader = camp_occupancy(state, opponent(invader))
        if occupied == CAMP_SIZE and by_invader >= 1:
            return invader
    return None


def terminal(state: State) -> bool:
    """True when the game is over.

    Either somebody has won, or the player to move has no legal action at all
    (which cannot happen from the standard opening, but is handled so that the
    search never explores a dead end).
    """
    if winner(state) is not None:
        return True
    return len(actions(state)) == 0


def utility(state: State) -> int:
    """The value of a terminal state, from P1's point of view.

    +1 if P1 has won, -1 if P2 has won, 0 for a game that ended without a
    winner. Matching the Lecture 0 convention, this is an absolute value and
    not relative to whoever is to move: P1 is MAX and P2 is MIN.
    """
    w = winner(state)
    if w == P1:
        return 1
    if w == P2:
        return -1
    if terminal(state):
        # No winner and no legal move: treat as a draw.
        return 0
    raise ValueError("utility called on a non-terminal state")


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------


def board_to_string(state: State) -> str:
    """The board in the coordinate layout handed out to the class."""
    header = "    " + "".join(f"{c:>8}" for c in range(SIZE))
    lines = [header]
    for r in range(SIZE):
        cells = []
        for c in range(SIZE):
            v = state.board[r][c]
            if v == EMPTY:
                cells.append(f"{0:>8}")
            else:
                cells.append(f"{('P1' if v == P1 else 'P2') + f'({r} {c})':>8}")
        lines.append(f"{r:<4}" + "".join(cells))
    return "\n".join(lines)


def board_to_grid(state: State) -> str:
    """A compact board view: . for empty, 1 and 2 for the players."""
    symbols = {EMPTY: ".", P1: "1", P2: "2"}
    header = "   " + " ".join(str(c) for c in range(SIZE))
    lines = [header]
    for r in range(SIZE):
        lines.append(f"{r}  " + " ".join(symbols[v] for v in state.board[r]))
    return "\n".join(lines)


def state_from_grid(rows: List[str], to_move: int) -> State:
    """Build a state from a compact grid, for tests and set-up positions.

    Each string is one row of 10 characters using '.', '1' and '2'.
    """
    if len(rows) != SIZE:
        raise ValueError(f"expected {SIZE} rows, got {len(rows)}")
    lookup = {".": EMPTY, "0": EMPTY, "1": P1, "2": P2}
    board = []
    for row in rows:
        cleaned = row.replace(" ", "")
        if len(cleaned) != SIZE:
            raise ValueError(f"expected {SIZE} columns, got {len(cleaned)!r}")
        board.append(tuple(lookup[ch] for ch in cleaned))
    return State(tuple(board), to_move)
