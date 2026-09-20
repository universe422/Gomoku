"""렌주 기본 규칙 + 클릭 대국. Python 3.12, 외부 패키지 불필요.

실행: python renju.py             검사: python renju.py --test
규칙 근거: https://www.renju.net/rifrules/ (3, 9.1~9.3)
범위: 15x15, 흑 중앙 시작, 이후 자유 착수. 대회용 교환/26개 오프닝,
시계, 승리 선언/이의 제기 절차는 구현하지 않는다. 결과는 즉시 자동 판정.
흑 금수는 착수를 차단하며 차례를 유지한다. 흑 차례에 금수 위치를 X로 표시한다. 두 번 연속 패스는 무승부. 처음 3수는 패스 불가.
학습 AI는 아직 없음. RenjuGame과 analyze_move는 향후 AI에서 재사용 가능.
"""
from dataclasses import dataclass
from typing import Optional
import sys

SIZE = 15
EMPTY, BLACK, WHITE = 0, 1, 2
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
MESSAGES = {
    'ok': '정상 착수', 'five': '5목', 'white_win': '백 5목 또는 장목',
    'overline': '장목 금수', 'double_four': '사사 금수',
    'double_three': '삼삼 금수', 'outside': '판 밖입니다',
    'occupied': '이미 돌이 있습니다', 'center': '첫 흑돌은 중앙에 놓으세요',
    'finished': '대국이 끝났습니다', 'draw': '무승부', 'pass': '패스',
    'early_pass': '처음 3수에는 패스할 수 없습니다', 'resign': '기권',
}
FOULS = {'overline', 'double_four', 'double_three'}


def create_board():
    return [[EMPTY for _ in range(SIZE)] for _ in range(SIZE)]


def inside(row, col):
    return 0 <= row < SIZE and 0 <= col < SIZE


def count_line(board, row, col, dr, dc):
    if not inside(row, col) or board[row][col] == EMPTY:
        return 0
    stone = board[row][col]
    count = 1
    for sign in (1, -1):
        r, c = row + sign * dr, col + sign * dc
        while inside(r, c) and board[r][c] == stone:
            count += 1
            r, c = r + sign * dr, c + sign * dc
    return count


def _fours(board, row, col):
    """(방향, 네 돌의 위치) -> 정확한 5목으로 완성하는 빈칸들.

    열린 4 양 끝 중복을 제거하고 같은 직선의 서로 다른 4도 구분한다.
    """
    result = {}
    for d, (dr, dc) in enumerate(DIRECTIONS):
        for offset in range(5):
            cells = [(row + (i - offset) * dr,
                      col + (i - offset) * dc) for i in range(5)]
            if not all(inside(r, c) for r, c in cells):
                continue
            stones = tuple((r, c) for r, c in cells if board[r][c] == BLACK)
            empty = [(r, c) for r, c in cells if board[r][c] == EMPTY]
            if len(stones) != 4 or len(empty) != 1:
                continue
            r, c = empty[0]
            board[r][c] = BLACK
            try:
                exact_five = count_line(board, r, c, dr, dc) == 5
            finally:
                board[r][c] = EMPTY
            if exact_five:
                result.setdefault((d, stones), set()).add((r, c))
    return result


def _three_candidates(board, row, col):
    """세 돌별 열린 4 확장 후보. 후보 착수의 금수 여부는 아직 미검사."""
    result = {}
    for d, (dr, dc) in enumerate(DIRECTIONS):
        for offset in range(4):
            cells = [(row + (i - offset) * dr,
                      col + (i - offset) * dc) for i in range(4)]
            before = (cells[0][0] - dr, cells[0][1] - dc)
            after = (cells[-1][0] + dr, cells[-1][1] + dc)
            if not all(inside(r, c) for r, c in cells + [before, after]):
                continue
            if any(board[r][c] != EMPTY for r, c in (before, after)):
                continue
            stones = tuple((r, c) for r, c in cells if board[r][c] == BLACK)
            empty = [(r, c) for r, c in cells if board[r][c] == EMPTY]
            if len(stones) != 3 or len(empty) != 1:
                continue
            # 양쪽 끝에 놓았을 때 모두 정확한 5목이어야 열린 4이다.
            outer = [(before[0] - dr, before[1] - dc),
                     (after[0] + dr, after[1] + dc)]
            if any(inside(r, c) and board[r][c] == BLACK for r, c in outer):
                continue
            result.setdefault((d, stones), set()).add(empty[0])
    return result


def _black_status(board, row, col, memo):
    """이미 놓인 흑돌 판정. 삼삼의 확장 수를 재귀 검사한다.

    재귀 깊이를 임의로 잘라 가짜 3을 진짜 3으로 간주하지 않는다.
    매 재귀에 빈칸이 줄어들므로 종료하며, 같은 상태는 캐시한다.
    """
    key = (bytes(cell for line in board for cell in line), row, col)
    if key in memo:
        return memo[key]
    lengths = [count_line(board, row, col, dr, dc) for dr, dc in DIRECTIONS]
    if 5 in lengths:
        status = 'five'  # 다른 방향 금수보다 정확한 5목 우선 (RIF 9.2)
    elif any(n >= 6 for n in lengths):
        status = 'overline'
    elif len(_fours(board, row, col)) >= 2:
        status = 'double_four'
    else:
        status = 'ok'
        candidates = _three_candidates(board, row, col)
        # 3 후보가 둘 미만이면 삼삼일 수 없다.
        if len(candidates) >= 2:
            real_threes = 0
            for extensions in candidates.values():
                for r, c in sorted(extensions):
                    board[r][c] = BLACK
                    try:
                        extension = _black_status(board, r, c, memo)
                    finally:
                        board[r][c] = EMPTY
                    # 확장 수가 금수이거나 동시에 5목이면 유효한 3이 아니다.
                    if extension == 'ok':
                        real_threes += 1
                        break
                if real_threes >= 2:
                    status = 'double_three'
                    break
    memo[key] = status
    return status


@dataclass(frozen=True)
class MoveResult:
    code: str
    legal: bool
    winner: Optional[int] = None

    @property
    def message(self):
        return MESSAGES[self.code]


def analyze_move(board, row, col, stone):
    """가상 착수 판정. 원본 판을 변경하지 않는다. 차례/첫수는 Game에서 검사.

    winner: None=미종료, BLACK/WHITE=승자. 금수이면 legal=False, winner=None (착수 차단).
    """
    if stone not in (BLACK, WHITE):
        raise ValueError('stone은 BLACK(1) 또는 WHITE(2)여야 합니다')
    if not inside(row, col):
        return MoveResult('outside', False)
    if board[row][col] != EMPTY:
        return MoveResult('occupied', False)
    board[row][col] = stone
    try:
        if stone == BLACK:
            code = _black_status(board, row, col, {})
            winner = BLACK if code == 'five' else None
            return MoveResult(code, code not in FOULS, winner)
        lengths = [count_line(board, row, col, dr, dc) for dr, dc in DIRECTIONS]
        if any(n >= 5 for n in lengths):
            return MoveResult('white_win', True, WHITE)
        return MoveResult('ok', True)
    finally:
        board[row][col] = EMPTY


class RenjuGame:
    """UI에서 독립된 게임 상태. winner=0은 무승부, None은 진행 중."""
    def __init__(self):
        self.board = create_board()
        self.turn = BLACK
        self.winner = None
        self.reason = 'ok'
        self.stone_count = 0
        self.passes = 0
        self.last_move = None
        self.history = []

    def _save(self):
        self.history.append(([line[:] for line in self.board], self.turn,
                             self.winner, self.reason, self.stone_count,
                             self.passes, self.last_move))

    def undo(self):
        if not self.history:
            return False
        (self.board, self.turn, self.winner, self.reason, self.stone_count,
         self.passes, self.last_move) = self.history.pop()
        return True

    def inspect(self, row, col):
        if self.winner is not None:
            return MoveResult('finished', False)
        if self.stone_count == 0 and (row, col) != (7, 7):
            return MoveResult('center', False)
        return analyze_move(self.board, row, col, self.turn)

    def legal_moves(self):
        """학습/탐색용. 흑의 금수를 제외한 착수 좌표 (패스는 별도)."""
        if self.winner is not None:
            return []
        if self.stone_count == 0:
            return [(7, 7)]
        return [(r, c) for r in range(SIZE) for c in range(SIZE)
                if self.board[r][c] == EMPTY and self.inspect(r, c).legal]

    def forbidden_moves(self):
        """현재 흑 차례에 표시할 금수 위치와 사유. 판을 변경하지 않는다."""
        if self.turn != BLACK or self.winner is not None or self.stone_count == 0:
            return {}
        result = {}
        for r in range(SIZE):
            for c in range(SIZE):
                if self.board[r][c] == EMPTY:
                    move = analyze_move(self.board, r, c, BLACK)
                    if move.code in FOULS:
                        result[(r, c)] = move.code
        return result

    def play(self, row, col):
        result = self.inspect(row, col)
        if not result.legal:
            return result
        self._save()
        self.board[row][col] = self.turn
        self.last_move = (row, col)
        self.stone_count += 1
        self.passes = 0
        self.winner = result.winner
        self.reason = result.code
        if self.winner is None and self.stone_count == SIZE * SIZE:
            self.winner, self.reason = EMPTY, 'draw'
            result = MoveResult('draw', True, EMPTY)
        if self.winner is None:
            self.turn = WHITE if self.turn == BLACK else BLACK
        return result

    def pass_turn(self):
        if self.winner is not None:
            return MoveResult('finished', False)
        if self.stone_count < 3:
            return MoveResult('early_pass', False)
        self._save()
        self.passes += 1
        self.last_move = None
        self.reason = 'pass'
        if self.passes == 2:
            self.winner, self.reason = EMPTY, 'draw'
        else:
            self.turn = WHITE if self.turn == BLACK else BLACK
        return MoveResult(self.reason, True, self.winner)

    def resign(self):
        if self.winner is not None:
            return MoveResult('finished', False)
        self._save()
        self.winner = WHITE if self.turn == BLACK else BLACK
        self.reason = 'resign'
        return MoveResult('resign', True, self.winner)


def launch_gui():
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.title('렌주 오목 | 2인 대국 · 규칙 엔진')
    root.resizable(False, False)
    game = RenjuGame()
    margin, gap = 36, 36
    side = margin * 2 + gap * (SIZE - 1)
    status = tk.StringVar()
    tk.Label(root, text='렌주 오목', font=('Malgun Gothic', 18, 'bold')).pack(pady=(10, 2))
    tk.Label(root, text='흑 중앙 시작 · 빨간 X는 흑 착수 금지 · 2인 대국',
             font=('Malgun Gothic', 10)).pack()
    canvas = tk.Canvas(root, width=side, height=side, bg='#DFC08B', highlightthickness=0)
    canvas.pack(padx=12, pady=10)
    tk.Label(root, textvariable=status, font=('Malgun Gothic', 12), wraplength=570).pack(pady=4)
    controls = tk.Frame(root)
    controls.pack(pady=(4, 12))

    def draw(note=''):
        canvas.delete('all')
        end = margin + gap * 14
        for i in range(SIZE):
            p = margin + i * gap
            canvas.create_line(margin, p, end, p, fill='#6D512D')
            canvas.create_line(p, margin, p, end, fill='#6D512D')
            canvas.create_text(p, 16, text=str(i), fill='#4B3820')
            canvas.create_text(16, p, text=str(i), fill='#4B3820')
        for r, c in ((3, 3), (3, 11), (7, 7), (11, 3), (11, 11)):
            x, y = margin + c * gap, margin + r * gap
            canvas.create_oval(x-3, y-3, x+3, y+3, fill='#6D512D', outline='')
        for r in range(SIZE):
            for c in range(SIZE):
                stone = game.board[r][c]
                if stone:
                    x, y = margin + c * gap, margin + r * gap
                    canvas.create_oval(x-15, y-15, x+15, y+15,
                                       fill='#202126' if stone == BLACK else '#FAFAF7',
                                       outline='#555555', width=1)
        for r, c in game.forbidden_moves():
            x, y = margin + c * gap, margin + r * gap
            canvas.create_line(x-7, y-7, x+7, y+7, fill='#C72F32', width=3,
                               tags='forbidden')
            canvas.create_line(x-7, y+7, x+7, y-7, fill='#C72F32', width=3,
                               tags='forbidden')
        if game.last_move:
            r, c = game.last_move
            x, y = margin + c * gap, margin + r * gap
            canvas.create_oval(x-4, y-4, x+4, y+4, fill='#EE735A', outline='')
        if game.winner is None:
            text = ('흑' if game.turn == BLACK else '백') + ' 차례'
            if game.stone_count == 0:
                text += ' — 중앙 (7, 7)을 클릭하세요'
            elif game.reason == 'pass':
                text += ' — 상대가 패스했습니다'
        elif game.winner == EMPTY:
            text = '무승부'
        else:
            text = ('흑' if game.winner == BLACK else '백') + ' 승리 — ' + MESSAGES[game.reason]
        status.set(text + (' | ' + note if note else ''))

    def click(event):
        col, row = round((event.x-margin)/gap), round((event.y-margin)/gap)
        if not inside(row, col):
            return
        if abs(event.x-(margin+col*gap)) > gap*0.45 or abs(event.y-(margin+row*gap)) > gap*0.45:
            return
        result = game.play(row, col)
        draw(result.message + ' — 다른 곳에 놓으세요' if not result.legal else '')

    def reset():
        nonlocal game
        if game.history and not messagebox.askyesno('새 게임', '현재 대국을 지우고 새로 시작할까요?'):
            return
        game = RenjuGame()
        draw()

    def undo():
        draw('') if game.undo() else draw('무를 수가 없습니다')

    def pass_move():
        result = game.pass_turn()
        draw('' if result.legal else result.message)

    def resign():
        if game.winner is None and messagebox.askyesno('기권', '현재 차례의 플레이어가 기권할까요?'):
            game.resign()
            draw()

    def rules():
        messagebox.showinfo('적용 규칙',
            '15×15 / 흑 첫수 중앙 / 이후 자유 착수\n\n'
            '흑: 정확한 5목 승리, 장목·사사·유효한 삼삼은 착수 불가\n'
            '백: 5개 이상 연결하면 승리, 삼삼·사사 제한 없음\n'
            '흑도 정확한 5목과 금수 모양을 동시에 만들면 승리\n'
            '가짜 3은 재귀적으로 검사하여 삼삼에서 제외\n'
            '양쪽이 막혀 있어도 정확한 5목이면 승리\n'
            '처음 3수 패스 불가, 두 번 연속 패스 또는 만석은 무승부\n\n'
            '대회용 교환 오프닝·시계·승리 선언 절차는 미포함\n'
            '규칙: https://www.renju.net/rifrules/')

    for title, action in [('새 게임', reset), ('한 수 무르기', undo), ('패스', pass_move),
                          ('기권', resign), ('규칙 안내', rules)]:
        tk.Button(controls, text=title, command=action, width=11).pack(side='left', padx=3)
    canvas.bind('<Button-1>', click)
    draw()
    root.mainloop()


def run_tests():
    """직접 python renju.py --test 로도 재검사 가능."""
    import unittest

    def board_with(black=(), white=()):
        board = create_board()
        for r, c in black:
            board[r][c] = BLACK
        for r, c in white:
            board[r][c] = WHITE
        return board

    class RulesTests(unittest.TestCase):
        def check(self, black, move, expected, white=(), stone=BLACK):
            board = board_with(black, white)
            snapshot = [line[:] for line in board]
            result = analyze_move(board, *move, stone)
            self.assertEqual(result.code, expected)
            self.assertEqual(board, snapshot, '판정 중 원본 판이 변경되었습니다')
            return result

        def test_black_five(self):
            self.check([(7,c) for c in (3,4,5,6)], (7,7), 'five')

        def test_black_overline(self):
            self.check([(7,c) for c in (3,4,5,7,8)], (7,6), 'overline')

        def test_white_overline(self):
            self.check([], (7,6), 'white_win', [(7,c) for c in (3,4,5,7,8)], WHITE)

        def test_five_beats_overline(self):
            stones = [(7,c) for c in (3,4,5,6)] + [(r,7) for r in (4,5,6,8,9)]
            self.check(stones, (7,7), 'five')

        def test_blocked_five_wins(self):
            self.check([(7,c) for c in (4,5,6,7)], (7,8), 'five', [(7,3),(7,9)])

        def test_open_four_counted_once(self):
            self.check([(7,c) for c in (5,6,8)], (7,7), 'ok')
            board = board_with([(7,c) for c in (5,6,7,8)])
            self.assertEqual(len(_fours(board,7,7)),1)

        def test_broken_four(self):
            board = board_with([(7,c) for c in (4,5,7,8)])
            self.assertEqual(len(_fours(board,7,7)),1)

        def test_double_four(self):
            self.check([(7,5),(7,6),(7,8),(5,7),(6,7),(8,7)], (7,7), 'double_four')

        def test_same_line_double_four(self):
            self.check([(7,c) for c in (3,5,6,9)], (7,7), 'double_four')

        def test_double_three(self):
            self.check([(7,6),(7,8),(6,7),(8,7)], (7,7), 'double_three')

        def test_broken_double_three(self):
            self.check([(7,5),(7,8),(5,7),(8,7)], (7,7), 'double_three')

        def test_blocked_three_not_counted(self):
            self.check([(7,6),(7,8),(6,7),(8,7)], (7,7), 'ok', [(7,5)])

        def test_edge_three_not_counted(self):
            self.check([(0,0),(0,2),(1,1),(2,1)], (0,1), 'ok')

        def test_fake_three_overline_extensions(self):
            stones = [(7,6),(7,8),(6,7),(8,7)]
            stones += [(r,c) for c in (5,9) for r in (4,5,6,8,9)]
            self.check(stones,(7,7),'ok')

        def test_fake_three_double_four_extensions(self):
            stones = [(7,6),(7,8),(6,7),(8,7)]
            stones += [(r,c) for c in (5,9) for r in (5,6,8)]
            self.check(stones,(7,7),'ok')

        def test_fake_three_recursive_double_three(self):
            stones = [(7,6),(7,8),(6,7),(8,7),
                      (6,5),(8,5),(6,4),(8,6),
                      (6,9),(8,9),(6,10),(8,8)]
            self.check(stones,(7,7),'ok')

        def test_white_double_three_allowed(self):
            self.check([], (7,7), 'ok', [(7,6),(7,8),(6,7),(8,7)], WHITE)

        def test_four_three_allowed(self):
            self.check([(7,5),(7,6),(7,8),(6,7),(8,7)], (7,7), 'ok')

        def test_five_beats_double_three(self):
            stones = [(7,c) for c in (3,4,5,6)] + [(6,7),(8,7),(6,6),(8,8)]
            self.check(stones,(7,7),'five')

        def test_invalid_and_board_restore(self):
            self.check([],(-1,7),'outside')
            self.check([(7,7)],(7,7),'occupied')

        def test_center_turn_undo(self):
            g = RenjuGame()
            self.assertEqual(g.legal_moves(),[(7,7)])
            self.assertEqual(g.play(0,0).code,'center')
            self.assertEqual(g.turn,BLACK)
            g.play(7,7)
            self.assertEqual(g.turn,WHITE)
            self.assertEqual(g.play(7,7).code,'occupied')
            self.assertEqual(g.turn,WHITE)
            self.assertTrue(g.undo())
            self.assertEqual(g.stone_count,0)
            self.assertEqual(g.board,create_board())

        def test_foul_blocked_without_state_change(self):
            cases = [
                ([(7,6),(7,8),(6,7),(8,7)], (7,7), 'double_three'),
                ([(7,5),(7,6),(7,8),(5,7),(6,7),(8,7)], (7,7), 'double_four'),
                ([(7,c) for c in (3,4,5,7,8)], (7,6), 'overline'),
            ]
            for stones, point, expected in cases:
                g = RenjuGame()
                g.board = board_with(stones)
                g.stone_count = len(stones)
                snapshot = repr(g.__dict__)
                self.assertEqual(g.forbidden_moves()[point], expected)
                self.assertNotIn(point, g.legal_moves())
                result = g.play(*point)
                self.assertFalse(result.legal)
                self.assertIsNone(result.winner)
                self.assertEqual(result.code, expected)
                self.assertEqual(repr(g.__dict__), snapshot)
                self.assertTrue(g.play(0,0).legal)
                self.assertEqual(g.turn, WHITE)
                self.assertEqual(g.forbidden_moves(), {})
                g.undo()
                self.assertEqual(g.forbidden_moves()[point], expected)

        def test_markers_hidden_and_wins_not_forbidden(self):
            g = RenjuGame()
            self.assertEqual(g.forbidden_moves(), {})
            stones = [(7,c) for c in (3,4,5,6)] + [(r,7) for r in (4,5,6,8,9)]
            g.board = board_with(stones)
            g.stone_count = len(stones)
            self.assertNotIn((7,7), g.forbidden_moves())
            self.assertEqual(g.play(7,7).winner, BLACK)
            self.assertEqual(g.forbidden_moves(), {})

        def test_pass_draw_and_undo(self):
            g = RenjuGame()
            self.assertFalse(g.pass_turn().legal)
            for p in [(7,7),(0,0),(7,8)]:
                g.play(*p)
            g.pass_turn()
            g.pass_turn()
            self.assertEqual(g.winner,EMPTY)
            g.undo()
            self.assertIsNone(g.winner)
            self.assertEqual(g.passes,1)

        def test_resign(self):
            g = RenjuGame()
            g.resign()
            self.assertEqual(g.winner,WHITE)
            g.undo()
            self.assertIsNone(g.winner)

        def test_symmetry(self):
            stones = [(7,6),(7,8),(6,7),(8,7)]
            for reflect in (False,True):
                for rotation in range(4):
                    def transform(p):
                        r,c = p
                        if reflect:
                            c=14-c
                        for _ in range(rotation):
                            r,c=c,14-r
                        return r,c
                    self.check([transform(p) for p in stones],transform((7,7)),'double_three')

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RulesTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    if '--test' in sys.argv:
        sys.exit(run_tests())
    try:
        launch_gui()
    except ImportError as exc:
        print('Tkinter가 필요합니다. Python 설치 프로그램의 Tcl/Tk 옵션을 확인하세요.')
        print(exc)
        sys.exit(1)
