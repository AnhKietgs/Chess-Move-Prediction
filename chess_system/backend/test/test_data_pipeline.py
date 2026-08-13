"""
Unit tests for the data pipeline (`src/data_processing`).

Scope, per the project's testing requirements:
    1. `fen_to_tensor` — output shape [C, 8, 8] and dtype torch.float32.
    2. `move_to_index` — returns a plain int, and raises for any move
       outside the defined 4672-move action space.

A couple of low-cost bonus tests are included for `dataset.py`'s
leakage-free game-level split, since "no data leakage between splits" was
an explicit requirement for that module and is cheap to verify without any
external dependencies (no PGN file, no Stockfish binary needed — these
tests never touch either).
"""

import chess
import pytest
import torch

from src.data_processing.encoder import (
    ACTION_SPACE_SIZE,
    NUM_CHANNELS,
    fen_to_tensor,
    index_to_move,
    move_to_index,
)
from src.data_processing.dataset import _Record, split_indices_by_game

# A fixed, non-trivial FEN used across several tests: mid-game, one side has
# lost castling rights on one wing, en passant is available, Black to move.
FIXED_FEN = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"


# ---------------------------------------------------------------------------
# fen_to_tensor
# ---------------------------------------------------------------------------

class TestFenToTensor:
    def test_shape_and_dtype(self):
        tensor = fen_to_tensor(FIXED_FEN)
        assert tensor.shape == (NUM_CHANNELS, 8, 8)
        assert tensor.dtype == torch.float32

    def test_starting_position_shape_and_dtype(self):
        # Cheap second data point: the encoder shouldn't depend on the
        # specific position for shape/dtype correctness.
        tensor = fen_to_tensor(chess.STARTING_FEN)
        assert tensor.shape == (NUM_CHANNELS, 8, 8)
        assert tensor.dtype == torch.float32

    def test_piece_channels_match_board(self):
        """Every occupied square in the FEN should be a 1.0 in exactly the
        channel matching its piece type/color, and 0.0 everywhere else on
        the piece-plane channels for that square."""
        tensor = fen_to_tensor(FIXED_FEN)
        board = chess.Board(FIXED_FEN)

        piece_channels = tensor[:12]  # channels 0-11 are the 12 piece planes
        occupied_count = int(piece_channels.sum().item())
        assert occupied_count == len(board.piece_map())

        # Spot-check one known piece: White king still on e1.
        white_king_channel = 5  # per _PIECE_TO_CHANNEL ordering P,N,B,R,Q,K
        e1_rank, e1_file = chess.square_rank(chess.E1), chess.square_file(chess.E1)
        assert tensor[white_king_channel, e1_rank, e1_file] == 1.0

    def test_side_to_move_channel(self):
        # FIXED_FEN has Black to move.
        tensor = fen_to_tensor(FIXED_FEN)
        assert torch.all(tensor[17] == 0.0)

        white_to_move_fen = FIXED_FEN.replace(" b ", " w ")
        tensor_white = fen_to_tensor(white_to_move_fen)
        assert torch.all(tensor_white[17] == 1.0)

    def test_castling_rights_channels(self):
        # No castling rights at all for either side.
        no_castling_fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
        tensor = fen_to_tensor(no_castling_fen)
        assert torch.all(tensor[12] == 0.0)  # white kingside
        assert torch.all(tensor[13] == 0.0)  # white queenside
        assert torch.all(tensor[14] == 0.0)  # black kingside
        assert torch.all(tensor[15] == 0.0)  # black queenside

        # FIXED_FEN grants all four rights ("KQkq").
        tensor_full = fen_to_tensor(FIXED_FEN)
        assert torch.all(tensor_full[12] == 1.0)
        assert torch.all(tensor_full[13] == 1.0)
        assert torch.all(tensor_full[14] == 1.0)
        assert torch.all(tensor_full[15] == 1.0)

    def test_invalid_fen_raises(self):
        with pytest.raises(ValueError):
            fen_to_tensor("this is not a fen string")


# ---------------------------------------------------------------------------
# move_to_index
# ---------------------------------------------------------------------------

class TestMoveToIndex:
    def test_returns_plain_int_within_action_space(self):
        move = chess.Move.from_uci("e2e4")
        index = move_to_index(move)
        assert isinstance(index, int)
        assert 0 <= index < ACTION_SPACE_SIZE

    @pytest.mark.parametrize(
        "uci",
        ["e2e4", "g1f3", "e1g1", "a7a8q", "a7a8n", "h7h8b"],
    )
    def test_various_legal_shaped_moves_map_into_range(self, uci):
        move = chess.Move.from_uci(uci)
        index = move_to_index(move)
        assert isinstance(index, int)
        assert 0 <= index < ACTION_SPACE_SIZE

    def test_raises_for_move_outside_action_space(self):
        """a1 -> c4 is not a straight line, diagonal, or knight shape —
        no real chess piece can move this way, so it must be rejected."""
        impossible_move = chess.Move(chess.A1, chess.C4)
        with pytest.raises(ValueError):
            move_to_index(impossible_move)

    def test_raises_for_zero_displacement_move(self):
        null_ish_move = chess.Move(chess.E4, chess.E4)
        with pytest.raises(ValueError):
            move_to_index(null_ish_move)

    def test_all_starting_position_legal_moves_are_encodable(self):
        board = chess.Board()
        for move in board.legal_moves:
            index = move_to_index(move)
            assert isinstance(index, int)
            assert 0 <= index < ACTION_SPACE_SIZE

    def test_round_trip_through_index_to_move(self):
        board = chess.Board()
        for move in board.legal_moves:
            index = move_to_index(move)
            decoded = index_to_move(index, board)
            assert decoded == move

    def test_index_to_move_raises_for_out_of_range_index(self):
        board = chess.Board()
        with pytest.raises(ValueError):
            index_to_move(ACTION_SPACE_SIZE, board)  # first invalid index
        with pytest.raises(ValueError):
            index_to_move(-1, board)


# ---------------------------------------------------------------------------
# dataset.split_indices_by_game — leakage-free splitting (bonus coverage)
# ---------------------------------------------------------------------------

class TestSplitIndicesByGame:
    @staticmethod
    def _fake_records(num_games: int, positions_per_game: int):
        records = []
        for game_id in range(num_games):
            for _ in range(positions_per_game):
                records.append(_Record(fen=chess.STARTING_FEN, move_uci="e2e4", game_id=game_id))
        return records

    def test_every_index_assigned_exactly_once(self):
        records = self._fake_records(num_games=20, positions_per_game=5)
        splits = split_indices_by_game(records)

        all_indices = splits["train"] + splits["val"] + splits["test"]
        assert sorted(all_indices) == list(range(len(records)))

    def test_no_game_appears_in_more_than_one_split(self):
        records = self._fake_records(num_games=20, positions_per_game=5)
        splits = split_indices_by_game(records)

        game_ids_per_split = {
            split_name: {records[i].game_id for i in indices}
            for split_name, indices in splits.items()
        }
        assert game_ids_per_split["train"].isdisjoint(game_ids_per_split["val"])
        assert game_ids_per_split["train"].isdisjoint(game_ids_per_split["test"])
        assert game_ids_per_split["val"].isdisjoint(game_ids_per_split["test"])

    def test_invalid_ratios_raise(self):
        records = self._fake_records(num_games=5, positions_per_game=2)
        with pytest.raises(ValueError):
            split_indices_by_game(records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.2)
